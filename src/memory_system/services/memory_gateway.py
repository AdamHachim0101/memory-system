import json
import hashlib
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime, timedelta
from memory_system.models import (
    IngestTurnRequest,
    IngestTurnResponse,
    RetrieveRequest,
    RetrieveResponse,
    WorkingMemorySnapshot,
    EpisodicEvent,
    SemanticMemory,
    MemoryDigest,
    MemoryCandidate,
    ExtractionResult,
    EventType,
    MemoryStatus,
    MemoryType,
    StabilityClass,
    RetrievalProfile,
)
from memory_system.services.postgres_service import PostgresService
from memory_system.services.redis_service import RedisService
from memory_system.services.neo4j_service import Neo4jService
from memory_system.services.embedding_service import NAGAEmbeddingService

if TYPE_CHECKING:
    from memory_system.services.topic_tracking_service import TopicTrackingService
    from memory_system.services.digest_service import DigestService


class MemoryGateway:
    def __init__(
        self,
        postgres: PostgresService,
        redis: RedisService,
        neo4j: Neo4jService,
        embedding_service: NAGAEmbeddingService,
    ):
        self.postgres = postgres
        self.redis = redis
        self.neo4j = neo4j
        self.embedding = embedding_service
        self._topic_tracking: Optional["TopicTrackingService"] = None
        self._digest_service: Optional["DigestService"] = None
        self._last_summary_turn: int = -1
        self._conversation_topics: List[str] = []

    def _get_topic_tracking(self) -> "TopicTrackingService":
        """Lazy initialization of topic tracking service."""
        if self._topic_tracking is None:
            if self.postgres is None or self.postgres.pool is None:
                raise RuntimeError("Cannot initialize TopicTrackingService: postgres not available")
            from memory_system.services.topic_tracking_service import TopicTrackingService
            self._topic_tracking = TopicTrackingService(self.postgres.pool, self.embedding)
        return self._topic_tracking

    def _get_digest_service(self) -> "DigestService":
        """Lazy initialization of digest service."""
        if self._digest_service is None:
            if self.postgres is None or self.postgres.pool is None:
                raise RuntimeError("Cannot initialize DigestService: postgres not available")
            from memory_system.services.digest_service import DigestService
            self._digest_service = DigestService(self.postgres.pool, self.embedding)
        return self._digest_service

    async def ingest_turn(self, request: IngestTurnRequest) -> IngestTurnResponse:
        with open("/tmp/memory_debug.log", "a") as f:
            f.write(f"MEMORY_GATEWAY_INGEST_ENTER|turn={request.turn_id}|msg={request.user_message[:50]}\n")
            f.flush()
        working_snapshot_id = ""
        episodic_event_ids = []
        candidate_count = 0

        # 1. Save episodic events
        user_event = EpisodicEvent(
            user_id=request.user_id,
            agent_id=request.agent_id,
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
            role="user",
            event_type=EventType.MESSAGE.value,
            content=request.user_message,
            metadata=request.metadata,
            salience_score=self._estimate_salience(request.user_message),
        )
        user_event.embedding = await self.embedding.get_embedding(request.user_message)
        user_event_id = await self.postgres.save_episodic_event(user_event)
        episodic_event_ids.append(user_event_id)

        assistant_event = EpisodicEvent(
            user_id=request.user_id,
            agent_id=request.agent_id,
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            turn_id=request.turn_id,
            role="assistant",
            event_type=EventType.MESSAGE.value,
            content=request.assistant_message,
            metadata={},
            salience_score=self._estimate_salience(request.assistant_message),
        )
        assistant_event_id = await self.postgres.save_episodic_event(assistant_event)
        episodic_event_ids.append(assistant_event_id)

        # 2. Save tool events if any
        for tool_event in request.tool_events:
            tool_event_obj = EpisodicEvent(
                user_id=request.user_id,
                agent_id=request.agent_id,
                conversation_id=request.conversation_id,
                session_id=request.session_id,
                turn_id=request.turn_id,
                role="tool",
                event_type=EventType.TOOL_RESULT.value,
                content=json.dumps(tool_event),
                metadata={"tool_name": tool_event.get("tool", "unknown")},
                salience_score=0.5,
            )
            tool_event_id = await self.postgres.save_episodic_event(tool_event_obj)
            episodic_event_ids.append(tool_event_id)

        # 3. Cache recent turns in Redis
        await self.redis.add_recent_turn(
            request.session_id, "user", request.user_message, request.turn_id
        )
        await self.redis.add_recent_turn(
            request.session_id, "assistant", request.assistant_message, request.turn_id
        )

        # 4. Extract and save memory candidates
        extraction = await self.extract_memories(request)
        for candidate in extraction.memory_candidates:
            semantic_memory = SemanticMemory(
                user_id=request.user_id,
                agent_id=request.agent_id,
                scope_type="conversation",
                scope_id=request.conversation_id,
                memory_type=candidate.memory_type,
                subject=candidate.subject,
                predicate=candidate.predicate,
                object_value=candidate.object_value,
                canonical_text=candidate.canonical_text,
                attributes=candidate.attributes,
                confidence=candidate.confidence,
                source_count=1,
                stability_class=candidate.stability_class,
                status=MemoryStatus.CANDIDATE,
            )
            semantic_memory.embedding = await self.embedding.get_embedding(
                candidate.canonical_text
            )
            await self.postgres.save_semantic_memory(semantic_memory)
            candidate_count += 1

        # 5. Update working memory
        working_snapshot = await self.build_working_memory(request, extraction)
        working_snapshot_id = await self.postgres.save_working_memory(working_snapshot)
        await self.redis.cache_working_memory(
            request.session_id, working_snapshot.model_dump(mode='json')
        )

        # 6. Track topics and generate summaries
        await self._track_topics_and_summarize(request)

        # 7. Update graph nodes in Neo4j
        await self._update_graph_async(request, extraction)

        return IngestTurnResponse(
            working_snapshot_id=working_snapshot_id,
            episodic_event_ids=episodic_event_ids,
            candidate_count=candidate_count,
        )

    async def _track_topics_and_summarize(self, request: IngestTurnRequest) -> None:
        """Track topics and generate summaries periodically."""
        import sys
        sys.stderr.write(f"DEBUG _track_topics_and_summarize called for turn {request.turn_id}\n")
        sys.stderr.flush()
        try:
            if self.postgres is None or self.postgres.pool is None:
                sys.stderr.write("⚠️ Topic tracking skipped: postgres pool not available\n")
                return
            sys.stderr.write(f"DEBUG: postgres.pool is available, getting topic_tracking\n")
            sys.stderr.flush()
            topic_tracking = self._get_topic_tracking()
            
            # Detect and track topics
            topics = await topic_tracking.detect_and_track_topics(
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                turn_id=request.turn_id,
                user_message=request.user_message,
                assistant_message=request.assistant_message
            )
            
            # Track topic changes
            old_topics = self._conversation_topics
            new_topic_names = [t.topic_name for t in topics]
            
            # Check for topic shift
            topic_shift = False
            for new_topic in new_topic_names:
                if new_topic not in old_topics:
                    topic_shift = True
                    break
            
            self._conversation_topics = new_topic_names
            
            # Generate summary if needed
            if self._should_generate_summary(request.turn_id):
                digest_service = self._get_digest_service()
                events = await self._get_recent_events(
                    request.conversation_id,
                    self._last_summary_turn + 1,
                    request.turn_id
                )
                if events:
                    summary_id = await digest_service.generate_summary(
                        user_id=request.user_id,
                        conversation_id=request.conversation_id,
                        session_id=request.session_id,
                        turn_start=self._last_summary_turn + 1,
                        turn_end=request.turn_id,
                        use_llm=False
                    )
                    if summary_id:
                        print(f"💾 Generated conversation summary: {summary_id}")
                self._last_summary_turn = request.turn_id
            
            # Link events to topics
            for topic in topics:
                for event_id in [request.user_message]:  # Simplified - would track actual event IDs
                    pass  # Topic linking happens in topic_tracking.detect_and_track_topics
                    
        except Exception as e:
            import traceback
            print(f"❌ Topic tracking/summary error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            # Re-raise for debugging - remove once topic tracking is stable
            raise

    def _should_generate_summary(self, current_turn: int) -> bool:
        """Check if we should generate a summary now."""
        MIN_TURNS_BETWEEN_SUMMARIES = 5
        if current_turn - self._last_summary_turn >= MIN_TURNS_BETWEEN_SUMMARIES:
            return True
        return False

    async def _get_recent_events(
        self,
        conversation_id: str,
        turn_start: int,
        turn_end: int
    ) -> List[dict]:
        """Get conversation events for a turn range."""
        async with self.postgres.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT turn_id, role, content 
                FROM episodic_events 
                WHERE conversation_id = $1 AND turn_id BETWEEN $2 AND $3
                ORDER BY turn_id, created_at
                """,
                conversation_id,
                turn_start,
                turn_end
            )
            return [
                {"turn_id": r["turn_id"], "role": r["role"], "content": r["content"]}
                for r in rows
            ]

    async def extract_memories(self, request: IngestTurnRequest) -> ExtractionResult:
        entities = self._extract_entities(request.user_message + " " + request.assistant_message)
        candidates = self._generate_candidates(request, entities)
        working_patch = self._build_working_patch(request, entities)

        return ExtractionResult(
            working_memory_patch=working_patch,
            memory_candidates=candidates,
            entities=entities,
            salience_score=self._estimate_salience(request.user_message),
        )

    def _extract_entities(self, text: str) -> list[dict]:
        entities = []
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                entities.append({"type": "entity", "value": word})
        technology_keywords = ["Redis", "Postgres", "Neo4j", "API", "memory", "agent"]
        for kw in technology_keywords:
            if kw.lower() in text.lower():
                entities.append({"type": "technology", "value": kw})
        return entities

    def _generate_candidates(
        self, request: IngestTurnRequest, entities: list[dict]
    ) -> list[MemoryCandidate]:
        candidates = []
        text = request.user_message + " " + request.assistant_message

        preference_signals = ["prefiero", "me gusta", "prefiero", "siempre", "habitualmente"]
        for signal in preference_signals:
            if signal in text.lower():
                candidates.append(
                    MemoryCandidate(
                        memory_type=MemoryType.USER_PREFERENCE,
                        subject="user",
                        predicate="prefers",
                        object_value=text[:100],
                        canonical_text=f"User preference expressed: {text[:150]}",
                        stability_class=StabilityClass.DURABLE,
                        confidence=0.7,
                    )
                )

        decision_signals = ["decidí", "hemos decidido", "acordamos", "vamos a", "elegí"]
        for signal in decision_signals:
            if signal in text.lower():
                candidates.append(
                    MemoryCandidate(
                        memory_type=MemoryType.DECISION,
                        subject="conversation",
                        predicate="decided",
                        object_value=text[:100],
                        canonical_text=f"Decision made: {text[:150]}",
                        stability_class=StabilityClass.PERMANENT,
                        confidence=0.8,
                    )
                )

        project_signals = ["proyecto", "implementar", "desarrollar", "construir"]
        for signal in project_signals:
            if signal in text.lower():
                candidates.append(
                    MemoryCandidate(
                        memory_type=MemoryType.PROJECT_FACT,
                        subject="project",
                        predicate="involves",
                        object_value=text[:100],
                        canonical_text=f"Project related: {text[:150]}",
                        stability_class=StabilityClass.DURABLE,
                        confidence=0.6,
                    )
                )

        if entities and len(candidates) == 0:
            for entity in entities[:3]:
                candidates.append(
                    MemoryCandidate(
                        memory_type=MemoryType.DOMAIN_FACT,
                        subject=entity.get("type", "entity"),
                        predicate="mentioned",
                        object_value=entity.get("value", ""),
                        canonical_text=f"Entity mentioned: {entity.get('value', '')} ({entity.get('type', 'unknown')})",
                        stability_class=StabilityClass.VOLATILE,
                        confidence=0.5,
                    )
                )

        return candidates

    def _build_working_patch(
        self, request: IngestTurnRequest, entities: list[dict]
    ) -> dict:
        return {
            "objective": request.user_message[:100],
            "active_tasks": [],
            "constraints": [],
            "open_questions": [],
            "active_entities": entities[:5],
            "active_references": [],
            "summary": request.user_message[:200],
        }

    async def build_working_memory(
        self, request: IngestTurnRequest, extraction: ExtractionResult
    ) -> WorkingMemorySnapshot:
        existing = await self.postgres.get_latest_working_memory(request.conversation_id)
        version = (existing.version + 1) if existing else 1

        patch = extraction.working_memory_patch
        active_entities = patch.get("active_entities", [])
        for entity in extraction.entities:
            if entity not in active_entities:
                active_entities.append(entity)

        return WorkingMemorySnapshot(
            user_id=request.user_id,
            agent_id=request.agent_id,
            conversation_id=request.conversation_id,
            session_id=request.session_id,
            version=version,
            objective=patch.get("objective", existing.objective if existing else None),
            active_tasks=patch.get("active_tasks", existing.active_tasks if existing else []),
            constraints=patch.get("constraints", existing.constraints if existing else []),
            open_questions=patch.get("open_questions", existing.open_questions if existing else []),
            active_entities=active_entities,
            active_references=patch.get("active_references", []),
            summary=patch.get("summary", f"Turn {request.turn_id}: {request.user_message[:100]}"),
        )

    async def _update_graph_async(self, request: IngestTurnRequest, extraction: ExtractionResult) -> None:
        try:
            await self.neo4j.ensure_user_node(request.user_id)
            await self.neo4j.ensure_conversation_node(request.conversation_id, request.user_id)
            await self.neo4j.ensure_session_node(request.session_id, request.conversation_id)

            for entity in extraction.entities[:5]:
                await self.neo4j.add_entity_mention(
                    entity.get("type", "unknown"),
                    entity.get("value", ""),
                    request.user_message[:200],
                    request.conversation_id,
                )
        except Exception:
            pass

    def _estimate_salience(self, text: str) -> float:
        high_importance_signals = [
            "decisión", "decidir", "acordado", "confirmado", "importante",
            "recordar", "preferencio", "nunca", "siempre", "requiere",
        ]
        score = 0.3
        text_lower = text.lower()
        for signal in high_importance_signals:
            if signal in text_lower:
                score += 0.1
        return min(score, 1.0)

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        query_hash = hashlib.md5(request.query.encode()).hexdigest()

        cached = await self.redis.get_cached_retrieval(request.session_id, query_hash)
        if cached:
            return RetrieveResponse(**cached)

        working_memory = await self._get_working_memory(request)
        semantic_memories = await self._retrieve_semantic_memories(request)
        episodic_events = await self._retrieve_episodic_events(request)
        digests = await self._retrieve_digests(request)
        graph_context = await self._retrieve_graph_context(request)
        conflicts = await self._detect_conflicts(request)

        response = RetrieveResponse(
            working_memory=working_memory,
            semantic_memories=semantic_memories,
            episodic_events=episodic_events,
            digests=digests,
            graph_context=graph_context,
            conflicts=conflicts,
        )

        try:
            await self.redis.cache_retrieval_results(
                request.session_id, query_hash, response.model_dump(mode='json')
            )
        except Exception:
            pass

        return response

    async def _get_working_memory(self, request: RetrieveRequest) -> Optional[WorkingMemorySnapshot]:
        cached = await self.redis.get_cached_working_memory(request.session_id)
        if cached:
            return WorkingMemorySnapshot(**cached)
        return await self.postgres.get_latest_working_memory(request.conversation_id)

    async def _retrieve_semantic_memories(self, request: RetrieveRequest) -> list[SemanticMemory]:
        text_results = await self.postgres.search_semantic_memories(
            request.user_id, request.query, limit=request.limit
        )

        query_embedding = await self.embedding.get_embedding(request.query)
        vector_results = await self.postgres.search_semantic_memories_vector(
            request.user_id, query_embedding, limit=request.limit
        )

        seen_ids = set()
        combined = []
        for r in text_results + vector_results:
            if r.memory_id not in seen_ids:
                seen_ids.add(r.memory_id)
                combined.append(r)

        if len(combined) < 3:
            recent = await self.postgres.get_active_semantic_memories(
                request.user_id, limit=request.limit
            )
            for r in recent:
                if r.memory_id not in seen_ids:
                    seen_ids.add(r.memory_id)
                    combined.append(r)

        return combined[: request.limit]

    async def _retrieve_episodic_events(self, request: RetrieveRequest) -> list[EpisodicEvent]:
        text_results = await self.postgres.search_episodic_by_text(
            request.user_id, request.query, limit=request.limit
        )

        query_embedding = await self.embedding.get_embedding(request.query)
        vector_results = await self.postgres.search_episodic_by_vector(
            request.user_id, query_embedding, limit=request.limit
        )

        seen_ids = set()
        combined = []
        for r in text_results + vector_results:
            if r.event_id not in seen_ids:
                seen_ids.add(r.event_id)
                combined.append(r)

        # Always include recent events from conversation for context continuity
        # This ensures name/intent queries work even if search doesn't match
        fallback = await self.postgres.get_episodic_events(
            request.conversation_id, limit=request.limit
        )
        for r in fallback:
            if r.event_id not in seen_ids:
                seen_ids.add(r.event_id)
                combined.append(r)

        return combined[: request.limit]

    async def _retrieve_digests(self, request: RetrieveRequest) -> list[MemoryDigest]:
        return await self.postgres.get_recent_digests(request.user_id, limit=3)

    async def _retrieve_graph_context(self, request: RetrieveRequest) -> list[dict]:
        results = []
        for entity_name in request.entities[:3]:
            context = await self.neo4j.get_entity_context("entity", entity_name)
            results.extend(context)
        return results[:10]

    async def _detect_conflicts(self, request: RetrieveRequest) -> list[dict]:
        return []