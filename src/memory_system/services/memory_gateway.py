import json
import hashlib
from typing import Optional
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

    async def ingest_turn(self, request: IngestTurnRequest) -> IngestTurnResponse:
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

        # 6. Update graph nodes in Neo4j
        await self._update_graph_async(request, extraction)

        return IngestTurnResponse(
            working_snapshot_id=working_snapshot_id,
            episodic_event_ids=episodic_event_ids,
            candidate_count=candidate_count,
        )

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

        if len(combined) < 5:
            fallback = await self.postgres.get_episodic_events(
                request.conversation_id, limit=request.limit
            )
            for r in fallback:
                if r.event_id not in seen_ids:
                    seen_ids.add(r.event_id)
                    combined.append(r)

        if len(combined) < 3:
            all_recent = await self.postgres.get_episodic_events(
                request.conversation_id, limit=20
            )
            for r in all_recent:
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