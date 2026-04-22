import asyncpg
import json
from typing import Optional
from datetime import datetime
from memory_system.config import settings
from memory_system.models import (
    WorkingMemorySnapshot,
    EpisodicEvent,
    SemanticMemory,
    MemoryDigest,
)


class PostgresService:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
            min_size=5,
            max_size=20,
        )

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def init_schema(self):
        sql_path = "/Users/neoris/Documents/projects/own/agentic/memory-system/sql/001_initial_schema.sql"
        with open(sql_path) as f:
            schema_sql = f.read()
        async with self.pool.acquire() as conn:
            await conn.execute(schema_sql)

    def _parse_json(self, value):
        if isinstance(value, str):
            return json.loads(value)
        return value if value is not None else {}

    # Working Memory Operations
    async def save_working_memory(self, snapshot: WorkingMemorySnapshot) -> str:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO working_memory_snapshots
                (user_id, agent_id, conversation_id, session_id, version, objective,
                active_tasks, constraints, open_questions, active_entities, active_references, summary)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING snapshot_id
                """,
                snapshot.user_id,
                snapshot.agent_id,
                snapshot.conversation_id,
                snapshot.session_id,
                snapshot.version,
                snapshot.objective,
                json.dumps(snapshot.active_tasks),
                json.dumps(snapshot.constraints),
                json.dumps(snapshot.open_questions),
                json.dumps(snapshot.active_entities),
                json.dumps(snapshot.active_references),
                snapshot.summary,
            )
            return str(row["snapshot_id"])

    async def get_latest_working_memory(
        self, conversation_id: str
    ) -> Optional[WorkingMemorySnapshot]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM working_memory_snapshots
                WHERE conversation_id = $1
                ORDER BY version DESC
                LIMIT 1
                """,
                conversation_id,
            )
            if row:
                return WorkingMemorySnapshot(
                    snapshot_id=str(row["snapshot_id"]),
                    user_id=str(row["user_id"]),
                    agent_id=str(row["agent_id"]),
                    conversation_id=str(row["conversation_id"]),
                    session_id=str(row["session_id"]),
                    version=row["version"],
                    objective=row["objective"],
                    active_tasks=self._parse_json(row["active_tasks"]),
                    constraints=self._parse_json(row["constraints"]),
                    open_questions=self._parse_json(row["open_questions"]),
                    active_entities=self._parse_json(row["active_entities"]),
                    active_references=self._parse_json(row["active_references"]),
                    summary=row["summary"],
                    created_at=row["created_at"],
                )
            return None

    # Episodic Memory Operations
    async def save_episodic_event(self, event: EpisodicEvent) -> str:
        async with self.pool.acquire() as conn:
            embedding = event.embedding
            embedding_param = embedding if embedding is None else str(embedding)
            row = await conn.fetchrow(
                """
                INSERT INTO episodic_events
                (user_id, agent_id, conversation_id, session_id, turn_id, role,
                event_type, content, normalized_content, metadata, salience_score, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::vector)
                RETURNING event_id
                """,
                event.user_id,
                event.agent_id,
                event.conversation_id,
                event.session_id,
                event.turn_id,
                event.role,
                event.event_type,
                event.content,
                event.normalized_content,
                json.dumps(event.metadata),
                event.salience_score,
                embedding_param,
            )
            return str(row["event_id"])

    async def get_episodic_events(
        self,
        conversation_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> list[EpisodicEvent]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT event_id, user_id, agent_id, conversation_id, session_id, turn_id,
                       role, event_type, content, normalized_content, metadata,
                       salience_score, embedding::text as embedding_str, created_at
                FROM episodic_events
                WHERE conversation_id = $1
                ORDER BY turn_id DESC
                LIMIT $2 OFFSET $3
                """,
                conversation_id,
                limit,
                offset,
            )
            return [
                EpisodicEvent(
                    event_id=str(row["event_id"]),
                    user_id=str(row["user_id"]),
                    agent_id=str(row["agent_id"]),
                    conversation_id=str(row["conversation_id"]),
                    session_id=str(row["session_id"]),
                    turn_id=row["turn_id"],
                    role=row["role"],
                    event_type=row["event_type"],
                    content=row["content"],
                    normalized_content=row["normalized_content"],
                    metadata=self._parse_json(row["metadata"]),
                    salience_score=row["salience_score"],
                    embedding=self._parse_json(row["embedding_str"]) if row["embedding_str"] else None,
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    async def search_episodic_by_text(
        self, user_id: str, query: str, limit: int = 10
    ) -> list[EpisodicEvent]:
        async with self.pool.acquire() as conn:
            words = query.split()[:5]
            if len(words) == 1:
                rows = await conn.fetch(
                    """
                    SELECT * FROM episodic_events
                    WHERE user_id = $1 AND content ILIKE $2
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    user_id,
                    f"%{words[0]}%",
                    limit,
                )
            elif len(words) == 2:
                rows = await conn.fetch(
                    """
                    SELECT * FROM episodic_events
                    WHERE user_id = $1 AND (content ILIKE $2 OR content ILIKE $3)
                    ORDER BY created_at DESC
                    LIMIT $4
                    """,
                    user_id,
                    f"%{words[0]}%",
                    f"%{words[1]}%",
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM episodic_events
                    WHERE user_id = $1 AND (content ILIKE $2 OR content ILIKE $3 OR content ILIKE $4)
                    ORDER BY created_at DESC
                    LIMIT $5
                    """,
                    user_id,
                    f"%{words[0]}%",
                    f"%{words[1]}%",
                    f"%{words[2]}%",
                    limit,
                )
            return [
                EpisodicEvent(
                    event_id=str(row["event_id"]),
                    user_id=str(row["user_id"]),
                    agent_id=str(row["agent_id"]),
                    conversation_id=str(row["conversation_id"]),
                    session_id=str(row["session_id"]),
                    turn_id=row["turn_id"],
                    role=row["role"],
                    event_type=row["event_type"],
                    content=row["content"],
                    normalized_content=row["normalized_content"],
                    metadata=self._parse_json(row["metadata"]),
                    salience_score=row["salience_score"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    async def search_episodic_by_vector(
        self, user_id: str, query_embedding: list[float], limit: int = 10
    ) -> list[EpisodicEvent]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT event_id, user_id, agent_id, conversation_id, session_id, turn_id,
                       role, event_type, content, normalized_content, metadata,
                       salience_score, embedding::text as embedding_str, created_at,
                       1 - (embedding <=> $2::vector) as similarity
                FROM episodic_events
                WHERE user_id = $1 AND embedding IS NOT NULL
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                user_id,
                str(query_embedding),
                limit,
            )
            return [
                EpisodicEvent(
                    event_id=str(row["event_id"]),
                    user_id=str(row["user_id"]),
                    agent_id=str(row["agent_id"]),
                    conversation_id=str(row["conversation_id"]),
                    session_id=str(row["session_id"]),
                    turn_id=row["turn_id"],
                    role=row["role"],
                    event_type=row["event_type"],
                    content=row["content"],
                    normalized_content=row["normalized_content"],
                    metadata=self._parse_json(row["metadata"]),
                    salience_score=row["salience_score"],
                    embedding=self._parse_json(row["embedding_str"]) if row["embedding_str"] else None,
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    # Semantic Memory Operations
    async def save_semantic_memory(self, memory: SemanticMemory) -> str:
        async with self.pool.acquire() as conn:
            embedding_param = memory.embedding if memory.embedding is None else str(memory.embedding)
            row = await conn.fetchrow(
                """
                INSERT INTO semantic_memories
                (user_id, agent_id, scope_type, scope_id, memory_type, subject,
                predicate, object_value, canonical_text, attributes, confidence,
                source_count, stability_class, status, supersedes_memory_id, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16::vector)
                RETURNING memory_id
                """,
                memory.user_id,
                memory.agent_id,
                memory.scope_type.value if hasattr(memory.scope_type, 'value') else memory.scope_type,
                memory.scope_id,
                memory.memory_type.value if hasattr(memory.memory_type, 'value') else memory.memory_type,
                memory.subject,
                memory.predicate,
                memory.object_value,
                memory.canonical_text,
                json.dumps(memory.attributes),
                memory.confidence,
                memory.source_count,
                memory.stability_class.value if hasattr(memory.stability_class, 'value') else memory.stability_class,
                memory.status.value if hasattr(memory.status, 'value') else memory.status,
                memory.supersedes_memory_id,
                embedding_param,
            )
            return str(row["memory_id"])

    async def get_active_semantic_memories(
        self, user_id: str, scope_type: Optional[str] = None, limit: int = 20
    ) -> list[SemanticMemory]:
        async with self.pool.acquire() as conn:
            if scope_type:
                rows = await conn.fetch(
                    """
                    SELECT * FROM semantic_memories
                    WHERE user_id = $1 AND status = 'active' AND scope_type = $2
                    ORDER BY confidence DESC, last_confirmed_at DESC NULLS LAST
                    LIMIT $3
                    """,
                    user_id,
                    scope_type,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM semantic_memories
                    WHERE user_id = $1 AND status = 'active'
                    ORDER BY confidence DESC, last_confirmed_at DESC NULLS LAST
                    LIMIT $2
                    """,
                    user_id,
                    limit,
                )
            return [self._row_to_semantic_memory(row) for row in rows]

    async def search_semantic_memories(
        self, user_id: str, query: str, limit: int = 10
    ) -> list[SemanticMemory]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM semantic_memories
                WHERE user_id = $1 AND status = 'active'
                AND (canonical_text ILIKE $2 OR subject ILIKE $2 OR predicate ILIKE $2)
                ORDER BY confidence DESC
                LIMIT $3
                """,
                user_id,
                f"%{query}%",
                limit,
            )
            return [self._row_to_semantic_memory(row) for row in rows]

    async def search_semantic_memories_vector(
        self, user_id: str, query_embedding: list[float], limit: int = 10
    ) -> list[SemanticMemory]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT memory_id, user_id, agent_id, scope_type, scope_id, memory_type,
                       subject, predicate, object_value, canonical_text, attributes,
                       confidence, source_count, stability_class, status,
                       supersedes_memory_id, embedding::text as embedding_str,
                       created_at, updated_at, last_confirmed_at,
                       1 - (embedding <=> $2::vector) as similarity
                FROM semantic_memories
                WHERE user_id = $1 AND status = 'active' AND embedding IS NOT NULL
                ORDER BY embedding <=> $2::vector
                LIMIT $3
                """,
                user_id,
                str(query_embedding),
                limit,
            )
            return [self._row_to_semantic_memory(row, embedding_str=row["embedding_str"]) for row in rows]

    async def update_memory_confidence(
        self, memory_id: str, new_confidence: float, increment_source: bool = True
    ) -> None:
        async with self.pool.acquire() as conn:
            if increment_source:
                await conn.execute(
                    """
                    UPDATE semantic_memories
                    SET confidence = $2,
                        source_count = source_count + 1,
                        last_confirmed_at = now(),
                        updated_at = now()
                    WHERE memory_id = $1
                    """,
                    memory_id,
                    min(new_confidence, 1.0),
                )
            else:
                await conn.execute(
                    """
                    UPDATE semantic_memories
                    SET confidence = $2, updated_at = now()
                    WHERE memory_id = $1
                    """,
                    memory_id,
                    new_confidence,
                )

    async def deprecate_memory(self, memory_id: str, supersedes_id: Optional[str] = None) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE semantic_memories
                SET status = 'deprecated',
                    supersedes_memory_id = $2,
                    updated_at = now()
                WHERE memory_id = $1
                """,
                memory_id,
                supersedes_id,
            )

    # Digest Operations
    async def save_digest(self, digest: MemoryDigest) -> str:
        async with self.pool.acquire() as conn:
            embedding_param = digest.embedding if digest.embedding is None else str(digest.embedding)
            row = await conn.fetchrow(
                """
                INSERT INTO memory_digests
                (user_id, conversation_id, period_type, period_start, period_end,
                title, summary, timeline, open_loops, decisions, entities, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::vector)
                RETURNING digest_id
                """,
                digest.user_id,
                digest.conversation_id,
                digest.period_type,
                digest.period_start,
                digest.period_end,
                digest.title,
                digest.summary,
                json.dumps(digest.timeline),
                json.dumps(digest.open_loops),
                json.dumps(digest.decisions),
                json.dumps(digest.entities),
                json.dumps(digest.metadata),
                embedding_param,
            )
            return str(row["digest_id"])

    async def get_recent_digests(
        self, user_id: str, limit: int = 5
    ) -> list[MemoryDigest]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM memory_digests
                WHERE user_id = $1
                ORDER BY period_start DESC
                LIMIT $2
                """,
                user_id,
                limit,
            )
            return [
                MemoryDigest(
                    digest_id=str(row["digest_id"]),
                    user_id=str(row["user_id"]),
                    conversation_id=str(row["conversation_id"]) if row["conversation_id"] else None,
                    period_type=row["period_type"],
                    period_start=row["period_start"],
                    period_end=row["period_end"],
                    title=row["title"],
                    summary=row["summary"],
                    timeline=self._parse_json(row["timeline"]),
                    open_loops=self._parse_json(row["open_loops"]),
                    decisions=self._parse_json(row["decisions"]),
                    entities=self._parse_json(row["entities"]),
                    metadata=self._parse_json(row["metadata"]),
                    embedding=row["embedding"].tolist() if row["embedding"] else None,
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def _row_to_semantic_memory(self, row, embedding_str: str = None) -> SemanticMemory:
        embedding = None
        if embedding_str is not None:
            embedding = self._parse_json(embedding_str)
        elif row.get("embedding") is not None:
            if isinstance(row["embedding"], str):
                embedding = self._parse_json(row["embedding"])
            else:
                embedding = row["embedding"].tolist() if hasattr(row["embedding"], 'tolist') else row["embedding"]
        return SemanticMemory(
            memory_id=str(row["memory_id"]),
            user_id=str(row["user_id"]),
            agent_id=str(row["agent_id"]),
            scope_type=row["scope_type"],
            scope_id=str(row["scope_id"]) if row["scope_id"] else None,
            memory_type=row["memory_type"],
            subject=row["subject"],
            predicate=row["predicate"],
            object_value=row["object_value"],
            canonical_text=row["canonical_text"],
            attributes=self._parse_json(row["attributes"]),
            confidence=row["confidence"],
            source_count=row["source_count"],
            stability_class=row["stability_class"],
            status=row["status"],
            supersedes_memory_id=str(row["supersedes_memory_id"]) if row["supersedes_memory_id"] else None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_confirmed_at=row["last_confirmed_at"],
            embedding=embedding,
        )
