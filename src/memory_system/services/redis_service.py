import json
import redis.asyncio as redis
from typing import Optional
from datetime import timedelta
from memory_system.config import settings


class RedisService:
    def __init__(self):
        self.client: Optional[redis.Redis] = None

    async def connect(self):
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True,
        )

    async def close(self):
        if self.client:
            await self.client.close()

    def _session_key(self, session_id: str, suffix: str) -> str:
        return f"session:{session_id}:{suffix}"

    # Recent Turns Cache
    async def add_recent_turn(
        self, session_id: str, role: str, content: str, turn_id: int
    ) -> None:
        key = self._session_key(session_id, "recent_turns")
        turn_data = json.dumps({"role": role, "content": content, "turn_id": turn_id})
        pipe = self.client.pipeline()
        pipe.lpush(key, turn_data)
        pipe.ltrim(key, 0, 19)  # Keep last 20 turns
        pipe.expire(key, timedelta(hours=settings.active_context_ttl_hours))
        await pipe.execute()

    async def get_recent_turns(self, session_id: str, limit: int = 20) -> list[dict]:
        key = self._session_key(session_id, "recent_turns")
        turns = await self.client.lrange(key, 0, limit - 1)
        return [json.loads(turn) for turn in turns]

    async def clear_recent_turns(self, session_id: str) -> None:
        key = self._session_key(session_id, "recent_turns")
        await self.client.delete(key)

    # Hot Entities Cache
    async def update_hot_entities(self, session_id: str, entities: list[dict]) -> None:
        key = self._session_key(session_id, "entities_hot")
        await self.client.setex(
            key, timedelta(hours=settings.active_context_ttl_hours), json.dumps(entities)
        )

    async def get_hot_entities(self, session_id: str) -> list[dict]:
        key = self._session_key(session_id, "entities_hot")
        data = await self.client.get(key)
        if data:
            return json.loads(data)
        return []

    # Working Memory Cache
    async def cache_working_memory(
        self, session_id: str, working_memory: dict
    ) -> None:
        key = self._session_key(session_id, "working_cache")
        await self.client.setex(
            key, timedelta(hours=settings.working_memory_ttl_hours), json.dumps(working_memory)
        )

    async def get_cached_working_memory(self, session_id: str) -> Optional[dict]:
        key = self._session_key(session_id, "working_cache")
        data = await self.client.get(key)
        if data:
            return json.loads(data)
        return None

    async def invalidate_working_cache(self, session_id: str) -> None:
        key = self._session_key(session_id, "working_cache")
        await self.client.delete(key)

    # Retrieval Cache
    async def cache_retrieval_results(
        self, session_id: str, query_hash: str, results: dict
    ) -> None:
        key = self._session_key(session_id, f"retrieval_cache:{query_hash}")
        await self.client.setex(
            key, timedelta(hours=settings.retrieval_cache_ttl_hours), json.dumps(results)
        )

    async def get_cached_retrieval(
        self, session_id: str, query_hash: str
    ) -> Optional[dict]:
        key = self._session_key(session_id, f"retrieval_cache:{query_hash}")
        data = await self.client.get(key)
        if data:
            return json.loads(data)
        return None

    # Session Lock
    async def acquire_session_lock(
        self, session_id: str, lock_id: str, ttl_seconds: int = 30
    ) -> bool:
        key = self._session_key(session_id, "lock")
        acquired = await self.client.set(key, lock_id, nx=True, ex=ttl_seconds)
        return bool(acquired)

    async def release_session_lock(self, session_id: str, lock_id: str) -> None:
        key = self._session_key(session_id, "lock")
        current = await self.client.get(key)
        if current == lock_id:
            await self.client.delete(key)

    # Full session cleanup
    async def clear_session(self, session_id: str) -> None:
        pattern = f"session:{session_id}:*"
        cursor = 0
        while True:
            cursor, keys = await self.client.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                await self.client.delete(*keys)
            if cursor == 0:
                break
