import httpx
import json
from typing import Optional
from memory_system.config import settings


class NAGAEmbeddingService:
    def __init__(self):
        self.api_key = settings.naga_api_key
        self.model = "sonar:free"
        self.embedding_dim = settings.default_embedding_dim
        self._cache: dict[str, list[float]] = {}

    async def get_embedding(self, text: str, use_cache: bool = True) -> list[float]:
        if use_cache and text in self._cache:
            return self._cache[text]

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.ng+1.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "input": text,
                    },
                )
                response.raise_for_status()
                data = response.json()
                embedding = data["data"][0]["embedding"]
                if use_cache:
                    self._cache[text] = embedding
                return embedding
        except Exception as e:
            return self._fallback_embedding(text)

    async def get_embeddings_batch(
        self, texts: list[str], use_cache: bool = True
    ) -> list[list[float]]:
        results = []
        for text in texts:
            results.append(await self.get_embedding(text, use_cache))
        return results

    def _fallback_embedding(self, text: str) -> list[float]:
        import hashlib
        import struct

        text_hash = hashlib.sha256(text.encode()).digest()
        seed = struct.unpack("<Q", text_hash[:8])[0]
        import random
        random.seed(seed)
        return [random.uniform(-1, 1) for _ in range(self.embedding_dim)]

    def clear_cache(self):
        self._cache.clear()
