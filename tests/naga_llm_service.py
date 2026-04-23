"""
Standalone NAGA LLM Service for memory-system testing.
Does not depend on agents repo structure.
"""

import re
import json
import httpx
from typing import Tuple, Type, Any, Optional, List
from pydantic import BaseModel, ValidationError
import asyncio


class StandaloneNAGALLMService:
    """
    Standalone NAGA LLM Service that uses direct httpx calls.
    No dependencies on agents repo.
    """

    def __init__(
        self,
        model: str = 'sonar:free',
        api_base: str = 'https://api.naga.ac/v1',
        api_key: str = 'ng-4P9TvWqfrr1lORngNt5vMAAuoOGVouPR',
        embedding_model: str = 'text-embedding-3-small',
        max_retries: int = 3
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.max_retries = max_retries

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0
    ) -> Tuple[str, None]:
        """Generate text response."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers=self._get_headers(),
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'], None

    async def structured_output(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: str = None,
        enable_self_correction: bool = True
    ) -> Tuple[Any, None]:
        """Execute prompt with structured output validation and self-correction."""
        schema_json = schema.model_json_schema()
        schema_str = json.dumps(schema_json, indent=2)

        base_system = system_prompt or "You are a structured output assistant. Always respond with valid JSON."
        full_system = f"""{base_system}

OUTPUT SCHEMA (JSON):
```json
{schema_str}
```

IMPORTANT: Your response must be valid JSON matching the schema above. Do not include any other text.
"""

        last_error = None
        last_failed_json = None

        for attempt in range(self.max_retries):
            is_correction = attempt > 0

            if is_correction and last_error:
                correction_system = """You are a JSON repair expert. Fix the following validation error while preserving ALL original data.

RULES:
1. Keep ALL data from the original JSON
2. Fix ONLY syntax or format issues
3. Return ONLY the corrected JSON, no markdown or explanations
4. Ensure valid JSON with correct quotes
5. Do not invent new data, only repair the format"""

                correction_query = f"""JSON WITH ERROR:
{last_failed_json[:2000] if last_failed_json else prompt}

VALIDATION ERROR:
{last_error}

FIX IT keeping all original data."""

                current_system = correction_system
                current_prompt = correction_query
            else:
                current_system = full_system
                current_prompt = prompt

            try:
                response_text, _ = await self.generate(current_system, current_prompt, 0.0)
                cleaned = self._clean_response(response_text)
                parsed = json.loads(cleaned)
                result = schema(**parsed)
                return result, None

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)
                last_failed_json = cleaned if 'cleaned' in dir() else response_text

                if not enable_self_correction:
                    raise

            except Exception as e:
                last_error = str(e)
                if not enable_self_correction or attempt == self.max_retries - 1:
                    raise

        raise ValidationError(f"Failed after {self.max_retries} attempts. Last error: {last_error}")

    async def embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using direct HTTP call."""
        payload = {
            "model": self.embedding_model,
            "input": texts
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.api_base}/embeddings",
                headers=self._get_headers(),
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            if "data" in data:
                return [item["embedding"] for item in data["data"]]
            return []

    def embed(self, text: str) -> list[float]:
        """Synchronous version of embeddings for a single text."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.embeddings([text]))[0]
            finally:
                loop.close()
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._sync_embed, text)
                return future.result()

    def _sync_embed(self, text: str) -> list[float]:
        """Sync helper for embed."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.embeddings([text]))[0]
        finally:
            loop.close()

    def _clean_response(self, raw: str) -> str:
        """Clean LLM JSON response - extract JSON from markdown if needed."""
        raw = raw.strip()
        if raw.startswith('```json'):
            raw = raw[7:]
        elif raw.startswith('```'):
            raw = raw[3:]
        if raw.endswith('```'):
            raw = raw[:-3]
        json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', raw)
        return json_match.group(0) if json_match else raw.strip()


# Global instance
naga_llm = StandaloneNAGALLMService()