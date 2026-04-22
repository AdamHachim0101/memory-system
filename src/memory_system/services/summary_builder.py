"""
Summary Builder for Source Workspace Engine
Generates hierarchical summaries: document, section, executive, entities, topic_map
"""

import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

import asyncpg


class SummaryType(Enum):
    DOCUMENT = "document"
    SECTION = "section"
    EXECUTIVE = "executive"
    ENTITIES = "entities"
    TOPIC_MAP = "topic_map"


@dataclass
class Summary:
    """A generated summary."""
    summary_id: str
    source_id: str
    section_id: Optional[str]
    summary_type: str
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]


class SummaryBuilder:
    """
    Builds hierarchical summaries for documents.

    Summary types:
    - document: High-level summary of the entire document
    - section: Summary of each section
    - executive: Key takeaways in bullet points
    - entities: Extracted entities and their mentions
    - topic_map: Topics and their relationships
    """

    def __init__(self, pool: asyncpg.Pool, naga_llm):
        self.pool = pool
        self.naga_llm = naga_llm

    async def build_document_summary(
        self,
        source_id: str,
        chunks: Optional[List[Dict[str, Any]]] = None
    ) -> Summary:
        """
        Build a document-level summary.

        Uses the first few chunks to generate a summary.
        """
        if not chunks:
            chunks = await self._get_chunks(source_id, limit=10)

        if not chunks:
            content = "No content available for summary."
        else:
            combined = "\n\n".join([c['content'] for c in chunks[:10]])
            content = await self._generate_summary(
                combined,
                "Summarize this document in 2-3 sentences:"
            )

        embedding = await self.naga_llm.embeddings(content)

        summary_id = str(uuid.uuid4())

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO source_summaries
                (summary_id, source_id, summary_type, content, embedding)
                VALUES ($1, $2, $3, $4, $5)
            """, summary_id, source_id, SummaryType.DOCUMENT.value, content, embedding)

        return Summary(
            summary_id=summary_id,
            source_id=source_id,
            section_id=None,
            summary_type=SummaryType.DOCUMENT.value,
            content=content,
            embedding=embedding,
            metadata={}
        )

    async def build_section_summary(
        self,
        section_id: str,
        source_id: str
    ) -> Summary:
        """Build a section-level summary."""
        async with self.pool.acquire() as conn:
            chunks = await conn.fetch("""
                SELECT content FROM source_chunks
                WHERE section_id = $1
                ORDER BY chunk_index
            """, section_id)

        if not chunks:
            content = "No content in section."
        else:
            combined = "\n\n".join([c['content'] for c in chunks])
            content = await self._generate_summary(
                combined,
                "Summarize this section in 1-2 sentences:"
            )

        embedding = await self.naga_llm.embeddings(content)
        summary_id = str(uuid.uuid4())

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO source_summaries
                (summary_id, source_id, section_id, summary_type, content, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, summary_id, source_id, section_id, SummaryType.SECTION.value,
                content, embedding)

        return Summary(
            summary_id=summary_id,
            source_id=source_id,
            section_id=section_id,
            summary_type=SummaryType.SECTION.value,
            content=content,
            embedding=embedding,
            metadata={}
        )

    async def build_executive_summary(
        self,
        source_id: str
    ) -> Summary:
        """Build an executive summary with key takeaways."""
        chunks = await self._get_chunks(source_id, limit=20)

        if not chunks:
            content = "No content for executive summary."
        else:
            combined = "\n\n".join([c['content'] for c in chunks[:20]])
            content = await self._generate_summary(
                combined,
                """Generate an executive summary with:
- Key findings (3-5 bullet points)
- Main conclusions
- Action items if applicable
Format as structured text."""
            )

        embedding = await self.naga_llm.embeddings(content)
        summary_id = str(uuid.uuid4())

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO source_summaries
                (summary_id, source_id, summary_type, content, embedding)
                VALUES ($1, $2, $3, $4, $5)
            """, summary_id, source_id, SummaryType.EXECUTIVE.value, content, embedding)

        return Summary(
            summary_id=summary_id,
            source_id=source_id,
            section_id=None,
            summary_type=SummaryType.EXECUTIVE.value,
            content=content,
            embedding=embedding,
            metadata={}
        )

    async def build_entities_summary(
        self,
        source_id: str
    ) -> Summary:
        """Build a summary of extracted entities."""
        async with self.pool.acquire() as conn:
            entities = await conn.fetch("""
                SELECT entity_type, canonical_name, COUNT(*) as count
                FROM source_entities
                WHERE source_id = $1
                GROUP BY entity_type, canonical_name
                ORDER BY count DESC
                LIMIT 50
            """, source_id)

        if not entities:
            content = "No entities extracted from document."
        else:
            entity_list = [f"{e['canonical_name']} ({e['entity_type']}): {e['count']} mentions"
                           for e in entities]
            content = "Entities found in document:\n\n" + "\n".join(entity_list)

        embedding = await self.naga_llm.embeddings(content)
        summary_id = str(uuid.uuid4())

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO source_summaries
                (summary_id, source_id, summary_type, content, embedding)
                VALUES ($1, $2, $3, $4, $5)
            """, summary_id, source_id, SummaryType.ENTITIES.value, content, embedding)

        return Summary(
            summary_id=summary_id,
            source_id=source_id,
            section_id=None,
            summary_type=SummaryType.ENTITIES.value,
            content=content,
            embedding=embedding,
            metadata={}
        )

    async def build_all_summaries(self, source_id: str) -> List[Summary]:
        """Build all summary types for a document."""
        summaries = []

        doc_summary = await self.build_document_summary(source_id)
        summaries.append(doc_summary)

        exec_summary = await self.build_executive_summary(source_id)
        summaries.append(exec_summary)

        entities_summary = await self.build_entities_summary(source_id)
        summaries.append(entities_summary)

        async with self.pool.acquire() as conn:
            sections = await conn.fetch("""
                SELECT section_id FROM source_sections
                WHERE source_id = $1
            """, source_id)

        for section in sections:
            sec_summary = await self.build_section_summary(
                section['section_id'],
                source_id
            )
            summaries.append(sec_summary)

        return summaries

    async def _get_chunks(
        self,
        source_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get chunks for a source."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT chunk_id, content, chunk_index
                FROM source_chunks
                WHERE source_id = $1
                ORDER BY chunk_index
                LIMIT $2
            """, source_id, limit)

            return [
                {'chunk_id': r['chunk_id'], 'content': r['content']}
                for r in rows
            ]

    async def _generate_summary(
        self,
        content: str,
        instruction: str
    ) -> str:
        """Generate summary using NAGA LLM."""
        prompt = f"{instruction}\n\n{content[:4000]}"

        try:
            result = await self.naga_llm.generate(prompt)
            return result.strip()
        except Exception:
            return f"Summary generation failed for: {content[:100]}..."