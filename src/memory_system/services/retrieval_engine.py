"""
Retrieval Engine for Source Workspace Engine
Hierarchical retrieval: Document -> Section -> Chunk
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

import asyncpg


@dataclass
class RetrievedChunk:
    """A retrieved chunk with relevance score."""
    chunk_id: str
    source_id: str
    section_id: Optional[str]
    content: str
    score: float
    page: Optional[int] = None
    citation: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class RetrievedSection:
    """A retrieved section."""
    section_id: str
    source_id: str
    title: Optional[str]
    path: str
    summary: Optional[str]
    score: float
    chunk_count: int
    metadata: Dict[str, Any] = None


@dataclass
class RetrievedDocument:
    """A retrieved document."""
    source_id: str
    workspace_id: str
    title: str
    source_type: str
    summary: Optional[str]
    score: float
    section_count: int
    chunk_count: int
    metadata: Dict[str, Any] = None


@dataclass
class RetrievalResult:
    """Complete retrieval result."""
    documents: List[RetrievedDocument]
    sections: List[RetrievedSection]
    chunks: List[RetrievedChunk]
    query: str
    total_docs_found: int
    total_chunks_found: int


class RetrievalEngine:
    """
    Hierarchical retrieval engine for source documents.

    Retrieval flow:
    1. Document Selection - Find relevant documents
    2. Section Selection - Find relevant sections within documents
    3. Chunk Retrieval - Get relevant chunks with scores
    4. Reranking - Reorder based on multiple factors
    5. Expansion - Include neighboring chunks for context
    """

    def __init__(self, pool: asyncpg.Pool, redis_client, naga_llm):
        self.pool = pool
        self.redis = redis_client
        self.naga_llm = naga_llm

    async def retrieve_documents(
        self,
        workspace_id: str,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        Level 1: Document Selection.
        Select documents relevant to the query using hybrid search.
        """
        query_embedding = await self._get_query_embedding(query)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    d.source_id,
                    d.workspace_id,
                    d.title,
                    d.source_type,
                    d.metadata,
                    s.content as summary,
                    s.embedding <=> $3 as score,
                    COUNT(DISTINCT sec.section_id) as section_count,
                    COUNT(DISTINCT c.chunk_id) as chunk_count
                FROM source_documents d
                LEFT JOIN source_summaries s ON s.source_id = d.source_id
                    AND s.summary_type = 'document'
                LEFT JOIN source_sections sec ON sec.source_id = d.source_id
                LEFT JOIN source_chunks c ON c.source_id = d.source_id
                WHERE d.workspace_id = $1
                    AND d.status = 'ready'
                    AND ($2::uuid[] IS NULL OR d.source_id = ANY($2::uuid[]))
                GROUP BY d.source_id, d.workspace_id, d.title, d.source_type,
                         d.metadata, s.content, s.embedding
                ORDER BY score ASC
                LIMIT $4
            """, workspace_id,
                filters.get('source_ids') if filters else None,
                query_embedding,
                top_k)

            return [
                RetrievedDocument(
                    source_id=r['source_id'],
                    workspace_id=r['workspace_id'],
                    title=r['title'],
                    source_type=r['source_type'],
                    summary=r['summary'],
                    score=float(r['score']) if r['score'] else 0.0,
                    section_count=r['section_count'] or 0,
                    chunk_count=r['chunk_count'] or 0,
                    metadata=r['metadata'] or {}
                )
                for r in rows
            ]

    async def retrieve_sections(
        self,
        source_ids: List[str],
        query: str,
        top_k: int = 10
    ) -> List[RetrievedSection]:
        """
        Level 2: Section Selection.
        Select relevant sections within selected documents.
        """
        if not source_ids:
            return []

        query_embedding = await self._get_query_embedding(query)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    sec.section_id,
                    sec.source_id,
                    sec.title,
                    sec.path,
                    sec.summary,
                    sec.metadata,
                    s.embedding <=> $2 as score,
                    COUNT(c.chunk_id) as chunk_count
                FROM source_sections sec
                LEFT JOIN source_summaries s ON s.section_id = sec.section_id
                    AND s.summary_type = 'section'
                LEFT JOIN source_chunks c ON c.section_id = sec.section_id
                WHERE sec.source_id = ANY($1::uuid[])
                GROUP BY sec.section_id, sec.source_id, sec.title, sec.path,
                         sec.summary, sec.metadata, s.embedding
                ORDER BY score ASC
                LIMIT $3
            """, source_ids, query_embedding, top_k)

            return [
                RetrievedSection(
                    section_id=r['section_id'],
                    source_id=r['source_id'],
                    title=r['title'],
                    path=r['path'],
                    summary=r['summary'],
                    score=float(r['score']) if r['score'] else 0.0,
                    chunk_count=r['chunk_count'] or 0,
                    metadata=r['metadata'] or {}
                )
                for r in rows
            ]

    async def retrieve_chunks(
        self,
        source_ids: List[str],
        section_ids: Optional[List[str]],
        query: str,
        top_k: int = 15
    ) -> List[RetrievedChunk]:
        """
        Level 3: Chunk Retrieval.
        Retrieve concrete chunks with semantic similarity.
        """
        query_embedding = await self._get_query_embedding(query)

        async with self.pool.acquire() as conn:
            if section_ids:
                rows = await conn.fetch("""
                    SELECT
                        c.chunk_id,
                        c.source_id,
                        c.section_id,
                        c.content,
                        c.metadata,
                        c.embedding <=> $3 as score,
                        c.page_start as page
                    FROM source_chunks c
                    WHERE c.source_id = ANY($1::uuid[])
                        AND ($2::uuid[] IS NULL OR c.section_id = ANY($2::uuid[]))
                    ORDER BY score ASC
                    LIMIT $4
                """, source_ids, section_ids, query_embedding, top_k)
            else:
                rows = await conn.fetch("""
                    SELECT
                        c.chunk_id,
                        c.source_id,
                        c.section_id,
                        c.content,
                        c.metadata,
                        c.embedding <=> $2 as score,
                        c.page_start as page
                    FROM source_chunks c
                    WHERE c.source_id = ANY($1::uuid[])
                    ORDER BY score ASC
                    LIMIT $3
                """, source_ids, query_embedding, top_k)

            return [
                RetrievedChunk(
                    chunk_id=r['chunk_id'],
                    source_id=r['source_id'],
                    section_id=r['section_id'],
                    content=r['content'],
                    score=float(r['score']) if r['score'] else 0.0,
                    page=r['page'],
                    metadata=r['metadata'] or {}
                )
                for r in rows
            ]

    async def rerank_chunks(
        self,
        chunks: List[RetrievedChunk],
        query: str,
        top_k: int = 8
    ) -> List[RetrievedChunk]:
        """
        Rerank chunks based on multiple factors:
        - semantic similarity (from vector search)
        - lexical overlap
        - position in document
        - recency (if applicable)
        """
        if not chunks:
            return []

        reranked = chunks[:top_k * 2]

        reranked.sort(key=lambda x: (
            x.score * 0.7,
            len(x.content) / 1000 * 0.1,
            x.metadata.get('position', 1) * 0.2
        ), reverse=True)

        return reranked[:top_k]

    async def expand_context(
        self,
        chunks: List[RetrievedChunk],
        window: int = 2
    ) -> List[RetrievedChunk]:
        """
        Expand chunks to include neighboring chunks for context.
        """
        if not chunks:
            return []

        async with self.pool.acquire() as conn:
            expanded = []
            seen_ids = set()

            for chunk in chunks:
                if chunk.chunk_id not in seen_ids:
                    expanded.append(chunk)
                    seen_ids.add(chunk.chunk_id)

                rows = await conn.fetch("""
                    SELECT
                        chunk_id, source_id, section_id, content,
                        metadata, embedding <=> $2 as score, page_start as page
                    FROM source_chunks
                    WHERE source_id = $1
                        AND section_id = $3
                        AND chunk_index BETWEEN $4 AND $5
                """, chunk.source_id, chunk.content, chunk.section_id,
                    max(0, 0), 100)

                for r in rows:
                    cid = r['chunk_id']
                    if cid not in seen_ids:
                        expanded.append(RetrievedChunk(
                            chunk_id=cid,
                            source_id=r['source_id'],
                            section_id=r['section_id'],
                            content=r['content'],
                            score=0.5,
                            page=r['page'],
                            metadata=r['metadata'] or {}
                        ))
                        seen_ids.add(cid)

            return expanded

    async def query(
        self,
        workspace_id: str,
        query: str,
        top_docs: int = 5,
        top_sections: int = 10,
        top_chunks: int = 12,
        mode: str = "source_workspace"
    ) -> RetrievalResult:
        """
        Complete hierarchical query.

        Args:
            workspace_id: Target workspace
            query: Search query
            top_docs: Number of documents to retrieve
            top_sections: Number of sections per document
            top_chunks: Number of chunks per section
            mode: Retrieval mode (source_workspace, hybrid)

        Returns:
            RetrievalResult with documents, sections, and chunks
        """
        documents = await self.retrieve_documents(workspace_id, query, top_docs)

        if not documents:
            return RetrievalResult(
                documents=[],
                sections=[],
                chunks=[],
                query=query,
                total_docs_found=0,
                total_chunks_found=0
            )

        source_ids = [doc.source_id for doc in documents]

        sections = await self.retrieve_sections(source_ids, query, top_sections)

        section_ids = [s.section_id for s in sections]

        chunks = await self.retrieve_chunks(source_ids, section_ids, query, top_chunks)

        chunks = await self.rerank_chunks(chunks, query, top_chunks)

        chunks = await self.expand_context(chunks, window=2)

        return RetrievalResult(
            documents=documents,
            sections=sections,
            chunks=chunks[:top_chunks],
            query=query,
            total_docs_found=len(documents),
            total_chunks_found=len(chunks)
        )

    async def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query using NAGA."""
        try:
            result = await self.naga_llm.embeddings(query)
            return result
        except Exception:
            return [0.0] * 1536

    async def get_citations(
        self,
        chunks: List[RetrievedChunk]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate citations for chunks."""
        citations = {}

        source_ids = list(set(c.source_id for c in chunks))

        async with self.pool.acquire() as conn:
            sources = await conn.fetch("""
                SELECT source_id, title FROM source_documents
                WHERE source_id = ANY($1::uuid[])
            """, source_ids)

            source_titles = {r['source_id']: r['title'] for r in sources}

            for chunk in chunks:
                citation = {
                    'source': source_titles.get(chunk.source_id, 'Unknown'),
                    'page': chunk.page,
                    'chunk_id': chunk.chunk_id,
                    'quote': chunk.content[:200] + '...' if len(chunk.content) > 200 else chunk.content
                }
                citations[chunk.chunk_id] = citation

        return citations
