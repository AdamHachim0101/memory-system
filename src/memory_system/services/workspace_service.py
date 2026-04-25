"""
Source Service - Main entry point for Source Workspace Engine
Integrates all components: registry, storage, ingestion, retrieval, citation
"""

import uuid
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json
import asyncio

import asyncpg
import redis.asyncio as redis

from memory_system.config import settings
from memory_system.services.source_registry import SourceRegistry, SourceDocument, Workspace
from memory_system.services.minio_service import MinIOService
from memory_system.services.retrieval_engine import RetrievalEngine, RetrievalResult
from memory_system.services.citation_engine import CitationEngine, GroundedEvidence
from memory_system.services.prompt_composer import (
    PromptComposer, AgentMemoryContext, SourceEvidenceContext, HybridPrompt
)
from memory_system.services.minio_notification import MinIONotificationService
from memory_system.workers.ingestion_worker import IngestionWorker, IngestionTask, WorkerType
from memory_system.parsers import ParserFactory


@dataclass
class WorkspaceQueryResult:
    """Result of a workspace query."""
    response: str
    documents: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    citations: Dict[str, Any]
    mode: str
    tokens_used: int


class SourceWorkspaceService:
    """
    Main service for Source Workspace Engine.

    Provides:
    - Workspace management
    - Source registration and upload
    - Async ingestion processing
    - Query with retrieval and grounding
    - Hybrid queries (agent memory + source evidence)
    """

    def __init__(
        self,
        postgres_pool: asyncpg.Pool,
        redis_client: redis.Redis,
        naga_llm
    ):
        self.pool = postgres_pool
        self.redis = redis_client
        self.naga_llm = naga_llm

        self.source_registry = SourceRegistry(postgres_pool)
        self.minio_service = MinIOService()
        self.notification_service = MinIONotificationService(redis_client)
        self.retrieval_engine = RetrievalEngine(postgres_pool, redis_client, naga_llm)
        self.citation_engine = CitationEngine()
        self.prompt_composer = PromptComposer()

        self.ingestion_worker = IngestionWorker(
            source_registry=self.source_registry,
            minio_service=self.minio_service,
            postgres_pool=postgres_pool,
            redis_client=redis_client,
            naga_llm=naga_llm
        )

    async def create_workspace(
        self,
        owner_id: str,
        name: str,
        description: Optional[str] = None
    ) -> Workspace:
        """Create a new workspace."""
        return await self.source_registry.create_workspace(owner_id, name, description)

    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get a workspace by ID."""
        return await self.source_registry.get_workspace(workspace_id)

    async def list_workspaces(self, owner_id: str) -> List[Workspace]:
        """List all workspaces for an owner."""
        return await self.source_registry.list_workspaces(owner_id)

    async def register_source(
        self,
        workspace_id: str,
        source_type: str,
        title: str,
        content: bytes,
        mime_type: Optional[str] = None,
        language: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[SourceDocument, bool]:
        """
        Register and upload a new source.

        Returns:
            Tuple of (SourceDocument, is_new) where is_new indicates
            if this is a new document (True) or a duplicate (False)
        """
        file_hash = self.minio_service.compute_hash(content)

        existing = await self.source_registry.get_sources_by_hash(workspace_id, file_hash)

        if existing:
            return existing[0], False

        # Ensure title has correct extension based on source_type
        ext = self._get_extension_for_type(source_type, mime_type, title)
        if ext and not title.lower().endswith(ext):
            title = f"{title}{ext}"

        # Generate source_id BEFORE upload so MinIO key matches DB
        source_id = str(uuid.uuid4())

        canonical_uri, _ = await self.minio_service.upload_source(
            source_id,
            title,
            content,
            mime_type or "application/octet-stream"
        )

        source = await self.source_registry.register_source(
            workspace_id=workspace_id,
            source_type=source_type,
            title=title,
            canonical_uri=canonical_uri,
            file_hash=file_hash,
            mime_type=mime_type,
            language=language,
            size_bytes=len(content),
            metadata=metadata
        )

        asyncio.create_task(self._process_source_async(source.source_id))

        # Notify via NATS/Redis pub/sub
        try:
            await self.notification_service.notify_upload(
                source_id=source.source_id,
                workspace_id=workspace_id,
                bucket=self.minio_service.bucket,
                object_key=f"sources/{source.source_id}/{title}",
                metadata={"source_type": source_type, "size": len(content)}
            )
        except Exception as e:
            print(f"Warning: Failed to publish upload notification: {e}")

        return source, True

    def _get_extension_for_type(self, source_type: str, mime_type: Optional[str], title: str) -> Optional[str]:
        """Get the correct file extension based on source_type, mime_type, or title."""
        # Check if title already has an extension
        for ext in ['.txt', '.md', '.html', '.json', '.pdf', '.log', '.text']:
            if title.lower().endswith(ext):
                return None  # Extension already present

        # Map source_type to extension
        type_to_ext = {
            'txt': '.txt',
            'md': '.md',
            'markdown': '.md',
            'html': '.html',
            'json': '.json',
            'pdf': '.pdf',
            'log': '.log',
        }

        if source_type.lower() in type_to_ext:
            return type_to_ext[source_type.lower()]

        # Map mime_type to extension
        mime_to_ext = {
            'text/plain': '.txt',
            'text/markdown': '.md',
            'text/html': '.html',
            'application/json': '.json',
            'application/pdf': '.pdf',
        }

        if mime_type and mime_type in mime_to_ext:
            return mime_to_ext[mime_type]

        return '.txt'  # Default extension

    async def _process_source_async(self, source_id: str):
        """Background processing of a source."""
        try:
            await self.ingestion_worker.process_source(source_id)
        except Exception as e:
            print(f"Background processing error for {source_id}: {e}")

    async def get_source(self, source_id: str) -> Optional[SourceDocument]:
        """Get a source by ID."""
        return await self.source_registry.get_source(source_id)

    async def list_sources(
        self,
        workspace_id: str,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[SourceDocument]:
        """List sources in a workspace."""
        return await self.source_registry.list_sources(workspace_id, status, limit)

    async def delete_source(self, source_id: str) -> bool:
        """Archive (soft delete) a source."""
        return await self.source_registry.delete_source(source_id)

    async def query(
        self,
        workspace_id: str,
        query: str,
        mode: str = "source_workspace",
        top_docs: int = 5,
        top_chunks: int = 12,
        agent_memory_context: Optional[Dict[str, str]] = None,
        include_citations: bool = True
    ) -> WorkspaceQueryResult:
        """
        Query the source workspace.

        Args:
            workspace_id: Target workspace
            query: Search/question query
            mode: Query mode (source_workspace, agent_memory, hybrid)
            top_docs: Number of documents to retrieve
            top_chunks: Number of chunks to retrieve
            agent_memory_context: Context from 5-memory system (for hybrid)
            include_citations: Whether to include citations

        Returns:
            WorkspaceQueryResult with response and evidence
        """
        result = await self.retrieval_engine.query(
            workspace_id=workspace_id,
            query=query,
            top_docs=top_docs,
            top_sections=top_docs * 2,
            top_chunks=top_chunks,
            mode=mode
        )

        source_titles = {}
        for doc in result.documents:
            source_titles[doc.source_id] = doc.title

        citations = await self.retrieval_engine.get_citations(result.chunks)

        chunks_data = [
            {
                'chunk_id': c.chunk_id,
                'source_id': c.source_id,
                'content': c.content,
                'page': c.page,
                'score': c.score
            }
            for c in result.chunks
        ]

        docs_data = [
            {
                'source_id': d.source_id,
                'title': d.title,
                'summary': d.summary,
                'score': d.score
            }
            for d in result.documents
        ]

        if mode == "hybrid" and agent_memory_context:
            agent_ctx = AgentMemoryContext(
                working_memory=agent_memory_context.get('working_memory', ''),
                semantic_memories=agent_memory_context.get('semantic_memories', ''),
                episodic_snippets=agent_memory_context.get('episodic', ''),
                recent_facts=agent_memory_context.get('recent_facts', ''),
                timeline=agent_memory_context.get('timeline', '')
            )

            source_ctx = SourceEvidenceContext(
                documents=docs_data,
                chunks=chunks_data,
                citations=citations
            )

            hybrid_prompt = self.prompt_composer.compose_hybrid(
                agent_memory=agent_ctx,
                source_evidence=source_ctx,
                user_query=query,
                include_citations=include_citations
            )

            response_str, _ = await self.naga_llm.generate(hybrid_prompt.full_prompt)

            return WorkspaceQueryResult(
                response=response_str,
                documents=docs_data,
                chunks=chunks_data,
                citations=citations,
                mode=mode,
                tokens_used=hybrid_prompt.token_count
            )

        elif mode == "source_only" or mode == "source_workspace":
            source_ctx = SourceEvidenceContext(
                documents=docs_data,
                chunks=chunks_data,
                citations=citations
            )

            source_prompt = self.prompt_composer.compose_source_only(
                source_evidence=source_ctx,
                user_query=query,
                include_citations=include_citations
            )

            response_str, _ = await self.naga_llm.generate(source_prompt.full_prompt)

            return WorkspaceQueryResult(
                response=response_str,
                documents=docs_data,
                chunks=chunks_data,
                citations=citations,
                mode=mode,
                tokens_used=source_prompt.token_count
            )

        else:
            response = "Query mode not supported. Use: source_workspace, agent_memory, hybrid"
            return WorkspaceQueryResult(
                response=response,
                documents=[],
                chunks=[],
                citations={},
                mode=mode,
                tokens_used=0
            )

    async def get_source_status(self, source_id: str) -> Dict[str, Any]:
        """Get processing status of a source."""
        source = await self.source_registry.get_source(source_id)

        if not source:
            return {"error": "Source not found"}

        return {
            "source_id": source.source_id,
            "title": source.title,
            "status": source.status,
            "error_message": source.error_message,
            "created_at": source.created_at.isoformat() if source.created_at else None,
            "updated_at": source.updated_at.isoformat() if source.updated_at else None
        }

    async def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get statistics for a workspace."""
        counts = await self.source_registry.get_source_count(workspace_id)

        total = sum(counts.values())

        return {
            "workspace_id": workspace_id,
            "total_sources": total,
            "by_status": counts
        }

    async def share_source(
        self,
        source_id: str,
        workspace_id: str
    ) -> bool:
        """Share a source to a workspace (for cross-workspace sharing)."""
        return await self.source_registry.add_shared_source(workspace_id, source_id)