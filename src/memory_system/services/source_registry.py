"""
Source Registry Service for Source Workspace Engine
Manages source document registration and status tracking
"""

import uuid
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import asyncpg


class SourceStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


@dataclass
class SourceDocument:
    source_id: str
    workspace_id: str
    source_type: str
    title: str
    canonical_uri: Optional[str]
    file_hash: Optional[str]
    mime_type: Optional[str]
    language: Optional[str]
    status: str
    size_bytes: Optional[int]
    metadata: dict
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class Workspace:
    workspace_id: str
    owner_id: str
    name: str
    description: Optional[str]
    shared_sources: List[str]
    created_at: datetime
    updated_at: datetime


class SourceRegistry:
    """Service for managing source documents and workspaces."""

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def create_workspace(
        self,
        owner_id: str,
        name: str,
        description: Optional[str] = None
    ) -> Workspace:
        """Create a new workspace."""
        workspace_id = str(uuid.uuid4())
        now = datetime.utcnow()

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO source_workspaces
                (workspace_id, owner_id, name, description, shared_sources, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, workspace_id, owner_id, name, description, '[]', now, now)

            return Workspace(
                workspace_id=workspace_id,
                owner_id=owner_id,
                name=name,
                description=description,
                shared_sources=[],
                created_at=now,
                updated_at=now
            )

    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get a workspace by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT workspace_id, owner_id, name, description, shared_sources, created_at, updated_at
                FROM source_workspaces
                WHERE workspace_id = $1
            """, workspace_id)

            if not row:
                return None

            return Workspace(
                workspace_id=row['workspace_id'],
                owner_id=row['owner_id'],
                name=row['name'],
                description=row['description'],
                shared_sources=row['shared_sources'] or [],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )

    async def list_workspaces(self, owner_id: str) -> List[Workspace]:
        """List all workspaces for an owner."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT workspace_id, owner_id, name, description, shared_sources, created_at, updated_at
                FROM source_workspaces
                WHERE owner_id = $1
                ORDER BY created_at DESC
            """, owner_id)

            return [
                Workspace(
                    workspace_id=r['workspace_id'],
                    owner_id=r['owner_id'],
                    name=r['name'],
                    description=r['description'],
                    shared_sources=r['shared_sources'] or [],
                    created_at=r['created_at'],
                    updated_at=r['updated_at']
                )
                for r in rows
            ]

    async def register_source(
        self,
        workspace_id: str,
        source_type: str,
        title: str,
        canonical_uri: Optional[str] = None,
        file_hash: Optional[str] = None,
        mime_type: Optional[str] = None,
        language: Optional[str] = None,
        size_bytes: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> SourceDocument:
        """Register a new source document."""
        source_id = str(uuid.uuid4())
        now = datetime.utcnow()
        status = SourceStatus.PENDING.value

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO source_documents
                (source_id, workspace_id, source_type, title, canonical_uri, file_hash,
                 mime_type, language, status, size_bytes, metadata, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12, $13)
            """, source_id, workspace_id, source_type, title, canonical_uri, file_hash,
                mime_type, language, status, size_bytes, json.dumps(metadata or {}), now, now)

            return SourceDocument(
                source_id=source_id,
                workspace_id=workspace_id,
                source_type=source_type,
                title=title,
                canonical_uri=canonical_uri,
                file_hash=file_hash,
                mime_type=mime_type,
                language=language,
                status=status,
                size_bytes=size_bytes,
                metadata=metadata or {},
                error_message=None,
                created_at=now,
                updated_at=now
            )

    async def get_source(self, source_id: str) -> Optional[SourceDocument]:
        """Get a source document by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT source_id, workspace_id, source_type, title, canonical_uri, file_hash,
                       mime_type, language, status, size_bytes, metadata, error_message,
                       created_at, updated_at
                FROM source_documents
                WHERE source_id = $1
            """, source_id)

            if not row:
                return None

            return SourceDocument(
                source_id=row['source_id'],
                workspace_id=row['workspace_id'],
                source_type=row['source_type'],
                title=row['title'],
                canonical_uri=row['canonical_uri'],
                file_hash=row['file_hash'],
                mime_type=row['mime_type'],
                language=row['language'],
                status=row['status'],
                size_bytes=row['size_bytes'],
                metadata=row['metadata'] or {},
                error_message=row['error_message'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )

    async def list_sources(
        self,
        workspace_id: str,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[SourceDocument]:
        """List source documents in a workspace."""
        async with self.pool.acquire() as conn:
            if status:
                rows = await conn.fetch("""
                    SELECT source_id, workspace_id, source_type, title, canonical_uri, file_hash,
                           mime_type, language, status, size_bytes, metadata, error_message,
                           created_at, updated_at
                    FROM source_documents
                    WHERE workspace_id = $1 AND status = $2
                    ORDER BY created_at DESC
                    LIMIT $3 OFFSET $4
                """, workspace_id, status, limit, offset)
            else:
                rows = await conn.fetch("""
                    SELECT source_id, workspace_id, source_type, title, canonical_uri, file_hash,
                           mime_type, language, status, size_bytes, metadata, error_message,
                           created_at, updated_at
                    FROM source_documents
                    WHERE workspace_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                """, workspace_id, limit, offset)

            return [
                SourceDocument(
                    source_id=r['source_id'],
                    workspace_id=r['workspace_id'],
                    source_type=r['source_type'],
                    title=r['title'],
                    canonical_uri=r['canonical_uri'],
                    file_hash=r['file_hash'],
                    mime_type=r['mime_type'],
                    language=r['language'],
                    status=r['status'],
                    size_bytes=r['size_bytes'],
                    metadata=r['metadata'] or {},
                    error_message=r['error_message'],
                    created_at=r['created_at'],
                    updated_at=r['updated_at']
                )
                for r in rows
            ]

    async def update_source_status(
        self,
        source_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """Update the status of a source document."""
        async with self.pool.acquire() as conn:
            if error_message:
                result = await conn.execute("""
                    UPDATE source_documents
                    SET status = $1, error_message = $2, updated_at = now()
                    WHERE source_id = $3
                """, status, error_message, source_id)
            else:
                result = await conn.execute("""
                    UPDATE source_documents
                    SET status = $1, updated_at = now()
                    WHERE source_id = $2
                """, status, source_id)

            return result != "UPDATE 0"

    async def delete_source(self, source_id: str) -> bool:
        """Soft delete a source document (archive)."""
        return await self.update_source_status(source_id, SourceStatus.ARCHIVED.value)

    async def add_shared_source(self, workspace_id: str, source_id: str) -> bool:
        """Add a source to the shared sources list of a workspace."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE source_workspaces
                SET shared_sources = COALESCE(shared_sources, '[]'::jsonb) || $1::jsonb,
                    updated_at = now()
                WHERE workspace_id = $2
            """, json.dumps([source_id]), workspace_id)
            return True

    async def get_sources_by_hash(
        self,
        workspace_id: str,
        file_hash: str
    ) -> List[SourceDocument]:
        """Find sources by file hash (for deduplication)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT source_id, workspace_id, source_type, title, canonical_uri, file_hash,
                       mime_type, language, status, size_bytes, metadata, error_message,
                       created_at, updated_at
                FROM source_documents
                WHERE workspace_id = $1 AND file_hash = $2 AND status != 'archived'
            """, workspace_id, file_hash)

            return [
                SourceDocument(
                    source_id=r['source_id'],
                    workspace_id=r['workspace_id'],
                    source_type=r['source_type'],
                    title=r['title'],
                    canonical_uri=r['canonical_uri'],
                    file_hash=r['file_hash'],
                    mime_type=r['mime_type'],
                    language=r['language'],
                    status=r['status'],
                    size_bytes=r['size_bytes'],
                    metadata=r['metadata'] or {},
                    error_message=r['error_message'],
                    created_at=r['created_at'],
                    updated_at=r['updated_at']
                )
                for r in rows
            ]

    async def get_source_count(self, workspace_id: str) -> Dict[str, int]:
        """Get count of sources grouped by status."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT status, COUNT(*) as count
                FROM source_documents
                WHERE workspace_id = $1
                GROUP BY status
            """, workspace_id)

            return {r['status']: r['count'] for r in rows}