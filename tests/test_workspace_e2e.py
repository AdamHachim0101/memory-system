"""
E2E Tests for Source Workspace Engine
Full flow: create workspace → register source → process → query
"""

import pytest
import asyncio
import uuid

import asyncpg
import redis.asyncio as redis

from src.memory_system.services.source_registry import SourceRegistry, SourceDocument, Workspace
from src.memory_system.services.minio_service import MinIOService
from src.memory_system.parsers import ParserFactory
from src.memory_system.config import settings


class MockNAGALLM:
    """Mock NAGA LLM for testing."""

    async def generate(self, prompt: str) -> str:
        return f"Mock response based on query about: {prompt[:50]}..."

    async def embeddings(self, text: str) -> list:
        import hashlib
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        import random
        random.seed(seed)
        return [random.random() for _ in range(1536)]


@pytest.fixture
async def postgres_pool():
    """Create PostgreSQL pool."""
    pool = await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        min_size=2,
        max_size=10
    )
    yield pool
    await pool.close()


@pytest.fixture
async def redis_client():
    """Create Redis client."""
    client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True
    )
    yield client
    await client.aclose()


@pytest.fixture
async def source_registry(postgres_pool):
    """Create source registry."""
    return SourceRegistry(postgres_pool)


@pytest.fixture
async def minio_service():
    """Create MinIO service."""
    return MinIOService()


class TestWorkspaceE2E:
    """End-to-end tests for workspace flow."""

    @pytest.mark.asyncio
    async def test_create_and_get_workspace(self, source_registry):
        """Test creating and retrieving a workspace."""
        owner_id = str(uuid.uuid4())

        workspace = await source_registry.create_workspace(
            owner_id=owner_id,
            name="E2E Test Workspace",
            description="Test workspace for E2E testing"
        )

        assert workspace is not None
        assert workspace.name == "E2E Test Workspace"

        fetched = await source_registry.get_workspace(workspace.workspace_id)
        assert fetched is not None
        assert fetched.name == workspace.name

    @pytest.mark.asyncio
    async def test_list_workspaces(self, source_registry):
        """Test listing workspaces."""
        owner_id = str(uuid.uuid4())

        await source_registry.create_workspace(owner_id, "Workspace A")
        await source_registry.create_workspace(owner_id, "Workspace B")

        workspaces = await source_registry.list_workspaces(owner_id)
        assert len(workspaces) >= 2

    @pytest.mark.asyncio
    async def test_register_and_get_source(self, source_registry):
        """Test registering and retrieving a source."""
        workspace = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="Source Test Workspace"
        )

        source = await source_registry.register_source(
            workspace_id=workspace.workspace_id,
            source_type="md",
            title="Test Document",
            canonical_uri="minio://bucket/test/doc.md",
            file_hash="test_hash_123",
            mime_type="text/markdown",
            size_bytes=1024
        )

        assert source is not None
        assert source.title == "Test Document"
        assert source.status == "pending"

        fetched = await source_registry.get_source(source.source_id)
        assert fetched is not None
        assert fetched.title == "Test Document"

    @pytest.mark.asyncio
    async def test_list_sources(self, source_registry):
        """Test listing sources."""
        workspace = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="List Sources Test"
        )

        await source_registry.register_source(
            workspace_id=workspace.workspace_id,
            source_type="txt",
            title="Doc 1"
        )
        await source_registry.register_source(
            workspace_id=workspace.workspace_id,
            source_type="txt",
            title="Doc 2"
        )

        sources = await source_registry.list_sources(workspace.workspace_id)
        assert len(sources) >= 2

    @pytest.mark.asyncio
    async def test_update_source_status(self, source_registry):
        """Test updating source status."""
        workspace = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="Status Test"
        )

        source = await source_registry.register_source(
            workspace_id=workspace.workspace_id,
            source_type="txt",
            title="Status Test"
        )

        success = await source_registry.update_source_status(
            source.source_id,
            "processing"
        )
        assert success is True

        updated = await source_registry.get_source(source.source_id)
        assert updated.status == "processing"

    @pytest.mark.asyncio
    async def test_delete_source(self, source_registry):
        """Test soft deleting a source."""
        workspace = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="Delete Test"
        )

        source = await source_registry.register_source(
            workspace_id=workspace.workspace_id,
            source_type="txt",
            title="Delete Test"
        )

        deleted = await source_registry.delete_source(source.source_id)
        assert deleted is True

        archived = await source_registry.get_source(source.source_id)
        assert archived.status == "archived"

    @pytest.mark.asyncio
    async def test_get_sources_by_hash(self, source_registry):
        """Test finding sources by hash."""
        workspace = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="Hash Test"
        )

        file_hash = "unique_test_hash"

        await source_registry.register_source(
            workspace_id=workspace.workspace_id,
            source_type="txt",
            title="Original",
            file_hash=file_hash
        )

        duplicates = await source_registry.get_sources_by_hash(
            workspace.workspace_id,
            file_hash
        )

        assert len(duplicates) == 1

    @pytest.mark.asyncio
    async def test_get_source_count(self, source_registry):
        """Test getting source count by status."""
        workspace = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="Count Test"
        )

        for i in range(3):
            await source_registry.register_source(
                workspace_id=workspace.workspace_id,
                source_type="txt",
                title=f"Doc {i}"
            )

        counts = await source_registry.get_source_count(workspace.workspace_id)
        assert counts.get('pending', 0) >= 3


class TestMinIO:
    """Test MinIO integration."""

    @pytest.mark.asyncio
    async def test_upload_and_download(self, minio_service):
        """Test uploading and downloading."""
        source_id = str(uuid.uuid4())
        filename = "test.txt"
        content = b"Hello MinIO!"

        uri, hash_val = await minio_service.upload_source(
            source_id, filename, content, "text/plain"
        )

        assert "minio://" in uri
        assert len(hash_val) == 64

        downloaded = await minio_service.download_source(source_id, filename)
        assert downloaded == content

    @pytest.mark.asyncio
    async def test_exists(self, minio_service):
        """Test file existence check."""
        source_id = str(uuid.uuid4())
        filename = "exists.txt"
        content = b"Test"

        await minio_service.upload_source(source_id, filename, content)

        exists = await minio_service.exists(source_id, filename)
        assert exists is True

        not_exists = await minio_service.exists(str(uuid.uuid4()), "nonexistent.txt")
        assert not_exists is False

    @pytest.mark.asyncio
    async def test_delete(self, minio_service):
        """Test file deletion."""
        source_id = str(uuid.uuid4())
        filename = "delete_me.txt"

        await minio_service.upload_source(source_id, filename, b"Delete this")

        deleted = await minio_service.delete_source(source_id, filename)
        assert deleted is True


class TestParsers:
    """Test parsers."""

    @pytest.mark.asyncio
    async def test_markdown_parsing(self):
        """Test markdown parsing."""
        content = b"""# Title

## Section

Content here.
"""
        result = await ParserFactory.parse(content, "test.md")

        assert "Title" in result.text
        assert result.metadata['parser'] == 'markdown'

    @pytest.mark.asyncio
    async def test_text_parsing(self):
        """Test text parsing."""
        content = b"Plain text content."
        result = await ParserFactory.parse(content, "test.txt")

        assert "Plain text" in result.text

    @pytest.mark.asyncio
    async def test_json_parsing(self):
        """Test JSON parsing."""
        content = b'{"key": "value", "num": 123}'
        result = await ParserFactory.parse(content, "test.json")

        assert "key" in result.text
        assert "value" in result.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])