"""
Tests for Source Workspace Engine - Phase 1 & 2
Tests the base functionality: workspace creation, source registration, parsing, ingestion
"""

import pytest
import asyncio
import uuid
from datetime import datetime

import asyncpg
import redis.asyncio as redis

from src.memory_system.services.source_registry import SourceRegistry, SourceDocument, Workspace
from src.memory_system.services.minio_service import MinIOService
from src.memory_system.services.retrieval_engine import RetrievalEngine
from src.memory_system.services.citation_engine import CitationEngine
from src.memory_system.parsers import ParserFactory, TextParser, MarkdownParser, JSONParser
from src.memory_system.workers.ingestion_worker import IngestionWorker, IngestionTask, WorkerType
from src.memory_system.config import settings


@pytest.fixture
async def postgres_pool():
    """Create a PostgreSQL connection pool."""
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
    pool.close()


@pytest.fixture
async def redis_client():
    """Create a Redis client."""
    client = redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True
    )
    yield client
    await client.aclose()


class MockNAGALLM:
    """Mock NAGA LLM for testing."""

    async def generate(self, prompt: str) -> str:
        return "This is a mock response based on: " + prompt[:50] + "..."

    async def embeddings(self, text: str) -> list:
        return [0.1] * 1536


@pytest.fixture
async def source_registry(postgres_pool):
    """Create SourceRegistry instance."""
    return SourceRegistry(postgres_pool)


@pytest.fixture
async def minio_service():
    """Create MinIO service instance."""
    return MinIOService()


class TestWorkspaces:
    """Test workspace management."""

    @pytest.mark.asyncio
    async def test_create_workspace(self, source_registry):
        """Test creating a new workspace."""
        workspace = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="Test Workspace",
            description="A test workspace"
        )

        assert workspace is not None
        assert workspace.name == "Test Workspace"
        assert workspace.workspace_id is not None
        assert len(workspace.shared_sources) == 0

    @pytest.mark.asyncio
    async def test_get_workspace(self, source_registry):
        """Test getting a workspace by ID."""
        created = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="Get Test Workspace"
        )

        fetched = await source_registry.get_workspace(created.workspace_id)

        assert fetched is not None
        assert str(fetched.workspace_id) == str(created.workspace_id)
        assert fetched.name == created.name

    @pytest.mark.asyncio
    async def test_list_workspaces(self, source_registry):
        """Test listing workspaces for an owner."""
        owner_id = str(uuid.uuid4())

        await source_registry.create_workspace(owner_id, "Workspace 1")
        await source_registry.create_workspace(owner_id, "Workspace 2")

        workspaces = await source_registry.list_workspaces(owner_id)

        assert len(workspaces) >= 2
        names = [w.name for w in workspaces]
        assert "Workspace 1" in names
        assert "Workspace 2" in names


class TestSources:
    """Test source document management."""

    @pytest.mark.asyncio
    async def test_register_source(self, source_registry):
        """Test registering a new source document."""
        workspace = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="Source Test Workspace"
        )

        source = await source_registry.register_source(
            workspace_id=workspace.workspace_id,
            source_type="md",
            title="Test Document",
            canonical_uri="minio://bucket/test/doc.md",
            file_hash="abc123",
            mime_type="text/markdown",
            size_bytes=1024
        )

        assert source is not None
        assert source.title == "Test Document"
        assert source.status == "pending"
        assert source.source_type == "md"

    @pytest.mark.asyncio
    async def test_get_source(self, source_registry):
        """Test getting a source by ID."""
        workspace = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="Get Source Test"
        )

        created = await source_registry.register_source(
            workspace_id=workspace.workspace_id,
            source_type="txt",
            title="Get Source Test Doc"
        )

        fetched = await source_registry.get_source(created.source_id)

        assert fetched is not None
        assert str(fetched.source_id) == str(created.source_id)
        assert fetched.title == "Get Source Test Doc"

    @pytest.mark.asyncio
    async def test_list_sources(self, source_registry):
        """Test listing sources in a workspace."""
        workspace = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="List Sources Test"
        )

        await source_registry.register_source(
            workspace_id=workspace.workspace_id,
            source_type="md",
            title="Doc 1"
        )
        await source_registry.register_source(
            workspace_id=workspace.workspace_id,
            source_type="md",
            title="Doc 2"
        )

        sources = await source_registry.list_sources(workspace.workspace_id)

        assert len(sources) >= 2
        titles = [s.title for s in sources]
        assert "Doc 1" in titles
        assert "Doc 2" in titles

    @pytest.mark.asyncio
    async def test_update_source_status(self, source_registry):
        """Test updating source status."""
        workspace = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="Status Update Test"
        )

        source = await source_registry.register_source(
            workspace_id=workspace.workspace_id,
            source_type="txt",
            title="Status Test Doc"
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
            title="Delete Test Doc"
        )

        success = await source_registry.delete_source(source.source_id)

        assert success is True

        deleted = await source_registry.get_source(source.source_id)
        assert deleted.status == "archived"

    @pytest.mark.asyncio
    async def test_get_sources_by_hash(self, source_registry):
        """Test finding sources by file hash (deduplication)."""
        workspace = await source_registry.create_workspace(
            owner_id=str(uuid.uuid4()),
            name="Hash Test"
        )

        file_hash = "unique_hash_123"

        await source_registry.register_source(
            workspace_id=workspace.workspace_id,
            source_type="txt",
            title="Original Doc",
            file_hash=file_hash
        )

        duplicates = await source_registry.get_sources_by_hash(
            workspace.workspace_id,
            file_hash
        )

        assert len(duplicates) == 1
        assert duplicates[0].title == "Original Doc"


class TestParsers:
    """Test document parsers."""

    @pytest.mark.asyncio
    async def test_text_parser(self):
        """Test plain text parsing."""
        content = b"Hello, this is a test document.\n\nIt has multiple lines."
        parser = TextParser()

        result = await parser.parse(content, "test.txt")

        assert result.text == "Hello, this is a test document.\n\nIt has multiple lines."
        assert len(result.structure) == 1
        assert result.metadata['parser'] == 'text'

    @pytest.mark.asyncio
    async def test_markdown_parser(self):
        """Test markdown parsing."""
        content = b"""# Title

## Section 1

This is content in section 1.

## Section 2

This is content in section 2.
"""
        parser = MarkdownParser()

        result = await parser.parse(content, "test.md")

        assert "Title" in result.text
        assert "Section 1" in result.text
        assert len(result.structure) >= 2
        assert result.metadata['parser'] == 'markdown'

    @pytest.mark.asyncio
    async def test_json_parser(self):
        """Test JSON parsing."""
        content = b'{"name": "test", "value": 123, "items": [1, 2, 3]}'
        parser = JSONParser()

        result = await parser.parse(content, "test.json")

        assert "test" in result.text
        assert "123" in result.text
        assert result.metadata['parser'] == 'json'

    @pytest.mark.asyncio
    async def test_parser_factory(self):
        """Test parser factory."""
        content = b"# Test"

        result = await ParserFactory.parse(content, "test.md")

        assert result is not None
        assert result.metadata['parser'] == 'markdown'

    @pytest.mark.asyncio
    async def test_unsupported_type(self):
        """Test handling unsupported file types."""
        content = b"Some content"

        with pytest.raises(ValueError):
            await ParserFactory.parse(content, "test.unknown")


class TestCitationEngine:
    """Test citation engine."""

    def test_create_citation(self):
        """Test creating a citation."""
        citation = CitationEngine.create_citation(
            chunk_id="chunk123",
            source_id="source456",
            source_title="Test Document",
            page=5,
            locator="paragraph",
            quote_text="This is a test quote."
        )

        assert citation.citation_id is not None
        assert citation.chunk_id == "chunk123"
        assert citation.source_title == "Test Document"
        assert citation.page == 5

    def test_format_apa(self):
        """Test APA citation format."""
        from src.memory_system.services.citation_engine import Citation

        citation = Citation(
            citation_id="cit123",
            chunk_id="chunk123",
            source_id="source456",
            source_title="Test Document",
            page=5,
            locator="para",
            quote_text="Quote",
            created_at=datetime.utcnow()
        )

        formatted = CitationEngine.format_apa(citation)

        assert "Test Document" in formatted
        assert "5" in formatted

    def test_format_direct(self):
        """Test direct citation format."""
        from src.memory_system.services.citation_engine import Citation

        citation = Citation(
            citation_id="cit123",
            chunk_id="chunk123",
            source_id="source456",
            source_title="Test Document",
            page=5,
            locator="para",
            quote_text="Quote",
            created_at=datetime.utcnow()
        )

        formatted = CitationEngine.format_direct(citation)

        assert "[Source:" in formatted
        assert "Test Document" in formatted
        assert "5" in formatted

    def test_build_citation_context(self):
        """Test building citation context."""
        chunks = [
            {
                'source_id': 'source1',
                'chunk_id': 'chunk1',
                'content': 'This is the first chunk content.',
                'page': 1
            },
            {
                'source_id': 'source2',
                'chunk_id': 'chunk2',
                'content': 'This is the second chunk content.',
                'page': 2
            }
        ]

        source_titles = {
            'source1': 'Document One',
            'source2': 'Document Two'
        }

        context = CitationEngine.build_citation_context(chunks, source_titles)

        assert "Document One" in context
        assert "Document Two" in context
        assert "first chunk" in context
        assert "[1]" in context
        assert "[2]" in context


class TestIngestionWorker:
    """Test ingestion worker."""

    @pytest.mark.asyncio
    async def test_ingestion_task_creation(self):
        """Test creating an ingestion task."""
        task = IngestionTask(
            source_id="src123",
            workspace_id="ws456",
            filename="test.md",
            mime_type="text/markdown",
            worker_type=WorkerType.INGESTION
        )

        assert task.source_id == "src123"
        assert task.workspace_id == "ws456"
        assert task.worker_type == WorkerType.INGESTION


class TestMinIOService:
    """Test MinIO service."""

    @pytest.mark.asyncio
    async def test_upload_download(self, minio_service):
        """Test uploading and downloading a file."""
        source_id = str(uuid.uuid4())
        filename = "test_upload.txt"
        content = b"Hello, MinIO!"

        canonical_uri, file_hash = await minio_service.upload_source(
            source_id,
            filename,
            content,
            "text/plain"
        )

        assert "minio://" in canonical_uri
        assert len(file_hash) == 64

        downloaded = await minio_service.download_source(source_id, filename)

        assert downloaded == content

    @pytest.mark.asyncio
    async def test_exists(self, minio_service):
        """Test checking if file exists."""
        source_id = str(uuid.uuid4())
        filename = "exists_test.txt"
        content = b"Test content"

        await minio_service.upload_source(source_id, filename, content)

        exists = await minio_service.exists(source_id, filename)

        assert exists is True

    @pytest.mark.asyncio
    async def test_delete(self, minio_service):
        """Test deleting a file."""
        source_id = str(uuid.uuid4())
        filename = "delete_test.txt"
        content = b"Delete this"

        await minio_service.upload_source(source_id, filename, content)

        deleted = await minio_service.delete_source(source_id, filename)

        assert deleted is True


class TestRetrievalEngine:
    """Test retrieval engine."""

    @pytest.mark.asyncio
    async def test_retrieval_result_structure(self, postgres_pool, redis_client):
        """Test that retrieval result structure is correct."""
        mock_naga = MockNAGALLM()
        engine = RetrievalEngine(postgres_pool, redis_client, mock_naga)

        assert engine.pool is not None
        assert engine.redis is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
