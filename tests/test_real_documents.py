"""
Integration Tests for Source Workspace Engine with Real Documents
Tests the full pipeline: upload → parse → index → query → grounded response
"""

import pytest
import asyncio
import uuid
import json
import tempfile
import os

import asyncpg
import redis.asyncio as aioredis

from src.memory_system.config import settings
from src.memory_system.services.minio_service import MinIOService
from src.memory_system.services.source_registry import SourceRegistry
from src.memory_system.services.workspace_service import SourceWorkspaceService
from src.memory_system.parsers import ParserFactory


class MockNAGALLM:
    """Mock NAGA LLM for testing."""

    async def generate(self, prompt: str) -> str:
        if 'memory' in prompt.lower():
            return "Based on the documents, the memory system uses Redis for caching and PostgreSQL for persistence."
        elif 'payment' in prompt.lower():
            return "Payments are processed with a 2% commission for USD transactions."
        return "Mock response based on the source documents."

    async def embeddings(self, text: str) -> list:
        import hashlib
        import random
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        return [random.random() for _ in range(1536)]


@pytest.fixture
async def pool():
    """Create PostgreSQL pool."""
    p = await asyncpg.create_pool(
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        min_size=2,
        max_size=10
    )
    yield p
    await p.close()


@pytest.fixture
async def redis_client():
    """Create Redis client."""
    client = aioredis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True
    )
    yield client
    await client.aclose()


@pytest.fixture
async def workspace_service(pool, redis_client):
    """Create workspace service."""
    return SourceWorkspaceService(pool, redis_client, MockNAGALLM())


class TestRealDocumentPipeline:
    """Test with real markdown document."""

    @pytest.mark.asyncio
    async def test_markdown_document(self, workspace_service):
        """Test processing a real markdown document."""
        owner_id = str(uuid.uuid4())

        workspace = await workspace_service.create_workspace(
            owner_id=owner_id,
            name='Real Docs Test'
        )

        content = b'''# Project Architecture

## Overview
This is a multi-agent banking system.

## Components

### Orchestrator
The orchestrator routes requests to appropriate domain agents.

### Domain Agents
- Financial Agent handles payments, refunds, queries
- Logistics Agent handles tracking, delivery
- Market Agent handles analysis and reports

## Memory System

The memory system uses a 5-tier architecture:
1. Active Context Cache (Redis)
2. Working Memory (PostgreSQL)
3. Episodic Memory (PostgreSQL + pgvector)
4. Semantic Memory (PostgreSQL + pgvector)
5. Relational Memory (Neo4j)

## Features

- Low latency responses
- Persistent storage
- Cross-session continuity
- Grounded responses with citations
'''

        source, is_new = await workspace_service.register_source(
            workspace_id=workspace.workspace_id,
            source_type='md',
            title='architecture.md',
            content=content
        )

        assert source is not None
        assert is_new is True
        assert source.title == 'architecture.md'

        # Query the document
        result = await workspace_service.query(
            workspace_id=workspace.workspace_id,
            query="Tell me about the memory system",
            mode="source_workspace"
        )

        assert result.response is not None

    @pytest.mark.asyncio
    async def test_document_search(self, workspace_service):
        """Test searching within a document."""
        owner_id = str(uuid.uuid4())

        workspace = await workspace_service.create_workspace(
            owner_id=owner_id,
            name='Search Test'
        )

        content = b'''# Financial Services

## Payments
- Domestic: 2% commission
- International: 3% commission

## Refunds
- Process within 5 days
- Full or partial available

## Queries
- Balance inquiry
- Transaction history
- Export options
'''

        source, _ = await workspace_service.register_source(
            workspace_id=workspace.workspace_id,
            source_type='md',
            title='financial.md',
            content=content
        )

        # List sources
        sources = await workspace_service.list_sources(workspace.workspace_id)
        assert len(sources) >= 1

        # Get source
        fetched = await workspace_service.get_source(source.source_id)
        assert str(fetched.source_id) == str(source.source_id)


class TestMultiDocumentWorkspace:
    """Test with multiple documents."""

    @pytest.mark.asyncio
    async def test_multiple_documents(self, workspace_service):
        """Test managing multiple documents in a workspace."""
        owner_id = str(uuid.uuid4())

        workspace = await workspace_service.create_workspace(
            owner_id=owner_id,
            name='Multi Doc Test'
        )

        docs = [
            ('intro.md', b'# Introduction\n\nThis is the introduction.'),
            ('api.md', b'# API Documentation\n\n## Endpoints\n- GET /status\n- POST /submit'),
            ('guide.md', b'# User Guide\n\n## Getting Started\n1. Create account\n2. Verify email'),
        ]

        for title, content in docs:
            source, _ = await workspace_service.register_source(
                workspace_id=workspace.workspace_id,
                source_type='md',
                title=title,
                content=content
            )

        # List all sources
        sources = await workspace_service.list_sources(workspace.workspace_id)
        assert len(sources) >= 3

        # Check stats
        stats = await workspace_service.get_workspace_stats(workspace.workspace_id)
        assert stats['total_sources'] >= 3


class TestWorkspaceSharing:
    """Test workspace sharing functionality."""

    @pytest.mark.asyncio
    async def test_share_source(self, workspace_service):
        """Test sharing a source between workspaces."""
        owner_id = str(uuid.uuid4())

        ws1 = await workspace_service.create_workspace(owner_id=owner_id, name='Workspace 1')
        ws2 = await workspace_service.create_workspace(owner_id=owner_id, name='Workspace 2')

        source, _ = await workspace_service.register_source(
            workspace_id=ws1.workspace_id,
            source_type='md',
            title='shared_doc.md',
            content=b'# Shared Document\n\nThis document is shared.'
        )

        shared = await workspace_service.share_source(source.source_id, ws2.workspace_id)
        assert shared is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])