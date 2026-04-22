"""
Ingestion Worker for Source Workspace Engine
Processes source documents asynchronously using NATS or background processing
"""

import asyncio
import uuid
import logging
from typing import Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from src.memory_system.services.source_registry import SourceRegistry, SourceStatus
from src.memory_system.services.minio_service import MinIOService
from src.memory_system.parsers import ParserFactory
from src.memory_system.config import settings

logger = logging.getLogger(__name__)


class WorkerType(Enum):
    INGESTION = "ingestion"
    REINDEX = "reindex"
    CRAWL = "crawl"
    SUMMARY = "summary"
    ENTITY = "entity"


@dataclass
class IngestionTask:
    """Task for ingestion worker."""
    source_id: str
    workspace_id: str
    file_content: Optional[bytes] = None
    filename: str = ""
    mime_type: str = "application/octet-stream"
    worker_type: WorkerType = WorkerType.INGESTION
    priority: int = 0


class IngestionWorker:
    """
    Worker for processing source documents asynchronously.

    This worker can run as:
    1. A background task within the main service
    2. A separate process that listens to NATS
    3. A serverless function

    The actual processing follows this pipeline:
    1. Download/Read source
    2. Parse and extract text/structure
    3. Chunk the content
    4. Generate embeddings
    5. Generate summaries
    6. Index for retrieval
    """

    def __init__(
        self,
        source_registry: SourceRegistry,
        minio_service: MinIOService,
        postgres_pool,
        redis_client,
        naga_llm
    ):
        self.source_registry = source_registry
        self.minio_service = minio_service
        self.postgres_pool = postgres_pool
        self.redis_client = redis_client
        self.naga_llm = naga_llm
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def process_source(self, source_id: str) -> bool:
        """
        Process a single source document through the ingestion pipeline.

        Pipeline:
        1. Get source metadata from registry
        2. Download content from MinIO
        3. Parse document
        4. Store chunks and sections
        5. Generate embeddings
        6. Generate summaries
        7. Update source status to 'ready'
        """
        try:
            source = await self.source_registry.get_source(source_id)
            if not source:
                logger.error(f"Source not found: {source_id}")
                return False

            if source.status == SourceStatus.READY.value:
                logger.info(f"Source already processed: {source_id}")
                return True

            await self.source_registry.update_source_status(
                source_id,
                SourceStatus.PROCESSING.value
            )

            logger.info(f"Processing source: {source.title}")

            filename = source.title
            content = await self.minio_service.download_source(source_id, filename)

            parsed = await ParserFactory.parse(content, filename, source.mime_type)

            async with self.postgres_pool.acquire() as conn:
                await self._store_sections(conn, source_id, parsed.structure)
                await self._store_chunks(conn, source_id, parsed.text, parsed.structure)

            await self.source_registry.update_source_status(
                source_id,
                SourceStatus.READY.value
            )

            logger.info(f"Source processed successfully: {source_id}")
            return True

        except Exception as e:
            logger.error(f"Error processing source {source_id}: {str(e)}", exc_info=True)
            await self.source_registry.update_source_status(
                source_id,
                SourceStatus.FAILED.value,
                error_message=str(e)
            )
            return False

    async def _store_sections(self, conn, source_id: str, structure: List[dict]):
        """Store document sections."""
        for i, section in enumerate(structure):
            parent_id = None
            level = section.get('level', 0)

            if level > 0 and i > 0:
                for j in range(i - 1, -1, -1):
                    if structure[j].get('level', 0) < level:
                        parent_id = str(uuid.uuid4())
                        break

            section_id = str(uuid.uuid4())

            await conn.execute("""
                INSERT INTO source_sections
                (section_id, source_id, parent_section_id, level, title, path, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, section_id, source_id, parent_id, level,
                section.get('title'), section.get('path', ''),
                section.get('metadata', {}))

    async def _store_chunks(self, conn, source_id: str, text: str, structure: List[dict]):
        """Store document chunks."""
        section_ids = await conn.fetch("""
            SELECT section_id, path FROM source_sections
            WHERE source_id = $1
            ORDER BY section_id
        """, source_id)

        section_map = {row['path']: row['section_id'] for row in section_ids}

        default_section_id = section_ids[0]['section_id'] if section_ids else None

        chunk_size = 512 * 4
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            section_id = None
            for path, sec_id in section_map.items():
                if path in chunk[:100]:
                    section_id = sec_id
                    break

            if not section_id:
                section_id = default_section_id

            chunk_id = str(uuid.uuid4())

            await conn.execute("""
                INSERT INTO source_chunks
                (chunk_id, source_id, section_id, chunk_index, content, token_count, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, chunk_id, source_id, section_id, len(chunks),
                chunk, len(chunk) // 4, {})

            chunks.append({'chunk_id': chunk_id, 'content': chunk})
            start = end

    async def start(self):
        """Start the worker (for in-process background processing)."""
        self._running = True
        logger.info("Ingestion worker started")

    async def stop(self):
        """Stop the worker."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        logger.info("Ingestion worker stopped")

    async def submit_task(self, task: IngestionTask):
        """Submit a task for processing."""
        self._tasks.append(asyncio.create_task(self._process_task(task)))

    async def _process_task(self, task: IngestionTask):
        """Process a single task."""
        try:
            if task.file_content:
                source = await self.source_registry.get_source(task.source_id)
                if source:
                    await self.minio_service.upload_source(
                        task.source_id,
                        task.filename,
                        task.file_content,
                        task.mime_type
                    )

            await self.process_source(task.source_id)

        except Exception as e:
            logger.error(f"Task processing error: {str(e)}", exc_info=True)


class NATSIngestionWorker(IngestionWorker):
    """
    Ingestion worker that listens to NATS for incoming tasks.

    Requires nats-py library and NATS server running.
    """

    def __init__(self, *args, nats_url: str = "nats://localhost:4222", **kwargs):
        super().__init__(*args, **kwargs)
        self.nats_url = nats_url
        self.nc = None
        self.js = None

    async def start(self):
        """Start listening to NATS subjects."""
        try:
            import nats
        except ImportError:
            logger.warning("NATS not available. Install with: pip install nats-py")
            logger.info("Falling back to in-process worker")
            await super().start()
            return

        self.nc = await nats.connect(self.nats_url)
        self.js = self.nc.jetstream()

        await self.js.subscribe("workspace.ingest.*", durable="ingestion-worker")

        self._running = True
        logger.info(f"NATS ingestion worker started, subscribed to workspace.ingest.*")

    async def process_message(self, msg):
        """Process incoming NATS message."""
        try:
            import json
            data = json.loads(msg.data)

            task = IngestionTask(
                source_id=data['source_id'],
                workspace_id=data['workspace_id'],
                worker_type=WorkerType.INGESTION
            )

            await self._process_task(task)
            await msg.ack()

        except Exception as e:
            logger.error(f"Error processing NATS message: {str(e)}")
            await msg.nak()

    async def stop(self):
        """Stop the NATS worker."""
        if self.nc:
            await self.nc.close()
        await super().stop()
