"""
MinIO Event Notification Service
Publishes events to NATS when files are uploaded
"""

import asyncio
import json
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import redis.asyncio as aioredis


class EventType(Enum):
    FILE_UPLOADED = "file.uploaded"
    FILE_DELETED = "file.deleted"
    FILE_PROCESSED = "file.processed"
    FILE_FAILED = "file.failed"


@dataclass
class MinIOEvent:
    event_type: str
    bucket: str
    object_key: str
    source_id: str
    workspace_id: str
    timestamp: str
    metadata: dict = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> 'MinIOEvent':
        d = json.loads(data)
        return cls(**d)


class MinIONotificationService:
    """
    Service for publishing MinIO events to NATS/Redis pub/sub.
    
    This enables event-driven processing where workers can
    subscribe to specific event types.
    """

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.pubsub_channel = "minio:events"

    async def publish_event(self, event: MinIOEvent) -> None:
        """
        Publish an event to the notification channel.
        Workers can subscribe to receive these events.
        """
        try:
            await self.redis.publish(
                self.pubsub_channel,
                event.to_json()
            )
        except Exception as e:
            print(f"Failed to publish event: {e}")

    async def notify_upload(
        self,
        source_id: str,
        workspace_id: str,
        bucket: str,
        object_key: str,
        metadata: Optional[dict] = None
    ) -> None:
        """Notify that a file was uploaded."""
        event = MinIOEvent(
            event_type=EventType.FILE_UPLOADED.value,
            bucket=bucket,
            object_key=object_key,
            source_id=source_id,
            workspace_id=workspace_id,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {}
        )
        await self.publish_event(event)

    async def notify_processed(
        self,
        source_id: str,
        workspace_id: str,
        bucket: str,
        object_key: str,
        metadata: Optional[dict] = None
    ) -> None:
        """Notify that a file was processed."""
        event = MinIOEvent(
            event_type=EventType.FILE_PROCESSED.value,
            bucket=bucket,
            object_key=object_key,
            source_id=source_id,
            workspace_id=workspace_id,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {}
        )
        await self.publish_event(event)

    async def notify_failed(
        self,
        source_id: str,
        workspace_id: str,
        bucket: str,
        object_key: str,
        error: str,
        metadata: Optional[dict] = None
    ) -> None:
        """Notify that file processing failed."""
        meta = metadata or {}
        meta['error'] = error
        event = MinIOEvent(
            event_type=EventType.FILE_FAILED.value,
            bucket=bucket,
            object_key=object_key,
            source_id=source_id,
            workspace_id=workspace_id,
            timestamp=datetime.utcnow().isoformat(),
            metadata=meta
        )
        await self.publish_event(event)


class MinIOEventSubscriber:
    """
    Subscribe to MinIO events for async processing.
    Useful for workers that need to react to file events.
    """

    def __init__(
        self,
        redis_client: aioredis.Redis,
        channel: str = "minio:events"
    ):
        self.redis = redis_client
        self.channel = channel
        self.pubsub = None
        self.running = False

    async def subscribe(
        self,
        callback: Callable[[MinIOEvent], Awaitable[None]],
        event_types: Optional[list] = None
    ) -> None:
        """
        Subscribe to MinIO events and call callback for each.
        
        Args:
            callback: Async function to call for each event
            event_types: List of event types to filter (None = all)
        """
        self.pubsub = self.redis.pubsub()
        await self.pubsub.subscribe(self.channel)
        self.running = True

        print(f"Subscribed to {self.channel}")

        while self.running:
            try:
                message = await self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    try:
                        event = MinIOEvent.from_json(message['data'])
                        
                        if event_types is None or event.event_type in event_types:
                            await callback(event)
                            
                    except Exception as e:
                        print(f"Error processing event: {e}")

            except Exception as e:
                print(f"Subscriber error: {e}")
                await asyncio.sleep(1)

    async def unsubscribe(self) -> None:
        """Stop subscribing to events."""
        self.running = False
        if self.pubsub:
            await self.pubsub.unsubscribe(self.channel)
            await self.pubsub.close()


async def example_worker(event: MinIOEvent) -> None:
    """Example worker that processes MinIO events."""
    print(f"Processing event: {event.event_type}")
    print(f"  Source: {event.source_id}")
    print(f"  Workspace: {event.workspace_id}")
    print(f"  Object: {event.object_key}")