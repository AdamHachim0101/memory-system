"""
Topic Tracking Service
Manages conversation topics for easier retrieval when returning to a previous topic.
"""

import uuid
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

import asyncpg

from ..models import ConversationTopic, ConversationSummary, TopicEventReference


class TopicTrackingService:
    """
    Tracks topics across conversation turns and generates summaries.
    
    Features:
    - Topic detection and indexing
    - Topic-to-event linking
    - Periodic conversation summaries
    - Topic-based retrieval
    """
    
    def __init__(self, pool: asyncpg.Pool, embedding_service):
        self.pool = pool
        self.embedding_service = embedding_service
        
        # Topic keywords for detection
        self.topic_keywords = {
            "pedido": ["pedido", "orden", "compra", "envío", "shipping"],
            "reclamo": ["reclamo", "queja", "problema", "error", "incidencia"],
            "pago": ["pago", "factura", "cobro", "cargo", "refund"],
            "cuenta": ["cuenta", "perfil", "usuario", "login", "contraseña"],
            "producto": ["producto", "artículo", "ítem", "stock"],
            "envio": ["envío", "delivery", "entrega", "tracking"],
        }
    
    async def detect_and_track_topics(
        self,
        user_id: str,
        conversation_id: str,
        turn_id: int,
        user_message: str,
        assistant_message: str
    ) -> List[ConversationTopic]:
        """
        Detect topics in a conversation turn and update tracking.
        """
        import sys
        text = f"{user_message} {assistant_message}".lower()
        sys.stderr.write(f"DEBUG detect_and_track_topics: text='{text[:100]}'\n")
        sys.stderr.flush()
        detected_topics = []
        
        for topic_name, keywords in self.topic_keywords.items():
            matches = [kw for kw in keywords if kw in text]
            if matches:
                sys.stderr.write(f"DEBUG: Found topic '{topic_name}' with keywords {matches}\n")
                sys.stderr.flush()
                topic = await self._get_or_create_topic(
                    user_id, conversation_id, topic_name, user_message, turn_id
                )
                if topic:
                    detected_topics.append(topic)
        
        sys.stderr.write(f"DEBUG: detect_and_track_topics returning {len(detected_topics)} topics\n")
        sys.stderr.flush()
        return detected_topics
    
    async def _get_or_create_topic(
        self,
        user_id: str,
        conversation_id: str,
        topic_name: str,
        context_message: str,
        turn_id: int
    ) -> Optional[ConversationTopic]:
        """Get existing topic or create new one."""
        async with self.pool.acquire() as conn:
            # Check if topic exists for this conversation
            existing = await conn.fetchrow(
                """
                SELECT * FROM conversation_topics 
                WHERE user_id = $1 AND conversation_id = $2 AND topic_name = $3 AND status = 'active'
                """,
                uuid.UUID(user_id),
                uuid.UUID(conversation_id),
                topic_name
            )
            
            if existing:
                # Update existing topic
                await conn.execute(
                    """
                    UPDATE conversation_topics 
                    SET last_mention_turn = $4, mention_count = mention_count + 1, updated_at = now()
                    WHERE topic_id = $1
                    """,
                    existing["topic_id"],
                    turn_id
                )
                return ConversationTopic(
                    topic_id=str(existing["topic_id"]),
                    user_id=str(existing["user_id"]),
                    conversation_id=str(existing["conversation_id"]),
                    topic_name=existing["topic_name"],
                    topic_keywords=json.loads(existing["topic_keywords"]) if isinstance(existing["topic_keywords"], str) else existing["topic_keywords"],
                    first_mention_turn=existing["first_mention_turn"],
                    last_mention_turn=turn_id,
                    mention_count=existing["mention_count"] + 1,
                    status=existing["status"]
                )
            else:
                # Create new topic
                embedding = await self.embedding_service.get_embedding(f"{topic_name}: {context_message[:200]}")
                embedding_param = embedding if embedding is None else str(embedding)
                
                row = await conn.fetchrow(
                    """
                    INSERT INTO conversation_topics 
                    (user_id, conversation_id, topic_name, topic_keywords, topic_summary, topic_embedding, 
                     first_mention_turn, last_mention_turn, mention_count)
                    VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $7, 1)
                    RETURNING *
                    """,
                    uuid.UUID(user_id),
                    uuid.UUID(conversation_id),
                    topic_name,
                    json.dumps(self.topic_keywords.get(topic_name, [])),
                    context_message[:200],
                    embedding_param,
                    turn_id
                )
                
                return ConversationTopic(
                    topic_id=str(row["topic_id"]),
                    user_id=str(row["user_id"]),
                    conversation_id=str(row["conversation_id"]),
                    topic_name=row["topic_name"],
                    topic_keywords=json.loads(row["topic_keywords"]) if isinstance(row["topic_keywords"], str) else row["topic_keywords"],
                    topic_summary=row["topic_summary"],
                    first_mention_turn=turn_id,
                    last_mention_turn=turn_id,
                    mention_count=1,
                    status=row["status"]
                )
    
    async def link_topic_event(
        self,
        topic_id: str,
        event_id: str,
        event_turn: int,
        context_excerpt: str,
        relevance_score: float = 1.0
    ) -> None:
        """Link a topic to a specific episodic event."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO topic_event_references (topic_id, event_id, event_turn, relevance_score, context_excerpt)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT DO NOTHING
                """,
                uuid.UUID(topic_id),
                uuid.UUID(event_id),
                event_turn,
                relevance_score,
                context_excerpt[:200] if context_excerpt else None
            )
    
    async def get_topics_for_conversation(
        self,
        conversation_id: str,
        status: str = "active"
    ) -> List[ConversationTopic]:
        """Get all topics for a conversation."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM conversation_topics 
                WHERE conversation_id = $1 AND status = $2
                ORDER BY last_mention_turn DESC
                """,
                uuid.UUID(conversation_id),
                status
            )
            
            return [
                ConversationTopic(
                    topic_id=str(r["topic_id"]),
                    user_id=str(r["user_id"]),
                    conversation_id=str(r["conversation_id"]),
                    topic_name=r["topic_name"],
                    topic_keywords=r["topic_keywords"],
                    topic_summary=r["topic_summary"],
                    first_mention_turn=r["first_mention_turn"],
                    last_mention_turn=r["last_mention_turn"],
                    mention_count=r["mention_count"],
                    sentiment_score=r["sentiment_score"],
                    status=r["status"],
                    metadata=r["metadata"]
                )
                for r in rows
            ]
    
    async def get_topics_by_semantic_search(
        self,
        user_id: str,
        query: str,
        limit: int = 5
    ) -> List[ConversationTopic]:
        """Search topics by semantic similarity."""
        query_embedding = await self.embedding_service.get_embedding(query)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM conversation_topics 
                WHERE user_id = $1 AND topic_embedding IS NOT NULL
                ORDER BY topic_embedding <=> $2::vector
                LIMIT $3
                """,
                uuid.UUID(user_id),
                query_embedding,
                limit
            )
            
            return [
                ConversationTopic(
                    topic_id=str(r["topic_id"]),
                    user_id=str(r["user_id"]),
                    conversation_id=str(r["conversation_id"]),
                    topic_name=r["topic_name"],
                    topic_keywords=r["topic_keywords"],
                    topic_summary=r["topic_summary"],
                    first_mention_turn=r["first_mention_turn"],
                    last_mention_turn=r["last_mention_turn"],
                    mention_count=r["mention_count"],
                    sentiment_score=r["sentiment_score"],
                    status=r["status"],
                    metadata=r["metadata"]
                )
                for r in rows
            ]
    
    async def generate_conversation_summary(
        self,
        user_id: str,
        conversation_id: str,
        session_id: str,
        turn_start: int,
        turn_end: int,
        episodic_events: List[Dict]
    ) -> Optional[ConversationSummary]:
        """Generate a summary for a range of conversation turns."""
        if not episodic_events:
            return None
        
        # Combine events into text for summarization
        event_texts = []
        for e in episodic_events:
            role = e.get("role", "unknown")
            content = e.get("content", "")[:200]
            event_texts.append(f"{role}: {content}")
        
        combined_text = "\n".join(event_texts[-10:])  # Last 10 events
        
        # Create summary text (in production, use LLM for better summarization)
        summary_text = self._generate_basic_summary(combined_text)
        
        # Generate embedding
        summary_embedding = await self.embedding_service.get_embedding(summary_text)
        
        # Extract entities mentioned
        entities = self._extract_entities(combined_text)
        
        # Detect topics covered
        topics_covered = await self._get_topic_ids_for_turns(
            conversation_id, turn_start, turn_end
        )
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO conversation_summaries 
                (user_id, conversation_id, session_id, turn_range_start, turn_range_end,
                 summary_text, summary_embedding, summary_type, topics_covered, entities_mentioned)
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector, 'turn_based', $8, $9)
                RETURNING *
                """,
                uuid.UUID(user_id),
                uuid.UUID(conversation_id),
                uuid.UUID(session_id),
                turn_start,
                turn_end,
                summary_text,
                summary_embedding,
                json.dumps(topics_covered),
                json.dumps(entities)
            )
            
            return ConversationSummary(
                summary_id=str(row["summary_id"]),
                user_id=str(row["user_id"]),
                conversation_id=str(row["conversation_id"]),
                session_id=str(row["session_id"]),
                turn_range_start=row["turn_range_start"],
                turn_range_end=row["turn_range_end"],
                summary_text=row["summary_text"],
                summary_embedding=row["summary_embedding"],
                summary_type=row["summary_type"],
                topics_covered=row["topics_covered"],
                key_decisions=row["key_decisions"],
                open_questions=row["open_questions"],
                entities_mentioned=row["entities_mentioned"],
                sentiment_overall=row["sentiment_overall"]
            )
    
    async def get_summaries_for_conversation(
        self,
        conversation_id: str,
        limit: int = 5
    ) -> List[ConversationSummary]:
        """Get conversation summaries ordered by turn range."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM conversation_summaries 
                WHERE conversation_id = $1
                ORDER BY turn_range_start DESC
                LIMIT $2
                """,
                uuid.UUID(conversation_id),
                limit
            )
            
            return [
                ConversationSummary(
                    summary_id=str(r["summary_id"]),
                    user_id=str(r["user_id"]),
                    conversation_id=str(r["conversation_id"]),
                    session_id=str(r["session_id"]),
                    turn_range_start=r["turn_range_start"],
                    turn_range_end=r["turn_range_end"],
                    summary_text=r["summary_text"],
                    summary_embedding=r["summary_embedding"],
                    summary_type=r["summary_type"],
                    topics_covered=r["topics_covered"],
                    key_decisions=r["key_decisions"],
                    open_questions=r["open_questions"],
                    entities_mentioned=r["entities_mentioned"],
                    sentiment_overall=r["sentiment_overall"],
                    metadata=r["metadata"]
                )
                for r in rows
            ]
    
    async def get_summaries_by_semantic_search(
        self,
        user_id: str,
        query: str,
        limit: int = 3
    ) -> List[ConversationSummary]:
        """Search summaries by semantic similarity."""
        query_embedding = await self.embedding_service.get_embedding(query)
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM conversation_summaries 
                WHERE user_id = $1 AND summary_embedding IS NOT NULL
                ORDER BY summary_embedding <=> $2::vector
                LIMIT $3
                """,
                uuid.UUID(user_id),
                query_embedding,
                limit
            )
            
            return [
                ConversationSummary(
                    summary_id=str(r["summary_id"]),
                    user_id=str(r["user_id"]),
                    conversation_id=str(r["conversation_id"]),
                    session_id=str(r["session_id"]),
                    turn_range_start=r["turn_range_start"],
                    turn_range_end=r["turn_range_end"],
                    summary_text=r["summary_text"],
                    summary_embedding=r["summary_embedding"],
                    summary_type=r["summary_type"],
                    topics_covered=r["topics_covered"],
                    key_decisions=r["key_decisions"],
                    open_questions=r["open_questions"],
                    entities_mentioned=r["entities_mentioned"],
                    sentiment_overall=r["sentiment_overall"]
                )
                for r in rows
            ]
    
    async def _get_topic_ids_for_turns(
        self,
        conversation_id: str,
        turn_start: int,
        turn_end: int
    ) -> List[str]:
        """Get topic IDs mentioned in a turn range."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT t.topic_id FROM conversation_topics t
                JOIN topic_event_references r ON t.topic_id = r.topic_id
                WHERE t.conversation_id = $1 AND r.event_turn BETWEEN $2 AND $3
                """,
                uuid.UUID(conversation_id),
                turn_start,
                turn_end
            )
            return [str(r["topic_id"]) for r in rows]
    
    def _generate_basic_summary(self, text: str) -> str:
        """Basic summary generation (fallback when LLM unavailable)."""
        lines = text.split("\n")
        user_lines = [l for l in lines if l.startswith("user:")]
        assistant_lines = [l for l in lines if l.startswith("assistant:")]
        
        summary_parts = []
        
        if len(user_lines) <= 3:
            summary_parts.append(f"Conversation with {len(user_lines)} user messages.")
        else:
            summary_parts.append(f"Extended conversation with {len(user_lines)} user messages.")
        
        # Add topic hints based on keywords
        text_lower = text.lower()
        if "pedido" in text_lower or "orden" in text_lower:
            summary_parts.append("Topic: Orders/Pedidos")
        if "reclamo" in text_lower or "problema" in text_lower:
            summary_parts.append("Topic: Complaints/Claims")
        if "pago" in text_lower or "factura" in text_lower:
            summary_parts.append("Topic: Payments/Billing")
        
        return " ".join(summary_parts)
    
    def _extract_entities(self, text: str) -> List[str]:
        """Basic entity extraction (simplified)."""
        entities = []
        
        # Look for order/pedido patterns
        import re
        order_patterns = re.findall(r'(?:pedido|orden|order)[#\s]+(\w+)', text, re.IGNORECASE)
        entities.extend([f"pedido:{p}" for p in order_patterns])
        
        # Look for numbers that might be IDs
        id_patterns = re.findall(r'#(\d{5,})', text)
        entities.extend([f"id:{i}" for i in id_patterns])
        
        return list(set(entities))[:10]  # Dedupe and limit
    
    async def resolve_topic(self, topic_id: str) -> None:
        """Mark a topic as resolved."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE conversation_topics SET status = 'resolved', updated_at = now() WHERE topic_id = $1",
                uuid.UUID(topic_id)
            )
    
    async def abandon_topic(self, topic_id: str) -> None:
        """Mark a topic as abandoned."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE conversation_topics SET status = 'abandoned', updated_at = now() WHERE topic_id = $1",
                uuid.UUID(topic_id)
            )