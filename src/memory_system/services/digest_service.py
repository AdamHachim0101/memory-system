"""
Conversation Digest Service
Generates periodic summaries (digests) of conversations for better long-term memory.
"""

import uuid
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

import asyncpg


class DigestService:
    """
    Generates and manages conversation digests.
    
    Features:
    - Periodic summary generation (every N turns)
    - Topic-shift detection for summary triggers
    - Milestone detection (decisions, commitments, etc.)
    - Cross-conversation topic retrieval
    """
    
    def __init__(self, pool: asyncpg.Pool, embedding_service, llm_service=None):
        self.pool = pool
        self.embedding_service = embedding_service
        self.llm_service = llm_service  # For LLM-based summarization
    
    async def should_generate_summary(
        self,
        conversation_id: str,
        current_turn: int,
        last_summary_turn: int,
        min_turns_between_summaries: int = 5
    ) -> bool:
        """Determine if a summary should be generated."""
        if current_turn - last_summary_turn >= min_turns_between_summaries:
            return True
        
        # Check for topic shift
        if last_summary_turn > 0:
            recent_topics = await self._get_recent_topic_shift(conversation_id, last_summary_turn, current_turn)
            if len(recent_topics) > 2:  # Multiple topic changes
                return True
        
        return False
    
    async def _get_recent_topic_shift(
        self,
        conversation_id: str,
        turn_start: int,
        turn_end: int
    ) -> List[str]:
        """Detect topic changes in a turn range."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT topic_name, COUNT(*) as cnt
                FROM conversation_topics
                WHERE conversation_id = $1 
                  AND first_mention_turn BETWEEN $2 AND $3
                GROUP BY topic_name
                """,
                uuid.UUID(conversation_id),
                turn_start,
                turn_end
            )
            return [r["topic_name"] for r in rows]
    
    async def generate_summary(
        self,
        user_id: str,
        conversation_id: str,
        session_id: str,
        turn_start: int,
        turn_end: int,
        use_llm: bool = True
    ) -> Optional[str]:
        """
        Generate a summary for a turn range.
        Returns the summary_id.
        """
        # Get events for the turn range
        events = await self._get_conversation_events(conversation_id, turn_start, turn_end)
        
        if not events:
            return None
        
        # Build summary text
        if use_llm and self.llm_service:
            summary_text = await self._generate_llm_summary(events)
        else:
            summary_text = self._generate_basic_summary(events)
        
        # Generate embedding
        summary_embedding = await self.embedding_service.get_embedding(summary_text)
        embedding_param = summary_embedding if summary_embedding is None else str(summary_embedding)
        
        # Extract metadata
        topics = self._extract_topics_from_events(events)
        entities = self._extract_entities_from_events(events)
        decisions = self._extract_decisions_from_events(events)
        open_questions = self._extract_open_questions(events)
        
        # Calculate sentiment
        sentiment = self._calculate_sentiment(events)
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO conversation_summaries 
                (user_id, conversation_id, session_id, turn_range_start, turn_range_end,
                 summary_text, summary_embedding, summary_type, topics_covered, 
                 key_decisions, open_questions, entities_mentioned, sentiment_overall)
                VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9, $10, $11, $12, $13)
                RETURNING summary_id
                """,
                uuid.UUID(user_id),
                uuid.UUID(conversation_id),
                uuid.UUID(session_id),
                turn_start,
                turn_end,
                summary_text,
                embedding_param,
                "turn_based",
                json.dumps(topics),
                json.dumps(decisions),
                json.dumps(open_questions),
                json.dumps(entities),
                sentiment
            )
        
        return str(row["summary_id"])
    
    async def generate_topic_shift_summary(
        self,
        user_id: str,
        conversation_id: str,
        session_id: str,
        turn_id: int,
        old_topic: str,
        new_topic: str
    ) -> Optional[str]:
        """Generate a summary when topic shifts significantly."""
        # Get last N turns before the shift
        events = await self._get_conversation_events(conversation_id, max(0, turn_id - 5), turn_id)
        
        if not events:
            return None
        
        summary_text = f"Topic shifted from '{old_topic}' to '{new_topic}'. "
        summary_text += self._generate_basic_summary(events)
        
        summary_embedding = await self.embedding_service.get_embedding(summary_text)
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO conversation_summaries 
                (user_id, conversation_id, session_id, turn_range_start, turn_range_end,
                 summary_text, summary_embedding, summary_type)
                VALUES ($1, $2, $3, $4, $4, $5, $6::vector, 'topic_shift')
                RETURNING summary_id
                """,
                uuid.UUID(user_id),
                uuid.UUID(conversation_id),
                uuid.UUID(session_id),
                turn_id,
                summary_text,
                summary_embedding
            )
        
        return str(row["summary_id"])
    
    async def generate_milestone_summary(
        self,
        user_id: str,
        conversation_id: str,
        session_id: str,
        turn_id: int,
        milestone_type: str,
        milestone_details: Dict[str, Any]
    ) -> Optional[str]:
        """Generate a summary for a milestone (decision made, task completed, etc.)."""
        summary_text = f"Milestone: {milestone_type}. "
        summary_text += f"Details: {json.dumps(milestone_details)}"
        
        summary_embedding = await self.embedding_service.get_embedding(summary_text)
        
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO conversation_summaries 
                (user_id, conversation_id, session_id, turn_range_start, turn_range_end,
                 summary_text, summary_embedding, summary_type, metadata)
                VALUES ($1, $2, $3, $4, $4, $5, $6::vector, 'milestone', $7)
                RETURNING summary_id
                """,
                uuid.UUID(user_id),
                uuid.UUID(conversation_id),
                uuid.UUID(session_id),
                turn_id,
                summary_text,
                summary_embedding,
                json.dumps({"milestone_type": milestone_type, "details": milestone_details})
            )
        
        return str(row["summary_id"])
    
    async def get_recent_summaries(
        self,
        user_id: str,
        conversation_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """Get recent summaries for a conversation."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM conversation_summaries 
                WHERE user_id = $1 AND conversation_id = $2
                ORDER BY turn_range_start DESC
                LIMIT $3
                """,
                uuid.UUID(user_id),
                uuid.UUID(conversation_id),
                limit
            )
            
            return [
                {
                    "summary_id": str(r["summary_id"]),
                    "turn_range": f"{r['turn_range_start']}-{r['turn_range_end']}",
                    "summary_text": r["summary_text"],
                    "summary_type": r["summary_type"],
                    "topics_covered": r["topics_covered"],
                    "key_decisions": r["key_decisions"],
                    "created_at": r["created_at"].isoformat()
                }
                for r in rows
            ]
    
    async def get_summaries_for_topic(
        self,
        user_id: str,
        topic_name: str,
        limit: int = 3
    ) -> List[Dict]:
        """Get summaries covering a specific topic."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT s.* FROM conversation_summaries s
                WHERE s.user_id = $1 AND $2 = ANY(s.topics_covered)
                ORDER BY s.created_at DESC
                LIMIT $3
                """,
                uuid.UUID(user_id),
                topic_name,
                limit
            )
            
            return [
                {
                    "summary_id": str(r["summary_id"]),
                    "conversation_id": str(r["conversation_id"]),
                    "turn_range": f"{r['turn_range_start']}-{r['turn_range_end']}",
                    "summary_text": r["summary_text"],
                    "created_at": r["created_at"].isoformat()
                }
                for r in rows
            ]
    
    async def _get_conversation_events(
        self,
        conversation_id: str,
        turn_start: int,
        turn_end: int
    ) -> List[Dict]:
        """Get conversation events for a turn range."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT turn_id, role, content 
                FROM episodic_events 
                WHERE conversation_id = $1 AND turn_id BETWEEN $2 AND $3
                ORDER BY turn_id, created_at
                """,
                uuid.UUID(conversation_id),
                turn_start,
                turn_end
            )
            
            return [
                {"turn_id": r["turn_id"], "role": r["role"], "content": r["content"]}
                for r in rows
            ]
    
    async def _generate_llm_summary(self, events: List[Dict]) -> str:
        """Use LLM to generate a better summary."""
        if not self.llm_service:
            return self._generate_basic_summary(events)
        
        # Build conversation text
        lines = []
        for e in events:
            role = e.get("role", "unknown").upper()
            content = e.get("content", "")[:300]
            lines.append(f"{role}: {content}")
        
        conversation_text = "\n".join(lines)
        
        prompt = f"""Summarize this conversation briefly, capturing:
1. Main topics discussed
2. Key decisions made
3. Any outstanding questions or follow-ups

Keep it concise (2-3 sentences max).

Conversation:
{conversation_text[:2000]}
"""
        
        try:
            response = await self.llm_service.generate(prompt)
            return response.strip()
        except Exception:
            return self._generate_basic_summary(events)
    
    def _generate_basic_summary(self, events: List[Dict]) -> str:
        """Generate a basic summary without LLM."""
        if not events:
            return "Empty conversation segment."
        
        turn_count = len(set(e.get("turn_id", 0) for e in events))
        user_count = sum(1 for e in events if e.get("role") == "user")
        
        summary_parts = [
            f"Segment of {turn_count} turns with {user_count} user messages."
        ]
        
        # Detect topic patterns
        text = " ".join(e.get("content", "").lower() for e in events)
        
        if any(w in text for w in ["pedido", "orden", "compra"]):
            summary_parts.append("Topic: Orders/Purchases")
        if any(w in text for w in ["reclamo", "problema", "error"]):
            summary_parts.append("Topic: Issues/Complaints")
        if any(w in text for w in ["pago", "factura", "cobro"]):
            summary_parts.append("Topic: Payments/Billing")
        if any(w in text for w in ["envío", "enviar", "delivery"]):
            summary_parts.append("Topic: Shipping/Delivery")
        if any(w in text for w in ["cuenta", "perfil", "usuario"]):
            summary_parts.append("Topic: Account/Profile")
        
        return " ".join(summary_parts)
    
    def _extract_topics_from_events(self, events: List[Dict]) -> List[str]:
        """Extract topic indicators from events."""
        text = " ".join(e.get("content", "").lower() for e in events)
        
        topics = []
        topic_keywords = {
            "orders": ["pedido", "orden", "compra", "shipping"],
            "complaints": ["reclamo", "queja", "problema", "incidencia"],
            "payments": ["pago", "factura", "cobro", "refund"],
            "shipping": ["envío", "enviar", "delivery", "tracking"],
            "account": ["cuenta", "perfil", "usuario", "login"],
            "products": ["producto", "artículo", "stock"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text for kw in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_entities_from_events(self, events: List[Dict]) -> List[str]:
        """Extract entity references from events."""
        import re
        
        text = " ".join(e.get("content", "") for e in events)
        
        entities = []
        
        # Order numbers
        order_patterns = re.findall(r'(?:pedido|orden|order)[#\s]+(\w+)', text, re.IGNORECASE)
        entities.extend([f"order:{p}" for p in order_patterns])
        
        # IDs (numbers with 5+ digits)
        id_patterns = re.findall(r'#(\d{5,})', text)
        entities.extend([f"id:{i}" for i in id_patterns])
        
        # Email patterns
        email_patterns = re.findall(r'[\w.-]+@[\w.-]+\.\w+', text)
        entities.extend(email_patterns)
        
        return list(set(entities))[:10]
    
    def _extract_decisions_from_events(self, events: List[Dict]) -> List[Dict]:
        """Extract decisions made during conversation."""
        decisions = []
        
        decision_keywords = ["decidí", "decide", "proceder", "confirmo", "acordamos", "vamos a"]
        
        for e in events:
            content = e.get("content", "").lower()
            if any(kw in content for kw in decision_keywords):
                decisions.append({
                    "turn": e.get("turn_id", 0),
                    "text": e.get("content", "")[:150]
                })
        
        return decisions[:5]  # Limit to 5 decisions
    
    def _extract_open_questions(self, events: List[Dict]) -> List[str]:
        """Extract open questions from conversation."""
        questions = []
        
        question_markers = ["?", "puedes", "podrías", "podemos", "necesito saber", "dime"]
        
        for e in events:
            if e.get("role") == "user":
                content = e.get("content", "")
                if any(m in content.lower() for m in question_markers):
                    questions.append(content[:100])
        
        return list(set(questions))[:5]  # Dedupe and limit
    
    def _calculate_sentiment(self, events: List[Dict]) -> float:
        """Basic sentiment calculation."""
        positive_words = ["gracias", "perfecto", "excelente", "ok", "bien", "genial"]
        negative_words = ["problema", "error", "mal", "reclamo", "queja", "incorrecto"]
        
        text = " ".join(e.get("content", "").lower() for e in events)
        
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.5
        
        return pos_count / total