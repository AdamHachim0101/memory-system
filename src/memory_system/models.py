import uuid
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


class MemoryScopeType(str, Enum):
    USER = "user"
    PROJECT = "project"
    CONVERSATION = "conversation"
    GLOBAL = "global"


class MemoryType(str, Enum):
    USER_PREFERENCE = "user_preference"
    USER_PROFILE_FACT = "user_profile_fact"
    PROJECT_FACT = "project_fact"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    COMMITMENT = "commitment"
    SUCCESSFUL_PATTERN = "successful_pattern"
    FAILURE_PATTERN = "failure_pattern"
    DOMAIN_FACT = "domain_fact"
    POLICY_FACT = "policy_fact"


class StabilityClass(str, Enum):
    VOLATILE = "volatile"
    DURABLE = "durable"
    PERMANENT = "permanent"


class MemoryStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    CONTRADICTED = "contradicted"
    ARCHIVED = "archived"


class EventType(str, Enum):
    MESSAGE = "message"
    DECISION = "decision"
    TOOL_RESULT = "tool_result"
    TASK_UPDATE = "task_update"
    ERROR = "error"
    HANDOFF = "handoff"
    STATE_CHANGE = "state_change"
    ARTIFACT_REFERENCE = "artifact_reference"


class RetrievalProfile(str, Enum):
    LIGHT = "light"
    TASK = "task"
    DEEP = "deep"
    AUDIT = "audit"


class TurnContext(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    turn_id: int = 0


class WorkingMemorySnapshot(BaseModel):
    snapshot_id: Optional[str] = None
    user_id: str
    agent_id: str
    conversation_id: str
    session_id: str
    version: int = 1
    objective: Optional[str] = None
    active_tasks: list[dict] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    active_entities: list[dict] = Field(default_factory=list)
    active_references: list[str] = Field(default_factory=list)
    summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EpisodicEvent(BaseModel):
    event_id: Optional[str] = None
    user_id: str
    agent_id: str
    conversation_id: str
    session_id: str
    turn_id: int
    role: Literal["user", "assistant", "tool", "system"]
    event_type: str
    content: str
    normalized_content: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    salience_score: float = 0.0
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SemanticMemory(BaseModel):
    memory_id: Optional[str] = None
    user_id: str
    agent_id: str
    scope_type: MemoryScopeType
    scope_id: Optional[str] = None
    memory_type: MemoryType
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object_value: Optional[str] = None
    canonical_text: str
    attributes: dict = Field(default_factory=dict)
    confidence: float = 0.5
    source_count: int = 1
    stability_class: StabilityClass = StabilityClass.DURABLE
    status: MemoryStatus = MemoryStatus.CANDIDATE
    supersedes_memory_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_confirmed_at: Optional[datetime] = None
    embedding: Optional[list[float]] = None


class MemoryDigest(BaseModel):
    digest_id: Optional[str] = None
    user_id: str
    conversation_id: Optional[str] = None
    period_type: Literal["session", "week", "month"]
    period_start: datetime
    period_end: datetime
    title: Optional[str] = None
    summary: str
    timeline: list[dict] = Field(default_factory=list)
    open_loops: list[str] = Field(default_factory=list)
    decisions: list[dict] = Field(default_factory=list)
    entities: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationTopic(BaseModel):
    topic_id: Optional[str] = None
    user_id: str
    conversation_id: str
    topic_name: str
    topic_keywords: list[str] = Field(default_factory=list)
    topic_summary: Optional[str] = None
    topic_embedding: Optional[list[float]] = None
    first_mention_turn: int = 0
    last_mention_turn: int = 0
    mention_count: int = 1
    sentiment_score: float = 0.5
    status: str = "active"
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ConversationSummary(BaseModel):
    summary_id: Optional[str] = None
    user_id: str
    conversation_id: str
    session_id: str
    turn_range_start: int
    turn_range_end: int
    summary_text: str
    summary_embedding: Optional[list[float]] = None
    summary_type: str = "turn_based"
    topics_covered: list[str] = Field(default_factory=list)
    key_decisions: list[dict] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    entities_mentioned: list[str] = Field(default_factory=list)
    sentiment_overall: float = 0.5
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TopicEventReference(BaseModel):
    ref_id: Optional[str] = None
    topic_id: str
    event_id: str
    event_turn: int
    relevance_score: float = 1.0
    context_excerpt: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IngestTurnRequest(BaseModel):
    user_id: str
    agent_id: str
    conversation_id: str
    session_id: str
    turn_id: int
    user_message: str
    assistant_message: str
    tool_events: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class IngestTurnResponse(BaseModel):
    working_snapshot_id: str
    episodic_event_ids: list[str]
    candidate_count: int


class RetrieveRequest(BaseModel):
    user_id: str
    agent_id: str
    conversation_id: str
    session_id: str
    query: str
    entities: list[str] = Field(default_factory=list)
    profile: RetrievalProfile = RetrievalProfile.TASK
    limit: int = 20


class RetrieveResponse(BaseModel):
    working_memory: Optional[WorkingMemorySnapshot] = None
    semantic_memories: list[SemanticMemory] = Field(default_factory=list)
    episodic_events: list[EpisodicEvent] = Field(default_factory=list)
    digests: list[MemoryDigest] = Field(default_factory=list)
    graph_context: list[dict] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)


class MemoryCandidate(BaseModel):
    memory_type: MemoryType
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object_value: Optional[str] = None
    canonical_text: str
    stability_class: StabilityClass = StabilityClass.DURABLE
    confidence: float = 0.5
    attributes: dict = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    working_memory_patch: dict = Field(default_factory=dict)
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    entities: list[dict] = Field(default_factory=list)
    salience_score: float = 0.0


__all__ = [
    "TurnContext",
    "IngestTurnRequest",
    "IngestTurnResponse",
    "RetrieveRequest",
    "RetrieveResponse",
    "WorkingMemorySnapshot",
    "EpisodicEvent",
    "SemanticMemory",
    "MemoryDigest",
    "MemoryCandidate",
    "ExtractionResult",
    "RetrievalProfile",
    "MemoryScopeType",
    "MemoryType",
    "StabilityClass",
    "MemoryStatus",
    "EventType",
    "ConversationTopic",
    "ConversationSummary",
    "TopicEventReference",
]
