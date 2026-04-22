-- Memory System Database Schema
-- PostgreSQL + pgvector for 5-memory architecture

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- MEMORY 1: Active Context Cache (Redis, not here but tracking)
-- We store metadata about cache state in Postgres for audit
-- =====================================================

CREATE TABLE IF NOT EXISTS active_context_cache_audit (
    cache_key TEXT PRIMARY KEY,
    user_id UUID NOT NULL,
    session_id UUID NOT NULL,
    cache_type TEXT NOT NULL, -- 'recent_turns', 'entities_hot', 'working_cache', 'retrieval_cache'
    last_refreshed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    entry_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

-- =====================================================
-- MEMORY 2: Working Memory Snapshots
-- =====================================================

CREATE TABLE IF NOT EXISTS working_memory_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    conversation_id UUID NOT NULL,
    session_id UUID NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    objective TEXT,
    active_tasks JSONB NOT NULL DEFAULT '[]',
    constraints JSONB NOT NULL DEFAULT '[]',
    open_questions JSONB NOT NULL DEFAULT '[]',
    active_entities JSONB NOT NULL DEFAULT '[]',
    active_references JSONB NOT NULL DEFAULT '[]',
    summary TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_working_memory_conv_ver
ON working_memory_snapshots(conversation_id, version DESC);

CREATE INDEX IF NOT EXISTS idx_working_memory_session
ON working_memory_snapshots(session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_working_memory_user
ON working_memory_snapshots(user_id, created_at DESC);

-- =====================================================
-- MEMORY 3: Episodic Memory (Append-only event store)
-- =====================================================

CREATE TABLE IF NOT EXISTS episodic_events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    conversation_id UUID NOT NULL,
    session_id UUID NOT NULL,
    turn_id BIGINT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool', 'system')),
    event_type TEXT NOT NULL, -- 'message', 'decision', 'tool_result', 'task_update', 'error', 'handoff', 'state_change', 'artifact_reference'
    content TEXT NOT NULL,
    normalized_content TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    salience_score REAL NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding VECTOR(1536)
);

CREATE INDEX IF NOT EXISTS idx_episodic_conv_turn
ON episodic_events(conversation_id, turn_id);

CREATE INDEX IF NOT EXISTS idx_episodic_created_at
ON episodic_events(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_episodic_metadata_gin
ON episodic_events USING gin(metadata);

CREATE INDEX IF NOT EXISTS idx_episodic_embedding_hnsw
ON episodic_events USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_episodic_user_conv
ON episodic_events(user_id, conversation_id, created_at DESC);

-- =====================================================
-- MEMORY 4: Semantic Memory (Atomic, traceable memories)
-- =====================================================

CREATE TABLE IF NOT EXISTS semantic_memories (
    memory_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    scope_type TEXT NOT NULL CHECK (scope_type IN ('user', 'project', 'conversation', 'global')),
    scope_id UUID,
    memory_type TEXT NOT NULL, -- 'user_preference', 'user_profile_fact', 'project_fact', 'decision', 'constraint', 'commitment', 'successful_pattern', 'failure_pattern', 'domain_fact', 'policy_fact'
    subject TEXT,
    predicate TEXT,
    object_value TEXT,
    canonical_text TEXT NOT NULL,
    attributes JSONB NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0.5,
    source_count INTEGER NOT NULL DEFAULT 1,
    stability_class TEXT NOT NULL CHECK (stability_class IN ('volatile', 'durable', 'permanent')),
    status TEXT NOT NULL CHECK (status IN ('candidate', 'active', 'deprecated', 'contradicted', 'archived')),
    supersedes_memory_id UUID REFERENCES semantic_memories(memory_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_confirmed_at TIMESTAMPTZ,
    embedding VECTOR(1536)
);

CREATE INDEX IF NOT EXISTS idx_semantic_scope
ON semantic_memories(user_id, scope_type, scope_id, status);

CREATE INDEX IF NOT EXISTS idx_semantic_type
ON semantic_memories(memory_type, stability_class, status);

CREATE INDEX IF NOT EXISTS idx_semantic_attributes_gin
ON semantic_memories USING gin(attributes);

CREATE INDEX IF NOT EXISTS idx_semantic_embedding_hnsw
ON semantic_memories USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_semantic_user_status
ON semantic_memories(user_id, status, created_at DESC);

-- Provenance table for semantic memories
CREATE TABLE IF NOT EXISTS semantic_memory_sources (
    memory_id UUID NOT NULL REFERENCES semantic_memories(memory_id) ON DELETE CASCADE,
    event_id UUID NOT NULL REFERENCES episodic_events(event_id) ON DELETE CASCADE,
    source_strength REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id, event_id)
);

-- =====================================================
-- MEMORY 5: Relational/Temporal Memory (Neo4j sync tracking)
-- We store sync state in Postgres; Neo4j is the primary store
-- =====================================================

CREATE TABLE IF NOT EXISTS graph_sync_audit (
    sync_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    entity_type TEXT NOT NULL, -- 'User', 'Conversation', 'Session', 'Entity', 'Decision', 'Memory', 'Task'
    entity_id UUID NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('CREATE', 'UPDATE', 'DELETE', 'RELATE')),
    cypher_query TEXT,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL CHECK (status IN ('pending', 'synced', 'failed')),
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_graph_sync_pending
ON graph_sync_audit(status, synced_at) WHERE status = 'pending';

-- =====================================================
-- Hierarchical Digests
-- =====================================================

CREATE TABLE IF NOT EXISTS memory_digests (
    digest_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    conversation_id UUID,
    period_type TEXT NOT NULL CHECK (period_type IN ('session', 'week', 'month')),
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,
    title TEXT,
    summary TEXT NOT NULL,
    timeline JSONB NOT NULL DEFAULT '[]',
    open_loops JSONB NOT NULL DEFAULT '[]',
    decisions JSONB NOT NULL DEFAULT '[]',
    entities JSONB NOT NULL DEFAULT '[]',
    metadata JSONB NOT NULL DEFAULT '{}',
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_digest_user_period
ON memory_digests(user_id, period_type, period_start DESC);

CREATE INDEX IF NOT EXISTS idx_digest_conversation
ON memory_digests(conversation_id, period_start DESC);

CREATE INDEX IF NOT EXISTS idx_digest_embedding_hnsw
ON memory_digests USING hnsw (embedding vector_cosine_ops);

-- =====================================================
-- Consolidation tracking
-- =====================================================

CREATE TABLE IF NOT EXISTS consolidation_jobs (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    session_id UUID,
    conversation_id UUID,
    job_type TEXT NOT NULL CHECK (job_type IN ('session', 'nightly', 'weekly', 'monthly', 'dedup', 'conflict_resolution', 'promotion')),
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    items_processed INTEGER DEFAULT 0,
    memories_created INTEGER DEFAULT 0,
    conflicts_resolved INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_consolidation_pending
ON consolidation_jobs(status, created_at) WHERE status IN ('pending', 'running');

-- =====================================================
-- Memory conflicts tracking
-- =====================================================

CREATE TABLE IF NOT EXISTS memory_conflicts (
    conflict_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    memory_id_1 UUID NOT NULL REFERENCES semantic_memories(memory_id),
    memory_id_2 UUID NOT NULL REFERENCES semantic_memories(memory_id),
    conflict_type TEXT NOT NULL CHECK (conflict_type IN ('contradiction', 'supersedes', 'mergeable')),
    resolution_status TEXT NOT NULL CHECK (resolution_status IN ('detected', 'pending_resolution', 'resolved', 'ignored')),
    resolved_by UUID,
    resolution_notes TEXT,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_conflicts_pending
ON memory_conflicts(resolution_status, detected_at) WHERE resolution_status = 'pending_resolution';
