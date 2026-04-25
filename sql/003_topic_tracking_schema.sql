-- Topic Tracking Schema
-- Index conversations by topic for easy retrieval when returning to a previous topic

-- Topics table: defines all topics discussed
CREATE TABLE IF NOT EXISTS conversation_topics (
    topic_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    conversation_id UUID NOT NULL,
    topic_name TEXT NOT NULL,           -- e.g., "pedido #12345", "reclamo", "información cuenta"
    topic_keywords JSONB NOT NULL DEFAULT '[]',  -- keywords that identify this topic
    topic_summary TEXT,                -- brief summary of what was discussed
    topic_embedding VECTOR(1536),      -- semantic embedding for topic
    first_mention_turn INT,             -- turn number when topic first appeared
    last_mention_turn INT,              -- turn number of most recent mention
    mention_count INT DEFAULT 1,       -- how many times topic was mentioned
    sentiment_score REAL DEFAULT 0.5,  -- overall sentiment (0-1)
    status TEXT NOT NULL DEFAULT 'active',  -- 'active', 'resolved', 'abandoned'
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_topics_user
ON conversation_topics(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_topics_conversation
ON conversation_topics(conversation_id, last_mention_turn DESC);

CREATE INDEX IF NOT EXISTS idx_topics_embedding_hnsw
ON conversation_topics USING hnsw (topic_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_topics_name
ON conversation_topics(user_id, topic_name);

CREATE INDEX IF NOT EXISTS idx_topics_status
ON conversation_topics(status, last_mention_turn DESC) WHERE status = 'active';

-- Topic references: links topics to specific episodic events
CREATE TABLE IF NOT EXISTS topic_event_references (
    ref_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    topic_id UUID NOT NULL REFERENCES conversation_topics(topic_id) ON DELETE CASCADE,
    event_id UUID NOT NULL REFERENCES episodic_events(event_id) ON DELETE CASCADE,
    event_turn INT NOT NULL,
    relevance_score REAL DEFAULT 1.0,   -- how relevant this event is to the topic
    context_excerpt TEXT,              -- relevant excerpt from event
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_topic_refs_topic
ON topic_event_references(topic_id, relevance_score DESC);

CREATE INDEX IF NOT EXISTS idx_topic_refs_event
ON topic_event_references(event_id);

-- Conversation summaries: periodic digests
CREATE TABLE IF NOT EXISTS conversation_summaries (
    summary_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    conversation_id UUID NOT NULL,
    session_id UUID NOT NULL,
    turn_range_start INT NOT NULL,      -- turn number range for this summary
    turn_range_end INT NOT NULL,
    summary_text TEXT NOT NULL,        -- the actual summary
    summary_embedding VECTOR(1536),    -- embedding for semantic search
    summary_type TEXT NOT NULL DEFAULT 'turn_based',  -- 'turn_based', 'topic_shift', 'milestone'
    topics_covered JSONB DEFAULT '[]', -- list of topic_ids covered
    key_decisions JSONB DEFAULT '[]',  -- important decisions made
    open_questions JSONB DEFAULT '[]',  -- unresolved items
    entities_mentioned JSONB DEFAULT '[]', -- entities discussed
    sentiment_overall REAL DEFAULT 0.5,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_summary_conversation
ON conversation_summaries(conversation_id, turn_range_start DESC);

CREATE INDEX IF NOT EXISTS idx_summary_user
ON conversation_summaries(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_summary_embedding_hnsw
ON conversation_summaries USING hnsw (summary_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_summary_session
ON conversation_summaries(session_id, turn_range_start DESC);