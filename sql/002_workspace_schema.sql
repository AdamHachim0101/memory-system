-- Source Workspace Engine Schema
-- Extensión del schema existente de memoria

-- =============================================================================
-- WORKSPACES
-- =============================================================================

create table if not exists source_workspaces (
    workspace_id uuid primary key default gen_random_uuid(),
    owner_id uuid not null,
    name text not null,
    description text,
    shared_sources jsonb default '[]',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index idx_workspaces_owner on source_workspaces(owner_id);

-- =============================================================================
-- SOURCE DOCUMENTS
-- =============================================================================

create table if not exists source_documents (
    source_id uuid primary key default gen_random_uuid(),
    workspace_id uuid not null references source_workspaces(workspace_id) on delete cascade,
    source_type text not null check (source_type in ('pdf', 'docx', 'md', 'txt', 'html', 'url', 'code', 'json', 'csv', 'other')),
    title text not null,
    canonical_uri text,
    file_hash text,
    mime_type text,
    language text,
    status text not null default 'pending' check (status in ('pending', 'processing', 'ready', 'failed', 'archived')),
    size_bytes bigint,
    metadata jsonb default '{}',
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index idx_documents_workspace on source_documents(workspace_id, status);
create index idx_documents_hash on source_documents(file_hash);
create index idx_documents_status on source_documents(status);

-- =============================================================================
-- SOURCE SECTIONS
-- =============================================================================

create table if not exists source_sections (
    section_id uuid primary key default gen_random_uuid(),
    source_id uuid not null references source_documents(source_id) on delete cascade,
    parent_section_id uuid references source_sections(section_id),
    level int not null default 0,
    title text,
    ordinal int,
    path text,
    summary text,
    metadata jsonb default '{}',
    created_at timestamptz not null default now()
);

create index idx_sections_source on source_sections(source_id, level, ordinal);

-- =============================================================================
-- SOURCE CHUNKS
-- =============================================================================

create table if not exists source_chunks (
    chunk_id uuid primary key default gen_random_uuid(),
    source_id uuid not null references source_documents(source_id) on delete cascade,
    section_id uuid references source_sections(section_id),
    chunk_index int not null default 0,
    content text not null,
    token_count int,
    page_start int,
    page_end int,
    char_start int,
    char_end int,
    metadata jsonb default '{}',
    embedding vector(1536),
    created_at timestamptz not null default now()
);

create index idx_chunks_source_section on source_chunks(source_id, section_id, chunk_index);
create index idx_chunks_embedding_hnsw on source_chunks using hnsw (embedding vector_cosine_ops);

-- =============================================================================
-- SOURCE SUMMARIES
-- =============================================================================

create table if not exists source_summaries (
    summary_id uuid primary key default gen_random_uuid(),
    source_id uuid not null references source_documents(source_id) on delete cascade,
    section_id uuid references source_sections(section_id),
    summary_type text not null check (summary_type in ('document', 'section', 'executive', 'entities', 'topic_map')),
    content text not null,
    metadata jsonb default '{}',
    embedding vector(1536),
    created_at timestamptz not null default now()
);

create index idx_summaries_source_section on source_summaries(source_id, section_id);
create index idx_summaries_embedding_hnsw on source_summaries using hnsw (embedding vector_cosine_ops);
create index idx_summaries_type on source_summaries(summary_type);

-- =============================================================================
-- SOURCE ENTITIES
-- =============================================================================

create table if not exists source_entities (
    entity_id uuid primary key default gen_random_uuid(),
    source_id uuid not null references source_documents(source_id) on delete cascade,
    section_id uuid references source_sections(section_id),
    entity_type text not null,
    canonical_name text not null,
    aliases jsonb default '[]',
    metadata jsonb default '{}',
    created_at timestamptz not null default now()
);

create index idx_entities_source on source_entities(source_id);
create index idx_entities_type on source_entities(entity_type);
create index idx_entities_name on source_entities(canonical_name);

-- =============================================================================
-- SOURCE CITATIONS
-- =============================================================================

create table if not exists source_citations (
    citation_id uuid primary key default gen_random_uuid(),
    chunk_id uuid not null references source_chunks(chunk_id) on delete cascade,
    source_id uuid not null references source_documents(source_id) on delete cascade,
    page int,
    locator text,
    quote_text text,
    metadata jsonb default '{}',
    created_at timestamptz not null default now()
);

create index idx_citations_chunk on source_citations(chunk_id);
create index idx_citations_source on source_citations(source_id);

-- =============================================================================
-- SOURCE INGESTION QUEUE (para tracking de workers asíncronos)
-- =============================================================================

create table if not exists source_ingestion_queue (
    queue_id uuid primary key default gen_random_uuid(),
    source_id uuid not null references source_documents(source_id) on delete cascade,
    worker_type text not null default 'ingestion',
    status text not null default 'pending' check (status in ('pending', 'processing', 'completed', 'failed')),
    priority int not null default 0,
    payload jsonb default '{}',
    error_message text,
    started_at timestamptz,
    completed_at timestamptz,
    created_at timestamptz not null default now()
);

create index idx_ingestion_queue_status on source_ingestion_queue(status, priority);
create index idx_ingestion_queue_source on source_ingestion_queue(source_id);

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

create or replace function update_workspace_timestamp()
returns trigger as $$
begin
    NEW.updated_at = now();
    return NEW;
end;
$$ language plpgsql;

create or replace function update_document_timestamp()
returns trigger as $$
begin
    NEW.updated_at = now();
    return NEW;
end;
$$ language plpgsql;

-- Triggers for automatic timestamps
drop trigger if exists workspaces_updated on source_workspaces;
create trigger workspaces_updated before update on source_workspaces
    for each row execute function update_workspace_timestamp();

drop trigger if exists documents_updated on source_documents;
create trigger documents_updated before update on source_documents
    for each row execute function update_document_timestamp();
