-- Part 1: Enable extensions
create extension if not exists vector;
create extension if not exists "uuid-ossp";

-- Part 2: Create tables if they don't exist
create table if not exists rpgjs_documentation (
    id uuid primary key default uuid_generate_v4(),
    url text not null,
    chunk_number integer not null,
    title text not null,
    summary text not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536),
    version text not null,
    source text not null default 'rpgjs_docs',
    last_updated timestamp with time zone not null default now(),
    heading text,
    subheading text,
    code_blocks jsonb default '[]'::jsonb,
    unique(url, chunk_number)
);

create table if not exists rpgjs_conversations (
    id uuid primary key default uuid_generate_v4(),
    conversation_id text not null,
    user_id text not null,
    query text not null,
    response text not null,
    source_documents jsonb,
    metadata jsonb,
    created_at timestamp with time zone default now(),
    unique(conversation_id, user_id)
);

-- Part 3: Create matching function
create or replace function match_rpgjs_documentation(
    query_embedding vector(1536),
    match_threshold float default 0.7,
    match_count int default 5,
    min_content_length int default 50,
    filter_params jsonb default '{}'::jsonb
)
returns table (
    id uuid,
    url text,
    chunk_number integer,
    title text,
    content text,
    metadata jsonb,
    version text,
    heading text,
    subheading text,
    code_blocks jsonb,
    similarity float
)
language plpgsql
as $$
#variable_conflict use_column
declare
    version_filter text = (filter_params->>'version')::text;
    source_filter text = (filter_params->>'source')::text;
begin
    return query
    select
        rd.id,
        rd.url,
        rd.chunk_number,
        rd.title,
        rd.content,
        rd.metadata,
        rd.version,
        rd.heading,
        rd.subheading,
        rd.code_blocks,
        1 - (rd.embedding <=> query_embedding) as similarity
    from rpgjs_documentation rd
    where 1 - (rd.embedding <=> query_embedding) > match_threshold
        and length(rd.content) >= min_content_length
        and (version_filter is null or rd.version = version_filter)
        and (source_filter is null or rd.source = source_filter)
    order by rd.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- Part 4: Add indexes for better performance
DO $$
BEGIN
    -- Save the current maintenance_work_mem
    SET maintenance_work_mem TO '64MB';
    
    -- Create the ivfflat index
    CREATE INDEX IF NOT EXISTS rpgjs_documentation_embedding_idx 
    ON rpgjs_documentation 
    USING ivfflat (embedding vector_cosine_ops) 
    WITH (lists = 100);
END $$;

-- Part 5: Enable RLS
alter table rpgjs_documentation enable row level security;
alter table rpgjs_conversations enable row level security;

-- Part 6: Create RLS policies
create policy "Enable read access for all users" on rpgjs_documentation
    for select
    to public
    using (true);

create policy "Enable read access for all users" on rpgjs_conversations
    for select
    to public
    using (true);

create policy "Enable insert access for all users" on rpgjs_documentation
    for insert
    to public
    with check (true);

create policy "Enable insert access for all users" on rpgjs_conversations
    for insert
    to public
    with check (true);