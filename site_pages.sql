-- Part 1: Enable extension and create table
-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create a table for storing RPGJS documentation chunks
create table if not exists rpgjs_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Ensure we don't have duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index on metadata for faster filtering
create index idx_rpgjs_pages_metadata on rpgjs_pages using gin (metadata);

-- Part 2: Create the matching function
create or replace function match_rpgjs_pages (
    query_embedding vector(1536),
    match_count int DEFAULT 5,
    filter jsonb DEFAULT '{}'
) returns table (
    id bigint,
    url varchar,
    title varchar,
    content text,
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        rpgjs_pages.id,
        rpgjs_pages.url,
        rpgjs_pages.title,
        rpgjs_pages.content,
        1 - (rpgjs_pages.embedding <=> query_embedding) as similarity
    from rpgjs_pages
    where metadata @> filter
    order by rpgjs_pages.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- Part 3: Add vector index (run this after inserting data)
-- NOTE: Run this separately after the table is created and data is inserted
/*
create index on rpgjs_pages 
using ivfflat (embedding vector_cosine_ops)
with (lists = 4);
*/

-- Everything above will work for any PostgreSQL database. The below commands are for Supabase security

-- Enable RLS on the table
alter table rpgjs_pages enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on rpgjs_pages
  for select
  to public
  using (true);