CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    article_number TEXT,
    article_title TEXT,
    breadcrumb JSONB,
    sub_chunk INT,
    sub_chunk_total INT,
    token_count INT,
    model_name TEXT NOT NULL,
    embedding VECTOR(1024),
    created_at TIMESTAMPTZ DEFAULT now()
);
