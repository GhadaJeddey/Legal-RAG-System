CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    text_content TEXT NOT NULL,
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

-- Dedup key for re-running store_chunks.py: one row per breadcrumb/article/sub-chunk/model.
-- breadcrumb is included because article_number is only unique within its own Livre/section
-- (two unrelated articles can both be numbered "322-4" in different Livres).
DROP INDEX IF EXISTS chunks_unique_idx;

CREATE UNIQUE INDEX chunks_unique_idx
    ON chunks (breadcrumb, article_number, COALESCE(sub_chunk, -1), model_name);

