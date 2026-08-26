CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    original_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | done | failed
    page_count INT,
    uploaded_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id),
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

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS document_id INT REFERENCES documents(id);

-- One-off backfill: chunks inserted before document_id existed all belong to the
-- original PCG PDF, ingested once via the CLI before uploads were tracked. No-op
-- on repeat runs since both the INSERT and the UPDATE only touch missing rows.
INSERT INTO documents (filename, original_path, status)
SELECT 'PCG--1er-janvier-2025.pdf', 'data/raw/PCG--1er-janvier-2025.pdf', 'done'
WHERE NOT EXISTS (SELECT 1 FROM documents WHERE filename = 'PCG--1er-janvier-2025.pdf')
  AND EXISTS (SELECT 1 FROM chunks WHERE document_id IS NULL);

UPDATE chunks SET document_id = (SELECT id FROM documents WHERE filename = 'PCG--1er-janvier-2025.pdf')
WHERE document_id IS NULL;

-- Dedup key for re-running store_chunks.py: one row per document/breadcrumb/article/sub-chunk/model.
-- breadcrumb is included because article_number is only unique within its own Livre/section
-- (two unrelated articles can both be numbered "322-4" in different Livres). document_id scopes
-- the dedup per document since two different uploaded documents (e.g. two PCG versions) could
-- legitimately share an article_number/breadcrumb.
DROP INDEX IF EXISTS chunks_unique_idx;

CREATE UNIQUE INDEX chunks_unique_idx
    ON chunks (document_id, breadcrumb, article_number, COALESCE(sub_chunk, -1), model_name);

-- Lexical (keyword) search support: generated tsvector kept in sync automatically,
-- indexed with GIN for fast full-text lookups via the /query "keyword" mode.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('french', text_content)) STORED;

CREATE INDEX IF NOT EXISTS chunks_search_vector_idx ON chunks USING GIN (search_vector);

