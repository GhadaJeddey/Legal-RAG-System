"""
Embed each chunk in data/output/chunks-<model>.json and store it in Postgres/pgvector.

Usage:
    python store_chunks.py
"""

import json
import os
from pathlib import Path

import psycopg  # driver postgres qui permet la cnx de python à postgres
from dotenv import load_dotenv
from pgvector.psycopg import register_vector # convertit un vecteur numpy en type 'vector' de Postegres ( et inversement)

from config.embedding_models import get_current_model

SCHEMA_PATH = Path(__file__).with_name("storage") / "schema.sql"


def main():
    load_dotenv()
    database_url = os.environ["DATABASE_URL"]

    model_config = get_current_model()
    if model_config["provider"] != "sentence_transformers":
        raise NotImplementedError(
            f"Provider '{model_config['provider']}' not wired into store_chunks.py yet "
            f"(only 'sentence_transformers' is supported here)."
        )

    safe_model_name = model_config["name"].replace("/", "_")
    chunks_path = Path(f"data/output/chunks-{safe_model_name}.json")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_config["name"])

    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        register_vector(conn)

        with conn.cursor() as cur:
            for chunk in chunks:
                meta = chunk["metadata"]
                breadcrumb = {k: v for k, v in meta.items() if k.startswith("level_")}
                embedding = model.encode(chunk["text"], normalize_embeddings=True)
                cur.execute(
                    """
                    INSERT INTO chunks
                        (text_content, article_number, article_title, breadcrumb,
                         sub_chunk, sub_chunk_total, token_count, model_name, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (breadcrumb, article_number, (COALESCE(sub_chunk, -1)), model_name)
                    DO UPDATE SET
                        text_content = EXCLUDED.text_content,
                        article_title = EXCLUDED.article_title,
                        breadcrumb = EXCLUDED.breadcrumb,
                        sub_chunk_total = EXCLUDED.sub_chunk_total,
                        token_count = EXCLUDED.token_count,
                        embedding = EXCLUDED.embedding,
                        created_at = now()
                    """,
                    (
                        chunk["text"],
                        meta["article_number"],
                        meta["article_title"],
                        json.dumps(breadcrumb, ensure_ascii=False),
                        meta["sub_chunk"],
                        meta["sub_chunk_total"],
                        meta["token_count"],
                        model_config["name"],
                        embedding,
                    ),
                )

    print(f"Inserted {len(chunks)} rows into 'chunks' (model={model_config['name']})")


if __name__ == "__main__":
    main()
