"""
Dense retrieval: embed a query and fetch the closest chunks from Postgres/pgvector .


The DB pool is created once at app startup and passed in here, rather than opened per call.
"""

import sys
from pathlib import Path
import asyncpg

sys.path.append(str(Path(__file__).resolve().parent.parent / "ingestion_pipeline"))
from config.embedding_models import get_current_model  # noqa: E402

_model = None

def get_query_embedder():
    """
    Lazily load the sentence-transformers model used to embed queries.
    """
    global _model
    if _model is None:
        model_config = get_current_model()
        if model_config["provider"] != "sentence_transformers":
            raise NotImplementedError(
                f"Provider '{model_config['provider']}' not wired into retriever.py yet "
                f"(only 'sentence_transformers' is supported here)."
            )
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_config["name"])
    return _model


MIN_SIMILARITY = 0.5  # top-k neighbors below this are treated as "not actually relevant"


async def retrieve(
    pool: asyncpg.Pool, query: str, top_k: int = 5, min_similarity: float = MIN_SIMILARITY
) -> list[dict]:
    """Return the top_k chunks most similar to `query`, ordered by cosine distance.

    Top-k always returns k rows regardless of how distant they are, so results
    below min_similarity are dropped afterwards -- otherwise an unrelated query
    would still get handed the "closest" chunks even though none are relevant.
    """
    model = get_query_embedder()
    embedding = model.encode(query, normalize_embeddings=True).tolist()

    rows = await pool.fetch(
        """
        SELECT text_content, article_number, article_title, breadcrumb,
               sub_chunk, sub_chunk_total,
               1 - (embedding <=> $1) AS similarity
        FROM chunks
        ORDER BY embedding <=> $1
        LIMIT $2
        """,
        str(embedding),
        top_k,
    )
    
    # <=> is the pgvector's cosine similarity operator

    return [
        {
            "text": row["text_content"],
            "article_number": row["article_number"],
            "article_title": row["article_title"],
            "breadcrumb": row["breadcrumb"],
            "sub_chunk": row["sub_chunk"],
            "sub_chunk_total": row["sub_chunk_total"],
            "similarity": float(row["similarity"]),
        }
        for row in rows
        if row["similarity"] >= min_similarity
    ]
