"""
The DB pool is created once at app startup and passed in here, rather than opened per call.

pgvector's <=> operator computes cosine distance between the query embedding and the stored chunk embeddings, and we order by that distance to get the closest chunks.

Retrieval via a bi-encoder ( bge ) using cosine similarity for scoring . 
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


# We'll apply this later after the reranker to filter out veery irrelevant chunks that should be
# answered with "Aucun extrait pertinent trouvé"
MIN_SIMILARITY = 0.5


async def retrieve(pool: asyncpg.Pool, query: str, top_k: int = 5) -> list[dict]:
    """Return the top_k chunks most similar to `query`, ordered by cosine distance."""
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
            "rank_bi": rank,
            "text": row["text_content"],
            "article_number": row["article_number"],
            "article_title": row["article_title"],
            "breadcrumb": row["breadcrumb"],
            "sub_chunk": row["sub_chunk"],
            "sub_chunk_total": row["sub_chunk_total"],
            "similarity": float(row["similarity"]),
        }
        for rank, row in enumerate(rows, start=1)
    ]
