"""Orchestrates retrieval + generation for a single query."""

import asyncpg
from generator import generate
from reranker import rerank
from retriever import retrieve

# Cross-encoder rerank_score below which the top result is treated as "not
# actually relevant" -- calibrated from eval/results/retrieval_eval_20260814T122645Z.json:
# on that run, off-topic questions scored at most 0.0023 and on-topic questions
# scored at least 0.0066 (bi+cross condition), so 0.005 sits in that gap.
# Caveat: calibrated on only 10 off-topic questions -- revisit if production
# traffic shows false rejects (relevant question blocked) or false positives
# (off-topic question slips through).
RERANK_MIN_SCORE = 0.005


async def answer_query(
    pool: asyncpg.Pool,
    query: str,
    groq_api_key: str,
    top_k: int = 5,
    use_reranker: bool = False,
    candidate_k: int = 20,
) -> dict:
    retrieve_k = candidate_k if use_reranker else top_k
    chunks = await retrieve(pool, query, top_k=retrieve_k)

    if use_reranker and chunks:
        chunks = rerank(query, chunks, top_k=top_k)
        if chunks[0]["rerank_score"] < RERANK_MIN_SCORE:
            chunks = []

    if not chunks:
        return {
            "answer": "Aucun extrait pertinent trouve dans le PCG pour repondre a cette question.",
            "sources": [],
        }

    answer = await generate(query, chunks, api_key=groq_api_key)

    sources = [
        {
            "rank_bi": c["rank_bi"],
            "rank_cross": c.get("rank_cross"),
            "article_number": c["article_number"],
            "article_title": c["article_title"],
            "similarity": c["similarity"],
        }
        for c in chunks
    ]
    
    return {"answer": answer, "sources": sources}
