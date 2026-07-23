"""Orchestrates retrieval + generation for a single query."""

import asyncpg
from generator import generate
from retriever import retrieve


async def answer_query(
    pool: asyncpg.Pool,
    query: str,
    groq_api_key: str,
    top_k: int = 5,
) -> dict:
    chunks = await retrieve(pool, query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Aucun extrait pertinent trouve dans le PCG pour repondre a cette question.",
            "sources": [],
        }

    answer = await generate(query, chunks, api_key=groq_api_key)

    sources = [
        {
            "article_number": c["article_number"],
            "article_title": c["article_title"],
            "similarity": c["similarity"],
        }
        for c in chunks
    ]
    
    return {"answer": answer, "sources": sources}
