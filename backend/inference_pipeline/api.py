"""FastAPI entrypoint for the inference pipeline.

Usage:
    uvicorn api:app --reload
"""
import json
import os
from contextlib import asynccontextmanager

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline import answer_query

load_dotenv()


async def _init_connection(conn: asyncpg.Connection):
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await asyncpg.create_pool(
        os.environ["DATABASE_URL"], init=_init_connection
    )
    yield
    await app.state.pool.close()


app = FastAPI(title="PCG RAG Inference API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    conversation_id: str | None = None  # reserved for future multi-turn support, unused for now


class Source(BaseModel):
    article_number: str | None
    article_title: str | None
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    result = await answer_query(
        pool=app.state.pool,
        query=request.query,
        groq_api_key=os.environ["GROQ_API_KEY"],
        top_k=request.top_k,
    )
    return QueryResponse(**result)

@app.get("/health")
async def health():
    return {"status": "ok"}
