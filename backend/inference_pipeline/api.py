"""FastAPI entrypoint for the inference pipeline.

Usage:
    uvicorn api:app --reload
"""
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

sys.path.append(str(Path(__file__).resolve().parent.parent / "ingestion_pipeline"))

from config.embedding_models import get_current_model as get_current_embedding_model
from config.llm_models import get_current_model as get_current_llm_model
from generator import GROQ_BASE_URL

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

# Local dev only: the Vite frontend runs on a different origin (localhost:5173).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    conversation_id: str | None = None  # reserved for future multi-turn support, unused for now


class Source(BaseModel):
    rank_bi: int
    rank_cross: int | None
    article_number: str | None
    article_title: str | None
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


def _check_embedding_model() -> dict:
    model_config = get_current_embedding_model()
    if model_config["provider"] != "sentence_transformers":
        return {
            "ok": False,
            "model": model_config["name"],
            "provider": model_config["provider"],
            "error": f"Unsupported provider: {model_config['provider']}",
        }

    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(model_config["name"])
        return {
            "ok": True,
            "model": model_config["name"],
            "provider": model_config["provider"],
        }
    except Exception as exc:
        return {
            "ok": False,
            "model": model_config["name"],
            "provider": model_config["provider"],
            "error": str(exc),
        }


def _check_llm_model() -> dict:
    model_config = get_current_llm_model()
    api_key = os.environ.get("GROQ_API_KEY")

    if model_config["provider"] != "groq":
        return {
            "ok": False,
            "model": model_config["name"],
            "provider": model_config["provider"],
            "error": f"Unsupported provider: {model_config['provider']}",
        }

    if not api_key:
        return {
            "ok": False,
            "model": model_config["name"],
            "provider": model_config["provider"],
            "error": "GROQ_API_KEY is missing",
        }

    try:
        AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        return {
            "ok": True,
            "model": model_config["name"],
            "provider": model_config["provider"],
        }
    except Exception as exc:
        return {
            "ok": False,
            "model": model_config["name"],
            "provider": model_config["provider"],
            "error": str(exc),
        }


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
    embedding = _check_embedding_model()
    llm = _check_llm_model()
    healthy = embedding["ok"] and llm["ok"]

    return {
        "status": "ok" if healthy else "degraded",
        "embedding_model": embedding,
        "llm": llm,
    }
