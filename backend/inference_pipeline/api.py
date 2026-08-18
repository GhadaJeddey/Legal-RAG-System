"""FastAPI entrypoint for the inference pipeline.

Usage:
    uvicorn api:app --reload
"""
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import asyncpg
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

sys.path.append(str(Path(__file__).resolve().parent.parent / "ingestion_pipeline"))

from config.embedding_models import get_current_model as get_current_embedding_model
from config.llm_models import get_current_model as get_current_llm_model
from generator import GROQ_BASE_URL

from rag_pipeline import answer_query
from full_pipeline import run_pipeline, _set_document_status

load_dotenv()

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "ingestion_pipeline" / "data" / "raw"


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


class DocumentOut(BaseModel):
    id: int
    filename: str
    status: str
    page_count: int | None
    uploaded_at: datetime


class ChunkOut(BaseModel):
    article_number: str | None
    article_title: str | None
    sub_chunk: int | None
    sub_chunk_total: int | None
    token_count: int | None
    text_content: str


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


def _run_ingestion(document_id: int, file_path: str) -> None:
    """Background job: runs on a worker thread (see BackgroundTasks), uses the
    sync psycopg connection from full_pipeline.py rather than the app's asyncpg pool."""
    _set_document_status(document_id, "processing")
    try:
        run_pipeline(file_path, document_id)
    except Exception as exc:
        _set_document_status(document_id, "failed")
        print(f"Ingestion failed for document {document_id}: {exc}")
    else:
        _set_document_status(document_id, "done")


@app.post("/documents")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    existing = await app.state.pool.fetchrow(
        "SELECT id FROM documents WHERE filename = $1 AND status = 'done'", file.filename
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"A document named '{file.filename}' has already been ingested",
        )

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DATA_DIR / file.filename
    file_path.write_bytes(await file.read())

    row = await app.state.pool.fetchrow(
        """
        INSERT INTO documents (filename, original_path, status)
        VALUES ($1, $2, 'pending')
        RETURNING id, status
        """,
        file.filename,
        str(file_path),
    )

    background_tasks.add_task(_run_ingestion, row["id"], str(file_path))

    return {"id": row["id"], "status": row["status"]}


@app.get("/documents", response_model=list[DocumentOut])
async def list_documents():
    rows = await app.state.pool.fetch(
        "SELECT id, filename, status, page_count, uploaded_at FROM documents ORDER BY uploaded_at DESC"
    )
    return [dict(row) for row in rows]


@app.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: int):
    row = await app.state.pool.fetchrow(
        "SELECT original_path FROM documents WHERE id = $1", document_id
    )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    await app.state.pool.execute("DELETE FROM chunks WHERE document_id = $1", document_id)
    await app.state.pool.execute("DELETE FROM documents WHERE id = $1", document_id)

    file_path = Path(row["original_path"])
    if not file_path.is_absolute():
        file_path = Path(__file__).resolve().parent.parent / "ingestion_pipeline" / file_path
    file_path.unlink(missing_ok=True)


@app.get("/documents/{document_id}/chunks", response_model=list[ChunkOut])
async def get_document_chunks(document_id: int):
    rows = await app.state.pool.fetch(
        """
        SELECT article_number, article_title, sub_chunk, sub_chunk_total, token_count, text_content
        FROM chunks
        WHERE document_id = $1
        ORDER BY article_number, sub_chunk
        """,
        document_id,
    )
    return [dict(row) for row in rows]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    result = await answer_query(
        pool=app.state.pool,
        query=request.query,
        groq_api_key=os.environ["GROQ_API_KEY"],
        top_k=request.top_k,
        use_reranker=True,
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
