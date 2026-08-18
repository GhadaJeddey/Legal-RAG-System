"""
End-to-end ingestion pipeline.

Input  : data/raw/<nom_du_pdf>.pdf
Output : data/output/markdown_pages.json (une entree par page)
         data/output/document.md (concatenation, pour relecture rapide)
         data/output/chunks-<model>.json (chunks semantiques)
         insertion des chunks dans Postgres/pgvector

Usage:
    python full_pipeline.py <chemin_vers_pdf>
    python full_pipeline.py <chemin_vers_pdf> --start-page 10 --end-page 50
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

from chunkers import ArticleChunker
from config.embedding_models import get_current_model
from parsers.markdown_parser import MarkdownParser

PARSER = MarkdownParser()
# Absolute, not cwd-relative: run_pipeline() can now be called from the API process
# (cwd = backend/inference_pipeline), not just the CLI (cwd = backend/ingestion_pipeline).
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "output"
SCHEMA_PATH = Path(__file__).with_name("storage") / "schema.sql"


def _validate_args(pdf_path: str, start_page: int | None, end_page: int | None) -> None:
    """valider les arguments saisis via cmd"""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"Fichier introuvable: {pdf_path}")

    if start_page is not None and start_page < 1:
        raise ValueError("--start-page doit etre >= 1")

    if start_page is not None and end_page is not None and start_page > end_page:
        raise ValueError("--start-page ne peut pas etre superieur a --end-page")


def _write_extracted_pages(pages: list[dict]) -> Path:
    """save the parsed pages (.json -- for debugging and reusability && .md) locally under data/output """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "markdown_pages.json"
    with open(json_path, "w", encoding="utf-8") as file_handle:
        json.dump(pages, file_handle, ensure_ascii=False, indent=2)
    print(f"Sauvegarde: {json_path}")

    md_path = OUTPUT_DIR / "document.md"
    with open(md_path, "w", encoding="utf-8") as file_handle:
        for page in pages:
            file_handle.write(f"\n\n<!-- page {page['page']} -->\n\n")
            file_handle.write(page["markdown"])
    print(f"Sauvegarde: {md_path}")

    return md_path


def _load_chunker() -> tuple[ArticleChunker, dict]:
    model_config = get_current_model()

    print(f"Loading {model_config['name']} ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_config["name"])
    return ArticleChunker(embedding_model=model), model_config


def _chunk_markdown(document_path: Path) -> tuple[list[dict], ArticleChunker, dict]:
    """run the chunker on the markdown file"""
    chunker, model_config = _load_chunker()
    markdown = document_path.read_text(encoding="utf-8")
    chunks = chunker.chunk(markdown)

    n_articles = len({
        (chunk["metadata"]["article_number"], chunk["metadata"]["article_title"])
        for chunk in chunks
    })
    
    n_sub_split = len({
        (chunk["metadata"]["article_number"], chunk["metadata"]["article_title"])
        for chunk in chunks
        if chunk["metadata"]["sub_chunk"] is not None
    })
    
    token_counts = [chunk["metadata"]["token_count"] for chunk in chunks]

    below_floor = [
        chunk for chunk in chunks
        if chunk["metadata"]["sub_chunk"] is not None
        and chunk["metadata"]["token_count"] < chunker.min_fragment_tokens
    ]

    print(f"Articles found: {n_articles}")
    print(f"Articles sub-split: {n_sub_split}")
    print(f"Total chunks produced: {len(chunks)}")
    
    if token_counts:
        print(
            f"Chunk token count -> min {min(token_counts)}, "
            f"median {statistics.median(token_counts)}, "
            f"mean {round(statistics.mean(token_counts))}, "
            f"max {max(token_counts)}"
        )
        
    print(
        f"Sub-chunks left below min_fragment_tokens ({chunker.min_fragment_tokens}) "
        f"because no neighbor fit under max_tokens: {len(below_floor)} "
        f"({round(100 * len(below_floor) / len(chunks), 1) if chunks else 0.0}% of all chunks)"
    )
    for chunk in below_floor:
        metadata = chunk["metadata"]
        print(
            f"  - {metadata['article_number']} / {metadata['article_title']} "
            f"(sub_chunk {metadata['sub_chunk'] + 1}/{metadata['sub_chunk_total']}, "
            f"{metadata['token_count']} tokens)"
        )

    safe_model_name = model_config["name"].replace("/", "_")
    chunks_path = OUTPUT_DIR / f"chunks-{safe_model_name}.json"
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {chunks_path}")

    return chunks, chunker, model_config


def _store_chunks(chunks: list[dict], model_config: dict, document_id: int) -> None:
    load_dotenv()
    database_url = os.environ["DATABASE_URL"]

    safe_model_name = model_config["name"].replace("/", "_")
    chunks_path = OUTPUT_DIR / f"chunks-{safe_model_name}.json"
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_config["name"])

    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        register_vector(connection)

        with connection.cursor() as cursor:
            for chunk in chunks:
                metadata = chunk["metadata"]
                breadcrumb = {key: value for key, value in metadata.items() if key.startswith("level_")}
                embedding = model.encode(chunk["text"], normalize_embeddings=True)
                cursor.execute(
                    """
                    INSERT INTO chunks
                        (document_id, text_content, article_number, article_title, breadcrumb,
                         sub_chunk, sub_chunk_total, token_count, model_name, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, breadcrumb, article_number, (COALESCE(sub_chunk, -1)), model_name)
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
                        document_id,
                        chunk["text"],
                        metadata["article_number"],
                        metadata["article_title"],
                        json.dumps(breadcrumb, ensure_ascii=False),
                        metadata["sub_chunk"],
                        metadata["sub_chunk_total"],
                        metadata["token_count"],
                        model_config["name"],
                        embedding,
                    ),
                )

    print(f"Inserted {len(chunks)} rows into 'chunks' (model={model_config['name']}, document_id={document_id})")


def _create_document(pdf_path: str) -> int:
    """Insert a 'documents' row for a standalone CLI run and return its id."""
    load_dotenv()
    database_url = os.environ["DATABASE_URL"]

    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (filename, original_path, status)
                VALUES (%s, %s, 'processing')
                RETURNING id
                """,
                (Path(pdf_path).name, pdf_path),
            )
            return cursor.fetchone()[0]


def _set_document_status(document_id: int, status: str) -> None:
    load_dotenv()
    database_url = os.environ["DATABASE_URL"]

    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute(
            "UPDATE documents SET status = %s WHERE id = %s", (status, document_id)
        )


def run_pipeline(
    pdf_path: str,
    document_id: int,
    start_page: int | None = None,
    end_page: int | None = None,
) -> None:
    """Parse -> chunk -> store a PDF for a given document_id. Raises on failure
    (callers running this as a background task are responsible for catching)."""
    _validate_args(pdf_path, start_page, end_page)

    pages = PARSER.extract(pdf_path, start_page=start_page, end_page=end_page)
    print(f"{len(pages)} pages extraites.")

    if not pages:
        raise ValueError("Aucune page extraite.")

    document_path = _write_extracted_pages(pages)
    chunks, _, model_config = _chunk_markdown(document_path)
    _store_chunks(chunks, model_config, document_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="Page de depart (1-indexee). Defaut: debut du document.",
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Page de fin (incluse). Defaut: fin du document.",
    )
    args = parser.parse_args()

    document_id = _create_document(args.pdf_path)
    try:
        run_pipeline(args.pdf_path, document_id, start_page=args.start_page, end_page=args.end_page)
    except Exception as exc:
        _set_document_status(document_id, "failed")
        print(f"Erreur: {exc}")
        sys.exit(1)
    else:
        _set_document_status(document_id, "done")


if __name__ == "__main__":
    main()
