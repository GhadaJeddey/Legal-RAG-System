"""
Quick runner: chunk data/output/document.md with ArticleChunker and report stats.

Usage:
    python run_chunking.py
"""
import json
import statistics
from pathlib import Path

from chunkers import ArticleChunker
from config.embedding_models import get_current_model


def main():
    doc_path = Path("data/output/document.md")
    markdown = doc_path.read_text(encoding="utf-8")

    model_config = get_current_model()
    if model_config["provider"] != "sentence_transformers":
        raise NotImplementedError(
            f"Provider '{model_config['provider']}' not wired into run_chunking.py yet "
            f"(only 'sentence_transformers' is supported here)."
        )

    print(f"Loading {model_config['name']} ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_config["name"])

    chunker = ArticleChunker(embedding_model=model)
    chunks = chunker.chunk(markdown)

    n_articles = len({(c["metadata"]["article_number"], c["metadata"]["article_title"]) for c in chunks})
    n_sub_split = len({
        (c["metadata"]["article_number"], c["metadata"]["article_title"])
        for c in chunks if c["metadata"]["sub_chunk"] is not None
    })
    token_counts = [c["metadata"]["token_count"] for c in chunks]

    below_floor = [
        c for c in chunks
        if c["metadata"]["sub_chunk"] is not None
        and c["metadata"]["token_count"] < chunker.min_fragment_tokens
    ]

    print(f"Articles found: {n_articles}")
    print(f"Articles sub-split: {n_sub_split}")
    print(f"Total chunks produced: {len(chunks)}")
    print(f"Chunk token count -> min {min(token_counts)}, "
          f"median {statistics.median(token_counts)}, "
          f"mean {round(statistics.mean(token_counts))}, "
          f"max {max(token_counts)}")
    print(f"Sub-chunks left below min_fragment_tokens ({chunker.min_fragment_tokens}) "
          f"because no neighbor fit under max_tokens: {len(below_floor)} "
          f"({round(100 * len(below_floor) / len(chunks), 1)}% of all chunks)")
    for c in below_floor:
        m = c["metadata"]
        print(f"  - {m['article_number']} / {m['article_title']} "
              f"(sub_chunk {m['sub_chunk'] + 1}/{m['sub_chunk_total']}, {m['token_count']} tokens)")

    safe_model_name = model_config["name"].replace("/", "_")
    out_path = Path(f"data/output/chunks-{safe_model_name}.json")
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

