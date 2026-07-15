"""
Build a JSON file of candidate articles for LLM-based segmentation
annotation: only articles long enough that any reasonable embedding model
would sub-split them, each with the `units` list _split_into_units() would
produce for it (the same units an annotator needs to pick ideal_breakpoints
over — see eval/segmentation_metrics.py).

Deliberately model-independent: this set must be reusable across every
candidate embedding model, so article selection uses a plain word count
instead of any one model's tokenizer (different tokenizers would flag
different articles as "too long", biasing the set toward whichever model
built it).

Usage:
    python -m eval.build_candidate_articles
"""
import json
from pathlib import Path

from chunkers.article_chunker import _parse_articles, _split_into_units

# Rough word-count proxy for DEFAULT_MAX_TOKENS (350 tokens): French legal
# text runs roughly 1.3-1.5 tokens/word, so ~250 words is a generous, model
# agnostic stand-in for "long enough to need sub-splitting".
MIN_WORDS = 250


def main():
    doc_path = Path("data/output/document.md")
    markdown = doc_path.read_text(encoding="utf-8")

    articles = _parse_articles(markdown)
    candidates = []
    for a in articles:
        body = a["body"]
        if not body:
            continue
        word_count = len(body.split())
        if word_count <= MIN_WORDS:
            continue
        candidates.append({
            "article_number": a["article_number"],
            "article_title": a["title"],
            "word_count": word_count,
            "units": _split_into_units(body),
        })

    out_path = Path("data/output/candidate_articles_for_annotation.json")
    out_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Articles long enough to need sub-split: {len(candidates)} / {len(articles)} total")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
