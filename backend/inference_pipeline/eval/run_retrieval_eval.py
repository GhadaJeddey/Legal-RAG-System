"""Retrieval-only evaluation: compares two retrieval strategies against the
eval set using rank-sensitive metrics (Recall@1, MRR, Recall@5, nDCG@5).

nDCG@5 uses each case's optional `primary_article` (the article judged most
central to the question) as a graded-relevance signal; when absent, all
expected articles are weighted equally.

Bypasses generation entirely -- this measures retrieval quality directly,
not the LLM's answer.

Conditions:
    A - bi-encoder only    : retrieve(top_k=5)
    C - bi + cross-encoder : retrieve(top_k=20) -> rerank(top_k=5)  (shipped design)

q13/q14-like off-topic questions (expected_articles=[]) are excluded from
Recall@1/MRR/Recall@5 -- those metrics are undefined without an expected
article -- and reported separately via their top-1 score vs. the average
top-1 score of the on-topic questions, as a signal for whether scores
naturally separate on/off-topic questions (relevant for a future post-rerank
relevance floor, see FUTURE_IMPROVEMENTS.md -- not implemented here).

Usage:
    python run_retrieval_eval.py
"""
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from retriever import retrieve  # noqa: E402
from reranker import rerank  # noqa: E402

from metrics import mrr, ndcg_at_k, recall_at_1, recall_at_k

EVAL_SET_PATH = Path(__file__).with_name("retrieval_eval_set.json")
RESULTS_DIR = Path(__file__).with_name("results")
CONDITIONS = ["A_bi_only", "C_bi_plus_cross"]


async def _init_connection(conn: asyncpg.Connection):
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


def _dedupe_articles(chunks: list[dict]) -> list[str]:
    """Ordered, deduplicated article numbers -- several sub-chunks can share
    the same article_number, we only want each article counted once."""
    seen = set()
    ordered = []
    for chunk in chunks:
        article = chunk["article_number"]
        if article and article not in seen:
            seen.add(article)
            ordered.append(article)
    return ordered


async def _run_conditions(pool: asyncpg.Pool, question: str) -> dict:
    """Run the two retrieval conditions for one question."""
    t0 = time.monotonic()
    dense_top5 = await retrieve(pool, question, top_k=5)
    print(f"    A_bi_only done ({time.monotonic() - t0:.1f}s)", flush=True)

    t0 = time.monotonic()
    dense_top20 = await retrieve(pool, question, top_k=20)
    hybrid = rerank(question, dense_top20, top_k=5) if dense_top20 else []
    print(f"    C_bi_plus_cross done ({time.monotonic() - t0:.1f}s)", flush=True)

    return {
        "A_bi_only": {
            "articles": _dedupe_articles(dense_top5),
            "top1_score": dense_top5[0]["similarity"] if dense_top5 else 0.0,
        },
        "C_bi_plus_cross": {
            "articles": _dedupe_articles(hybrid),
            "top1_score": hybrid[0]["rerank_score"] if hybrid else 0.0,
        },
    }


async def run_eval() -> list[dict]:
    load_dotenv()
    cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))

    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], init=_init_connection)

    per_question = []
    try:
        for i, case in enumerate(cases, start=1):
            print(f"[{i}/{len(cases)}] {case['id']}: {case['question'][:70]}", flush=True)
            conditions = await _run_conditions(pool, case["question"])
            per_question.append({
                "id": case["id"],
                "question": case["question"],
                "expected": set(case["expected_articles"]),
                "primary_article": case.get("primary_article"),
                "conditions": conditions,
            })
    finally:
        await pool.close()

    return per_question


def compute_metrics(results: list[dict]) -> dict:
    """Recall@1 / MRR / Recall@5 per condition, averaged over on-topic
    questions only -- plus the off-topic top-1-score comparison."""
    on_topic = [r for r in results if r["expected"]]
    off_topic = [r for r in results if not r["expected"]]

    metrics = {}
    for condition in CONDITIONS:
        recall1_scores, mrr_scores, recall5_scores, ndcg_scores = [], [], [], []
        for r in on_topic:
            retrieved = r["conditions"][condition]["articles"]
            expected = r["expected"]
            recall1_scores.append(recall_at_1(retrieved, expected))
            mrr_scores.append(mrr(retrieved, expected))
            recall5_scores.append(recall_at_k(retrieved, expected, k=5))
            ndcg_scores.append(
                ndcg_at_k(retrieved, expected, k=5, primary=r["primary_article"])
            )

        on_topic_top1_avg = sum(
            r["conditions"][condition]["top1_score"] for r in on_topic
        ) / len(on_topic)

        metrics[condition] = {
            "recall@1": sum(recall1_scores) / len(recall1_scores),
            "mrr": sum(mrr_scores) / len(mrr_scores),
            "recall@5": sum(recall5_scores) / len(recall5_scores),
            "ndcg@5": sum(ndcg_scores) / len(ndcg_scores),
            "on_topic_top1_avg": on_topic_top1_avg,
            "off_topic_top1": {
                r["id"]: r["conditions"][condition]["top1_score"] for r in off_topic
            },
        }

    return metrics


def print_report(results: list[dict], metrics: dict) -> None:
    for r in results:
        print(f"{r['id']}: {r['question']}")
        print(f"  expected={sorted(r['expected'])}")
        for condition in CONDITIONS:
            c = r["conditions"][condition]
            print(f"  [{condition}] retrieved={c['articles']} top1_score={c['top1_score']:.3f}")
        print()

    print("=" * 80)
    print(f"{'condition':<18}{'Recall@1':>10}{'MRR':>10}{'Recall@5':>10}{'nDCG@5':>10}")
    for condition in CONDITIONS:
        m = metrics[condition]
        print(
            f"{condition:<18}{m['recall@1']:>10.3f}{m['mrr']:>10.3f}"
            f"{m['recall@5']:>10.3f}{m['ndcg@5']:>10.3f}"
        )

    print("\nOff-topic questions -- top-1 score vs. on-topic average:")
    for condition in CONDITIONS:
        m = metrics[condition]
        print(
            f"  [{condition}] on-topic avg top1={m['on_topic_top1_avg']:.3f} "
            f"| off-topic: {m['off_topic_top1']}"
        )


def _serialize_results(results: list[dict]) -> list[dict]:
    """JSON can't serialize sets -- convert `expected` to a sorted list."""
    return [
        {**r, "expected": sorted(r["expected"])}
        for r in results
    ]


def save_report(results: list[dict], metrics: dict) -> Path:
    """Persist the per-question results and the summary metrics to a
    timestamped JSON file under eval/results/, so runs can be compared
    over time (e.g. before/after enabling the reranker)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"retrieval_eval_{timestamp}.json"

    report = {
        "generated_at": timestamp,
        "results": _serialize_results(results),
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    results = asyncio.run(run_eval())
    metrics = compute_metrics(results)
    print_report(results, metrics)
    saved_path = save_report(results, metrics)
    print(f"\nRapport sauvegarde dans: {saved_path}")
