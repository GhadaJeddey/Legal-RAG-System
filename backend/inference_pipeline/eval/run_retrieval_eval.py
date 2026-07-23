"""Compare the RAG pipeline's retrieved sources against expected articles.

For each question in retrieval_eval_set.json, runs the full rag_pipeline
(retrieval + generation) and checks whether the expected article(s) show up
among the sources the LLM's answer was actually grounded on.

Usage:
    python run_retrieval_eval.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import asyncpg
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from rag_pipeline import answer_query  # noqa: E402

EVAL_SET_PATH = Path(__file__).with_name("retrieval_eval_set.json")


async def _init_connection(conn: asyncpg.Connection):
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


async def run_eval():
    load_dotenv()
    cases = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))

    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], init=_init_connection)
    groq_api_key = os.environ["GROQ_API_KEY"]

    results = []
    try:
        for case in cases:
            result = await answer_query(
                pool=pool, query=case["question"], groq_api_key=groq_api_key
            )
            retrieved = {s["article_number"] for s in result["sources"] if s["article_number"]}
            expected = set(case["expected_articles"])

            if expected:
                passed = bool(expected & retrieved)
            else:
                passed = not retrieved

            results.append({
                "id": case["id"],
                "question": case["question"],
                "expected": sorted(expected),
                "retrieved": sorted(retrieved),
                "passed": passed,
            })
    finally:
        await pool.close()

    return results


def print_report(results: list[dict]):
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['id']}: {r['question']}")
        print(f"       expected={r['expected']} retrieved={r['retrieved']}")

    n_passed = sum(r["passed"] for r in results)
    print(f"\n{n_passed}/{len(results)} passed ({round(100 * n_passed / len(results), 1)}%)")


if __name__ == "__main__":
    results = asyncio.run(run_eval())
    print_report(results)
