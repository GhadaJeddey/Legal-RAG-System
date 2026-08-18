"""Rank-sensitive retrieval metrics.

All four take an ordered list of retrieved article numbers (best match first,
deduplicated -- a document can have several chunks/sub-chunks sharing the same
article_number) and the set of expected article numbers for that question.

Undefined for `expected == set()` (off-topic questions) -- callers should
exclude those questions from these metrics rather than pass an empty set in.
"""
import math


def recall_at_1(retrieved: list[str], expected: set[str]) -> float:
    """1 if the very first retrieved article is an expected one, else 0."""
    if not retrieved:
        return 0.0
    return 1.0 if retrieved[0] in expected else 0.0


def mrr(retrieved: list[str], expected: set[str]) -> float:
    """1 / rank of the first retrieved article that's expected, 0 if none found."""
    for rank, article in enumerate(retrieved, start=1):
        if article in expected:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved: list[str], expected: set[str], k: int = 5) -> float:
    """Fraction of expected articles found among the first k retrieved."""
    found = set(retrieved[:k]) & expected
    return len(found) / len(expected)


def _dcg(gains: list[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=2))


def ndcg_at_k(
    retrieved: list[str], expected: set[str], k: int = 5, primary: str | None = None
) -> float:
    """Position-weighted, graded-relevance recall.

    Each expected article gets a gain: 2.0 if it's `primary` (the article
    judged most central to the question), 1.0 otherwise. When `primary` is
    None -- i.e. all expected articles are equally important -- every
    expected article gets the same gain, so this reduces to a position-
    weighted version of recall_at_k rather than requiring a forced hierarchy.

    Normalized against the ideal ranking (all expected articles, best gain
    first, in the top k), so the result is always in [0, 1] regardless of
    how many articles are expected.
    """
    def gain(article: str) -> float:
        if article == primary:
            return 2.0
        return 1.0 if article in expected else 0.0

    dcg = _dcg([gain(article) for article in retrieved[:k]])

    ideal_gains = sorted((2.0 if a == primary else 1.0 for a in expected), reverse=True)[:k]
    idcg = _dcg(ideal_gains)

    return dcg / idcg if idcg else 0.0
