"""
Evaluate an embedding model's ability to place chunk breakpoints in the
right place, independent of retrieval quality.

Annotation format (one JSON object per line, see
segmentation_annotations.example.jsonl):

    {
      "article_id": "Art-12",
      "article_title": "...",
      "units": ["sentence 1", "sentence 2", ...],
      "ideal_breakpoints": [5]
    }

`units` must be the same ordered sentence/table units the chunker would
produce for that article body (i.e. the output of
chunkers.article_chunker._split_into_units). `ideal_breakpoints` is the
list of unit indices where a human annotator would start a new chunk —
index i means "a new chunk starts at units[i]", so a breakpoint list of
[5] on an 8-unit article means chunks are units[0:5] and units[5:8].

Usage:
    python -m eval.segmentation_metrics annotations.jsonl BAAI/bge-m3 intfloat/multilingual-e5-large
    python -m eval.segmentation_metrics --sweep annotations.jsonl BAAI/bge-m3 intfloat/multilingual-e5-large
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chunkers.article_chunker import BREAKPOINT_PERCENTILE  # noqa: E402


def _boundary_vector(breakpoint_indices: list[int], n_units: int) -> list[int]:
    """Convert breakpoint indices into a binary vector of length n_units-1,
    where position i is 1 if there's a boundary between units[i] and units[i+1]."""
    vec = [0] * (n_units - 1)
    for idx in breakpoint_indices:
        if 0 < idx < n_units:
            vec[idx - 1] = 1
    return vec


def predict_breakpoints(units: list[str], embeddings: np.ndarray, percentile: int) -> list[int]:
    """Mirror ArticleChunker's pre-merge breakpoint detection: this isolates
    the embedding model's segmentation signal from the token-count-driven
    merge step, which is deliberately excluded here."""
    distances = [
        1 - float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]
    if not distances:
        return []
    threshold = float(np.percentile(distances, percentile))
    return [i + 1 for i, d in enumerate(distances) if d > threshold]


def _default_window(ref: list[int]) -> int:
    """Standard Pk/WindowDiff convention: half the average segment length."""
    boundaries = [i for i, b in enumerate(ref) if b == 1]
    n_units = len(ref) + 1
    n_segments = len(boundaries) + 1
    avg_seg_len = n_units / n_segments
    return max(1, round(avg_seg_len / 2))


def pk_metric(ref: list[int], hyp: list[int], k: int | None = None) -> float:
    """Beeferman et al. (1999) Pk: probability that a randomly placed window
    of size k disagrees on whether its two ends are in the same segment."""
    n = len(ref)
    if n == 0:
        return 0.0
    k = k or _default_window(ref)
    k = min(k, n)

    def same_segment(vec: list[int], i: int, j: int) -> bool:
        return sum(vec[i:j]) == 0

    disagreements = 0
    total = 0
    for i in range(n - k + 1):
        j = i + k
        if same_segment(ref, i, j) != same_segment(hyp, i, j):
            disagreements += 1
        total += 1
    return disagreements / total if total else 0.0


def window_diff(ref: list[int], hyp: list[int], k: int | None = None) -> float:
    """Pevzner & Hearst (2002) WindowDiff: like Pk but compares the *count*
    of boundaries inside each window rather than same/different segment."""
    n = len(ref)
    if n == 0:
        return 0.0
    k = k or _default_window(ref)
    k = min(k, n)

    disagreements = 0
    total = 0
    for i in range(n - k + 1):
        j = i + k
        if sum(ref[i:j]) != sum(hyp[i:j]):
            disagreements += 1
        total += 1
    return disagreements / total if total else 0.0


def load_annotations(path: str) -> list[dict]:
    annotations = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    return annotations


def evaluate_model(model, annotations: list[dict], percentile: int = BREAKPOINT_PERCENTILE) -> dict:
    pk_scores, wd_scores = [], []
    for ann in annotations:
        units = ann["units"]
        if len(units) < 2:
            continue
        embeddings = model.encode(units, normalize_embeddings=True, show_progress_bar=False)
        hyp_breaks = predict_breakpoints(units, embeddings, percentile)

        ref_vec = _boundary_vector(ann["ideal_breakpoints"], len(units))
        hyp_vec = _boundary_vector(hyp_breaks, len(units))

        pk_scores.append(pk_metric(ref_vec, hyp_vec))
        wd_scores.append(window_diff(ref_vec, hyp_vec))

    return {
        "n_articles": len(pk_scores),
        "pk": round(sum(pk_scores) / len(pk_scores), 4) if pk_scores else None,
        "window_diff": round(sum(wd_scores) / len(wd_scores), 4) if wd_scores else None,
    }


def sweep_percentile(
    model, annotations: list[dict], percentiles: list[int] | None = None
) -> list[dict]:
    """Score a model at each candidate percentile so it's compared to other
    models at its own best setting rather than a shared default of 90."""
    percentiles = percentiles or list(range(80, 100))

    # Embed once per article, reuse across percentiles.
    cached = []
    for ann in annotations:
        units = ann["units"]
        if len(units) < 2:
            continue
        embeddings = model.encode(units, normalize_embeddings=True, show_progress_bar=False)
        ref_vec = _boundary_vector(ann["ideal_breakpoints"], len(units))
        cached.append((units, embeddings, ref_vec))

    results = []
    for p in percentiles:
        pk_scores, wd_scores = [], []
        for units, embeddings, ref_vec in cached:
            hyp_breaks = predict_breakpoints(units, embeddings, p)
            hyp_vec = _boundary_vector(hyp_breaks, len(units))
            pk_scores.append(pk_metric(ref_vec, hyp_vec))
            wd_scores.append(window_diff(ref_vec, hyp_vec))
        results.append({
            "percentile": p,
            "n_articles": len(pk_scores),
            "pk": round(sum(pk_scores) / len(pk_scores), 4) if pk_scores else None,
            "window_diff": round(sum(wd_scores) / len(wd_scores), 4) if wd_scores else None,
        })
    return results


def main():
    args = sys.argv[1:]
    sweep = False
    if args and args[0] == "--sweep":
        sweep = True
        args = args[1:]

    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    annotations_path = args[0]
    model_names = args[1:]

    from sentence_transformers import SentenceTransformer

    annotations = load_annotations(annotations_path)
    print(f"Loaded {len(annotations)} annotated articles from {annotations_path}\n")

    if sweep:
        for name in model_names:
            print(f"Loading {name} ...")
            model = SentenceTransformer(name)
            results = sweep_percentile(model, annotations)
            print(f"\n{name}")
            print(f"{'percentile':>10} {'n':>4} {'Pk (lower=better)':>20} {'WindowDiff (lower=better)':>26}")
            for r in results:
                print(f"{r['percentile']:>10} {r['n_articles']:>4} {r['pk']:>20} {r['window_diff']:>26}")
        return

    results = []
    for name in model_names:
        print(f"Loading {name} ...")
        model = SentenceTransformer(name)
        scores = evaluate_model(model, annotations)
        scores["model"] = name
        results.append(scores)

    print(f"\n{'model':<45} {'n':>4} {'Pk (lower=better)':>20} {'WindowDiff (lower=better)':>26}")
    for r in results:
        print(f"{r['model']:<45} {r['n_articles']:>4} {r['pk']:>20} {r['window_diff']:>26}")


if __name__ == "__main__":
    main()