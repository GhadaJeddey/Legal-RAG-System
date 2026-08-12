"""Cross-encoder (bge-reranking model)  re-ranking of the bi-encoder's candidate chunks.
The model takes as input the following : [CLS] query [SEP] chunk [SEP] , computes attetion accross 
full input and returns relevance score 
"""

_model = None


def get_reranker():
    """Lazily load the cross-encoder used to re-rank retrieved chunks."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Re-order `chunks` (as returned by retriever.retrieve) by cross-encoder
    relevance to `query`, keeping the top_k. Each chunk's original `rank_bi`
    (its dense-retrieval rank) is preserved so the effect of re-ranking can
    be traced -- e.g. a chunk moving from rank_bi=3 to rank_cross=1.
    """
    
    model = get_reranker()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)[:top_k]

    return [
        {**chunk, "rerank_score": float(score), "rank_cross": i + 1}
        for i, (chunk, score) in enumerate(ranked)
    ]
