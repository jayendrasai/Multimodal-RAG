"""
app/services/retrieval_service.py

Hybrid retrieval: BM25 (Elasticsearch) + dense (Qdrant), fused with
Reciprocal Rank Fusion (k=60, per blueprint spec).

RRF formula: score(d) = sum over each ranking list r where d appears of
  1 / (k + rank_r(d))

Why RRF over a weighted score blend: BM25 scores and cosine similarity
scores live on completely different numeric scales (BM25 is unbounded,
cosine is [-1, 1]) and aren't comparable without normalization that's
fragile across corpora. RRF only uses RANK POSITION within each list,
not the raw score — it's scale-invariant and well-established for
exactly this kind of heterogeneous fusion. k=60 is the standard
damping constant from the original RRF paper; it down-weights the
importance of rank differences at the tail of each list.

Both searches run concurrently (asyncio.gather) — they hit two
completely independent stores and have no data dependency on each other.
"""

import asyncio
import uuid

import structlog

from app.config import get_settings
from app.search.es_search import bm25_search
from app.vector_store.embedder import embed_query
from app.vector_store.qdrant_search import dense_search

logger = structlog.get_logger(__name__)
settings = get_settings()


async def hybrid_retrieve(
    user_id: uuid.UUID | str,
    query: str,
    top_k: int | None = None,
) -> list[dict]:
    """
    Runs BM25 and dense search concurrently, fuses with RRF, returns the
    top_k fused results (default: settings.RERANKER_RECALL_K = 100,
    matching the blueprint's "Top-100 recall" stage before reranking).

    Each returned dict:
      {"chunk_id", "text", "document_id", "filename", "rrf_score"}
    """
    top_k = top_k or settings.RERANKER_RECALL_K

    # BM25 needs only the raw query text. Dense search needs an embedding —
    # compute it once here, not inside dense_search, so it's visible and
    # testable as a distinct step in the pipeline.
    query_vector = await embed_query(query)

    bm25_results, dense_results = await asyncio.gather(
        bm25_search(user_id, query, top_k=top_k),
        dense_search(user_id, query_vector, top_k=top_k),
    )

    fused = _rrf_fuse(bm25_results, dense_results, k=settings.RRF_K)

    logger.info(
        "hybrid_retrieval_complete",
        user_id=str(user_id),
        bm25_count=len(bm25_results),
        dense_count=len(dense_results),
        fused_count=len(fused),
    )

    return fused[:top_k]


def _rrf_fuse(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Fuses two ranked lists by chunk_id using Reciprocal Rank Fusion.

    A chunk that appears in both lists accumulates score from both ranks —
    this is the whole point of hybrid retrieval: a chunk that's both
    lexically relevant (BM25) and semantically relevant (dense) ranks
    higher than one that only satisfies one signal.
    """
    scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for rank, result in enumerate(bm25_results, start=1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_data[cid] = result

    for rank, result in enumerate(dense_results, start=1):
        cid = result["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        # If this chunk wasn't in BM25 results, dense_results provides the data
        chunk_data.setdefault(cid, result)

    fused = [
        {**chunk_data[cid], "rrf_score": score}
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda r: r["rrf_score"], reverse=True)

    return fused