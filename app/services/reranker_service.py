"""
app/services/reranker_service.py

Cross-encoder reranking: Top-100 recall (from hybrid_retrieve) →
Top-10 precision (per blueprint spec).

Why rerank at all when RRF already produced a ranked list: RRF only
knows rank position from two cheap, independent retrieval signals
(BM25 term overlap, dense cosine similarity). A cross-encoder reranker
actually reads the query and each candidate chunk together and scores
their semantic relevance jointly — far more accurate, but too expensive
to run over an entire corpus. The two-stage design (cheap broad recall,
then expensive narrow precision) is standard IR practice.

Two backends, selected via settings.USE_LOCAL_RERANKER:
  - Cohere Rerank v3 API (default) — no local model load, network call
  - BGE-Reranker-Large (local fallback) — for offline/no-API-key operation

If COHERE_API_KEY is missing and USE_LOCAL_RERANKER is False, this
raises at call time rather than silently falling back — a misconfigured
reranker should be loud, not silently degrade retrieval quality.
"""

import asyncio

import cohere
import structlog

from app.config import get_settings
from app.core.exceptions import RerankerError

logger = structlog.get_logger(__name__)
settings = get_settings()

_cohere_client: cohere.AsyncClient | None = None
_local_reranker = None
_local_reranker_lock = asyncio.Lock()


def _get_cohere_client() -> cohere.AsyncClient:
    global _cohere_client
    if _cohere_client is None:
        if not settings.COHERE_API_KEY:
            raise RerankerError()
        _cohere_client = cohere.AsyncClient(api_key=settings.COHERE_API_KEY)
    return _cohere_client


async def _get_local_reranker():
    global _local_reranker
    if _local_reranker is None:
        async with _local_reranker_lock:
            if _local_reranker is None:
                logger.info("loading_local_reranker")
                _local_reranker = await asyncio.to_thread(_load_local_reranker_sync)
                logger.info("local_reranker_loaded")
    return _local_reranker


def _load_local_reranker_sync():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("BAAI/bge-reranker-large", device="cpu")


async def rerank(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """
    Takes the fused candidate list from hybrid_retrieve (each dict has
    at least "chunk_id" and "text"), returns the top_k most relevant,
    re-scored and re-ordered.

    Each returned dict has the original fields plus "rerank_score".
    """
    top_k = top_k or settings.RERANKER_TOP_K

    if not candidates:
        return []

    try:
        if settings.USE_LOCAL_RERANKER:
            return await _rerank_local(query, candidates, top_k)
        return await _rerank_cohere(query, candidates, top_k)
    except RerankerError:
        raise
    except Exception as e:
        logger.error("rerank_failed", error=str(e))
        raise RerankerError() from e


async def _rerank_cohere(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    client = _get_cohere_client()
    documents = [c["text"] for c in candidates]

    response = await client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=min(top_k, len(documents)),
    )

    reranked = []
    for result in response.results:
        original = candidates[result.index]
        reranked.append({**original, "rerank_score": result.relevance_score})

    logger.debug("cohere_rerank_complete", candidate_count=len(candidates), returned=len(reranked))
    return reranked


async def _rerank_local(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    model = await _get_local_reranker()
    pairs = [(query, c["text"]) for c in candidates]

    scores = await asyncio.to_thread(model.predict, pairs)

    scored = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(candidates, scores)
    ]
    scored.sort(key=lambda r: r["rerank_score"], reverse=True)

    logger.debug("local_rerank_complete", candidate_count=len(candidates), returned=top_k)
    return scored[:top_k]