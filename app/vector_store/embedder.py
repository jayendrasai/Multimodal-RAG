"""
app/vector_store/embedder.py

Embedding generation. Two backends supported, selected via
settings.USE_LOCAL_EMBEDDING:
  - Local: BGE-M3 via FlagEmbedding (CPU torch — matches your Dockerfile)
  - API: Jina AI v3 (not implemented in Phase 1 stub — local is the
    pinned choice per ADR-001)

CRITICAL: EMBEDDING_MODEL_VERSION in config is the source of truth for
which model produced existing vectors. Changing it without a full
reindex of every user's Qdrant collection corrupts retrieval — old
vectors and new vectors are not comparable.

The model is loaded once at module level (lazy singleton) — loading
BGE-M3 takes several seconds and must not happen per-request.
"""

import asyncio
import structlog

from app.config import get_settings
from app.core.exceptions import EmbeddingError

logger = structlog.get_logger(__name__)
settings = get_settings()

_model = None
_model_lock = asyncio.Lock()


async def _get_model():
    """
    Lazy-loaded singleton. First call loads the model (slow, ~3-10s on CPU).
    Subsequent calls reuse the loaded instance.
    """
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:  # re-check after acquiring lock
                logger.info("loading_embedding_model", model=settings.EMBEDDING_MODEL_VERSION)
                _model = await asyncio.to_thread(_load_model_sync)
                logger.info("embedding_model_loaded")
    return _model


def _load_model_sync():
    from FlagEmbedding import BGEM3FlagModel
    # use_fp16=False — CPU inference does not benefit from fp16 and some
    # CPU backends error on it. Explicit False avoids ambiguity.
    return BGEM3FlagModel(settings.EMBEDDING_MODEL_VERSION, use_fp16=False)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts. Returns dense vectors only (BGE-M3 also
    supports sparse + multi-vector output — Phase 1 uses dense only;
    BM25 via Elasticsearch covers the sparse/lexical retrieval need).
    """
    if not texts:
        return []

    try:
        model = await _get_model()
        result = await asyncio.to_thread(
            lambda: model.encode(texts, batch_size=12, max_length=8192)
        )
        # FlagEmbedding's encode returns a dict with 'dense_vecs' when
        # using BGEM3FlagModel — extract and convert to plain lists
        dense_vecs = result["dense_vecs"]
        return [vec.tolist() for vec in dense_vecs]
    except Exception as e:
        logger.error("embedding_failed", error=str(e), text_count=len(texts))
        raise EmbeddingError() from e


async def embed_query(query: str) -> list[float]:
    """Embed a single query string. Used at retrieval time, not ingestion."""
    vectors = await embed_texts([query])
    return vectors[0]