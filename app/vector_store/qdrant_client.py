"""
app/vector_store/qdrant_client.py

Qdrant client wrapper. Collection naming uses sha256(user_id), not the
raw UUID — if Qdrant's admin interface is ever accidentally exposed,
raw UUIDs are enumerable; hashed names add a meaningful obscurity layer.

One collection per user enforces hard isolation at the vector store
level — even a bug in application-layer filtering cannot leak another
user's vectors, because the query physically cannot reach their collection.
"""

import hashlib
import uuid

import structlog
from qdrant_client import AsyncQdrantClient, models

from app.config import get_settings
from app.core.exceptions import RetrievalError

logger = structlog.get_logger(__name__)
settings = get_settings()

_client: AsyncQdrantClient | None = None


def get_qdrant_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=str(settings.QDRANT_URL),
            api_key=settings.QDRANT_API_KEY,
        )
    return _client


def collection_name_for_user(user_id: uuid.UUID | str) -> str:
    """
    sha256(user_id) truncated to 32 hex chars, prefixed for readability
    in Qdrant's dashboard. Truncation is fine — collision probability
    at 32 hex chars (128 bits) is astronomically low for any realistic
    user count.
    """
    digest = hashlib.sha256(str(user_id).encode()).hexdigest()[:32]
    return f"user_{digest}"


async def ensure_collection_exists(user_id: uuid.UUID | str) -> str:
    """
    Idempotent — creates the user's collection if it doesn't exist yet.
    Call this before the first ingestion for any user.
    """
    client = get_qdrant_client()
    collection = collection_name_for_user(user_id)

    try:
        exists = await client.collection_exists(collection)
        if not exists:
            await client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=settings.QDRANT_COLLECTION_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
                hnsw_config=models.HnswConfigDiff(
                    ef_construct=settings.QDRANT_HNSW_EF_CONSTRUCT,
                    m=settings.QDRANT_HNSW_M,
                ),
            )
            logger.info("qdrant_collection_created", collection=collection)
    except Exception as e:
        logger.error("qdrant_collection_create_failed", error=str(e))
        raise RetrievalError() from e

    return collection


async def upsert_chunks(
    user_id: uuid.UUID | str,
    document_id: uuid.UUID | str,
    chunk_ids: list[str],
    embeddings: list[list[float]],
    chunk_texts: list[str],
    chunk_metadata: list[dict] | None = None,
) -> None:
    """
    Write embedded chunks to the user's collection.
    chunk_metadata, if provided, is merged into each point's payload
    (e.g. page number, document filename, custom metadata tags).
    """
    if len(chunk_ids) != len(embeddings) or len(chunk_ids) != len(chunk_texts):
        raise ValueError("chunk_ids, embeddings, and chunk_texts must be the same length")

    client = get_qdrant_client()
    collection = await ensure_collection_exists(user_id)

    points = []
    for i, (cid, vector, text) in enumerate(zip(chunk_ids, embeddings, chunk_texts)):
        payload = {
            "document_id": str(document_id),
            "user_id": str(user_id),  # defense-in-depth — also filter on this at query time
            "chunk_index": i,
            "text": text,
        }
        if chunk_metadata and i < len(chunk_metadata):
            payload.update(chunk_metadata[i])

        points.append(models.PointStruct(id=cid, vector=vector, payload=payload))

    try:
        await client.upsert(collection_name=collection, points=points)
        logger.info(
            "qdrant_chunks_upserted",
            collection=collection,
            document_id=str(document_id),
            chunk_count=len(points),
        )
    except Exception as e:
        logger.error("qdrant_upsert_failed", error=str(e))
        raise RetrievalError() from e


async def delete_document_chunks(user_id: uuid.UUID | str, document_id: uuid.UUID | str) -> int:
    """
    Delete all chunks belonging to a document. Used by DELETE /documents/{id}.
    Filters on document_id within the user's own collection — cannot
    accidentally touch another user's data since collections are isolated.
    """
    client = get_qdrant_client()
    collection = collection_name_for_user(user_id)

    try:
        if not await client.collection_exists(collection):
            return 0

        result = await client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
        )
        logger.info("qdrant_chunks_deleted", collection=collection, document_id=str(document_id))
        return 1  # Qdrant delete doesn't return count directly; caller tracks via Postgres chunk_count
    except Exception as e:
        logger.error("qdrant_delete_failed", error=str(e))
        raise RetrievalError() from e