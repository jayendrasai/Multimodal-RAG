"""
app/vector_store/qdrant_search.py

Dense vector search against a user's Qdrant collection.
Separated from qdrant_client.py (writes) for the same reason as
es_search.py — read and write paths kept distinct.
"""

import uuid

import structlog

from app.core.exceptions import RetrievalError
from app.vector_store.qdrant_client import collection_name_for_user, get_qdrant_client

logger = structlog.get_logger(__name__)


async def dense_search(
    user_id: uuid.UUID | str,
    query_vector: list[float],
    top_k: int = 100,
) -> list[dict]:
    """
    Returns up to top_k results as:
      [{"chunk_id": str, "text": str, "document_id": str, "score": float, "filename": str}, ...]

    Isolation is structural: the collection name is derived from user_id
    and no other collection is ever queried in this call. Even if the
    caller passed a malicious query_vector, it cannot retrieve another
    user's data because the search only ever touches one collection.
    """
    client = get_qdrant_client()
    collection = collection_name_for_user(user_id)

    try:
        if not await client.collection_exists(collection):
            return []  # User has no documents yet — not an error

        response = await client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
    except Exception as e:
        logger.error("dense_search_failed", error=str(e))
        raise RetrievalError() from e

    results = []
    for point in response.points:
        payload = point.payload or {}
        results.append({
            "chunk_id": str(point.id),
            "text": payload.get("text", ""),
            "document_id": payload.get("document_id"),
            "score": point.score,
            "filename": payload.get("filename"),
        })

    logger.debug("dense_search_complete", user_id=str(user_id), result_count=len(results))
    return results