"""
app/search/es_search.py

BM25 lexical search against a user's Elasticsearch index.
Separated from es_client.py (which handles indexing) to keep
read and write paths distinct — search is called on every query,
indexing only on ingestion.
"""

import uuid

import structlog
from elasticsearch import NotFoundError

from app.core.exceptions import RetrievalError
from app.search.es_client import get_es_client, index_name_for_user

logger = structlog.get_logger(__name__)


async def bm25_search(
    user_id: uuid.UUID | str,
    query: str,
    top_k: int = 100,
) -> list[dict]:
    """
    Returns up to top_k results as:
      [{"chunk_id": str, "text": str, "document_id": str, "score": float, "filename": str}, ...]

    user_id filter is applied via the index name itself (per-user index) —
    blueprint requires this be "not optional." There is no shared index
    where a missing filter could leak across users; the physical index
    boundary makes the isolation structural, not a query-time choice.
    """
    client = get_es_client()
    index = index_name_for_user(user_id)

    try:
        response = await client.search(
            index=index,
            query={"match": {"text": query}},
            size=top_k,
        )
    except NotFoundError:
        # User has no documents indexed yet — not an error, just empty.
        return []
    except Exception as e:
        logger.error("bm25_search_failed", error=str(e))
        raise RetrievalError() from e

    results = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        results.append({
            "chunk_id": hit["_id"],
            "text": source["text"],
            "document_id": source["document_id"],
            "score": hit["_score"],
            "filename": source.get("filename"),
        })

    logger.debug("bm25_search_complete", user_id=str(user_id), result_count=len(results))
    return results