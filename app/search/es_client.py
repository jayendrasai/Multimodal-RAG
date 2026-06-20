"""
app/search/es_client.py

Elasticsearch wrapper for BM25 lexical search.
Index naming mirrors the Qdrant collection convention: one index per
user, named with the same sha256(user_id) scheme, for consistency and
the same enumeration-resistance rationale.
"""

import hashlib
import uuid

import structlog
from elasticsearch import AsyncElasticsearch, NotFoundError

from app.config import get_settings
from app.core.exceptions import RetrievalError

logger = structlog.get_logger(__name__)
settings = get_settings()

_client: AsyncElasticsearch | None = None


def get_es_client() -> AsyncElasticsearch:
    global _client
    if _client is None:
        _client = AsyncElasticsearch(
            str(settings.ELASTIC_URL),
            basic_auth=(settings.ELASTIC_USER, settings.ELASTIC_PASSWORD),
        )
    return _client


def index_name_for_user(user_id: uuid.UUID | str) -> str:
    digest = hashlib.sha256(str(user_id).encode()).hexdigest()[:32]
    return f"es_user_{digest}"


async def ensure_index_exists(user_id: uuid.UUID | str) -> str:
    client = get_es_client()
    index = index_name_for_user(user_id)

    try:
        exists = await client.indices.exists(index=index)
        if not exists:
            await client.indices.create(
                index=index,
                mappings={
                    "properties": {
                        "document_id": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "chunk_index": {"type": "integer"},
                        "text": {"type": "text", "analyzer": "english"},
                        "filename": {"type": "keyword"},
                    }
                },
            )
            logger.info("es_index_created", index=index)
    except Exception as e:
        logger.error("es_index_create_failed", error=str(e))
        raise RetrievalError() from e

    return index


async def index_chunks(
    user_id: uuid.UUID | str,
    document_id: uuid.UUID | str,
    chunk_ids: list[str],
    chunk_texts: list[str],
    chunk_metadata: list[dict] | None = None,
) -> None:
    client = get_es_client()
    index = await ensure_index_exists(user_id)

    operations = []
    for i, (cid, text) in enumerate(zip(chunk_ids, chunk_texts)):
        doc = {
            "document_id": str(document_id),
            "user_id": str(user_id),
            "chunk_index": i,
            "text": text,
        }
        if chunk_metadata and i < len(chunk_metadata):
            doc["filename"] = chunk_metadata[i].get("filename")

        operations.append({"index": {"_index": index, "_id": cid}})
        operations.append(doc)

    try:
        response = await client.bulk(operations=operations)
        if response.get("errors"):
            failed = [item for item in response["items"] if item.get("index", {}).get("error")]
            logger.error("es_bulk_index_partial_failure", failed_count=len(failed))
            raise RetrievalError()
        logger.info("es_chunks_indexed", index=index, document_id=str(document_id), count=len(chunk_ids))
    except RetrievalError:
        raise
    except Exception as e:
        logger.error("es_bulk_index_failed", error=str(e))
        raise RetrievalError() from e


async def delete_document_chunks(user_id: uuid.UUID | str, document_id: uuid.UUID | str) -> None:
    client = get_es_client()
    index = index_name_for_user(user_id)

    try:
        await client.delete_by_query(
            index=index,
            query={"term": {"document_id": str(document_id)}},
        )
        logger.info("es_chunks_deleted", index=index, document_id=str(document_id))
    except NotFoundError:
        pass  # Index doesn't exist yet — nothing to delete
    except Exception as e:
        logger.error("es_delete_failed", error=str(e))
        raise RetrievalError() from e