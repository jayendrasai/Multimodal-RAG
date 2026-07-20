import logging
import uuid
from app.core.celery_app import celery_app
from app.db.sync_session import SyncSessionLocal
from app.vector_store.qdrant_client import get_sync_qdrant_client
from app.search.es_client import get_sync_es_client
from sqlalchemy import text
from app.vector_store.embedder import _load_model_sync

logger = logging.getLogger(__name__)
MAX_RETRIES = 5

# FIX 2: Synchronous singleton for the worker process
_sync_embedding_model = None

def get_sync_embedding_model():
    global _sync_embedding_model
    if _sync_embedding_model is None:
        logger.info("loading synchronous embedding model for celery worker...")
        _sync_embedding_model = _load_model_sync()
    return _sync_embedding_model

@celery_app.task
def relay_outbox_batch():
    with SyncSessionLocal() as session:
        rows = session.execute(
            text(
                """
                SELECT * FROM outbox
                WHERE status = 'PENDING'
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 100
                """
            )
        ).mappings().all()

        for row in rows:
            try:
                # Route the row to the correct handler based on the operation intent
                if row["operation_type"] == "UPSERT_MEMORY":
                    _handle_upsert_memory(row)
                elif row["operation_type"] == "UPSERT_CHUNK":
                    _handle_upsert_chunk(row)
                elif row["operation_type"] == "DELETE_CHUNKS":
                    _handle_delete_chunks(row, session)
                else:
                    raise ValueError(f"unknown operation_type {row['operation_type']}")

                # Mark as published and commit
                session.execute(
                    text("UPDATE outbox SET status='PUBLISHED' WHERE id=:id"), 
                    {"id": row["id"]}
                )
                session.commit()

                # If this event belongs to a document, check if the document is completely processed
                if row["document_id"] is not None:
                    _maybe_mark_document_ready(session, row["document_id"])

            except Exception:
                logger.exception("outbox relay failed for row %s", row["id"])
                new_count = row["retry_count"] + 1
                status = "DEAD" if new_count >= MAX_RETRIES else "PENDING"
                
                session.execute(
                    text("UPDATE outbox SET retry_count=:rc, status=:st WHERE id=:id"),
                    {"rc": new_count, "st": status, "id": row["id"]},
                )
                session.commit()
                
                if status == "DEAD":
                    logger.error("outbox row %s dead-lettered after %s retries", row["id"], new_count)


def _handle_upsert_memory(row):
    """Handles semantic_memory / episodic_memory -> Qdrant only."""
    qdrant = get_sync_qdrant_client()
    qdrant.upsert(
        collection_name="memory",
        points=[{
            "id": str(row["aggregate_id"]), 
            "vector": row["payload"]["embedding"], 
            "payload": row["payload"]["metadata"]
        }],
    )


def _handle_upsert_chunk(row):
    model = get_sync_embedding_model()
    text_content = row["payload"]["text"]
    
    # FIX 3: Extract dense_vecs from the BGE-M3 dictionary response
    encoded = model.encode([text_content], batch_size=1, max_length=8192)
    vector = encoded["dense_vecs"][0].tolist()

    qdrant = get_sync_qdrant_client()
    collection = f"user_docs_{row['payload']['metadata'].get('user_hash', 'default')}"
    
    qdrant.upsert(
        collection_name=collection,
        points=[{
            "id": str(row["aggregate_id"]), 
            "vector": vector, 
            "payload": row["payload"]["metadata"]
        }],
    )

    es = get_sync_es_client()
    es.index(
        index=collection,
        id=str(row["aggregate_id"]),
        document={
            "text": text_content, 
            "chunk_index": row["payload"]["chunk_index"]
        },
    )


def _handle_delete_chunks(row, session):
    doc_id = row["document_id"]
    
    # See note below about fetching chunk_count
    chunk_count = row["payload"].get("chunk_count")
    if chunk_count is None:
        chunk_count = session.execute(
            text("SELECT chunk_count FROM documents WHERE id = :id"),
            {"id": doc_id}
        ).scalar_one()
    
    qdrant = get_sync_qdrant_client()
    es = get_sync_es_client()

    point_ids = [str(uuid.uuid5(uuid.UUID(str(doc_id)), str(i))) for i in range(chunk_count)]
    
    # Note: user_docs collection routing must match your ingestion setup exactly
    qdrant.delete(collection_name="user_docs", points_selector=point_ids)
    
    for pid in point_ids:
        es.delete(index="user_docs", id=pid, ignore_status=[404])


def _maybe_mark_document_ready(session, document_id):
    """
    Checks if all chunks for this document are PUBLISHED. 
    Guarded by 'PROCESSING' check to prevent duplicate firing.
    """
    remaining = session.execute(
        text("SELECT count(*) FROM outbox WHERE document_id=:doc_id AND status != 'PUBLISHED'"),
        {"doc_id": document_id},
    ).scalar_one()

    if remaining == 0:
        # session.execute(
        #     text("UPDATE documents SET status='READY' WHERE id=:doc_id AND status='PROCESSING'"),
        #     {"doc_id": document_id},
        # )
        session.execute(
            # FIX: changed to 'ready' and 'processing'
            text("UPDATE documents SET status='ready' WHERE id=:doc_id AND status='processing'"),
            {"doc_id": document_id},
        )
        session.commit()