"""
app/services/ingestion_service.py

Business logic for document ingestion. The route handler stays thin —
all orchestration logic lives here.

ARCHITECTURAL UPDATE (Phase 2):
  This service now uses the Outbox Pattern. It no longer writes directly 
  to Qdrant or Elasticsearch. The KNOWN PHASE 1 LIMITATION (split-brain 
  inconsistency between Postgres/Qdrant/ES) is solved. 
  
  Ingestion and Deletion simply write intents (`UPSERT_CHUNK`, `DELETE_CHUNKS`) 
  to the `outbox` table and commit the transaction. The Celery relay worker 
  handles the actual external sync.
"""

import uuid
import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.core.file_validation import detect_file_type, validate_file_size
from app.models.document import Document
from app.models.outbox import Outbox
from app.config import get_settings

# Assuming you have or will extract these two helpers from your old ingest_pipeline:
# from app.parsers.document_parser import extract_text 
# from app.core.chunking import chunk_text

logger = structlog.get_logger(__name__)
settings = get_settings()


class IngestionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def ingest_document(
        self,
        user_id: uuid.UUID,
        filename: str,
        file_bytes: bytes,
        doc_metadata: dict | None = None,
    ) -> Document:
        """
        Full ingestion flow (Outbox Pattern):
          1. Validate size + magic-number type detection
          2. Extract text and chunk (CPU bound, fast)
          3. Create Document row with status='processing'
          4. Create Outbox rows for every chunk
          5. Commit ALL OF IT in a single atomic database transaction
        """
        validate_file_size(file_bytes)
        file_type = detect_file_type(file_bytes, filename)

        mime_map = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "eml": "message/rfc822",
        }

        # 1. Parse and Chunk inline (No network calls)
        # Note: Replace with your actual text extraction/chunking functions
        # raw_text = await extract_text(file_bytes, file_type)
        # chunks = chunk_text(raw_text, size=512, overlap=64)
        
        # Placeholder for compilation:
        chunks = ["chunk 1 text", "chunk 2 text"] 
        chunk_count = len(chunks)

        # 2. Prepare Document Record
        document_id = uuid.uuid4()
        document = Document(
            id=document_id,
            user_id=user_id,
            filename=filename,
            mime_type=mime_map[file_type],
            size_bytes=len(file_bytes),
            status="processing",  # Stays 'processing' until worker finishes
            chunk_count=chunk_count,
            embedding_model=settings.EMBEDDING_MODEL_VERSION,
            doc_metadata=doc_metadata,
        )
        self.db.add(document)

        logger.info(
            "ingestion_started",
            document_id=str(document.id),
            user_id=str(user_id),
            file_type=file_type,
            size_bytes=len(file_bytes),
            chunk_count=chunk_count,
        )

        # 3. Prepare Outbox Events
        for idx, chunk_text_data in enumerate(chunks):
            # Deterministic ID allows safe worker retries without duplicates
            chunk_id = uuid.uuid5(document.id, str(idx))
            
            outbox_event = Outbox(
                aggregate_type="chunk",
                aggregate_id=chunk_id,
                document_id=document.id,
                operation_type="UPSERT_CHUNK",
                payload={
                    "chunk_index": idx,
                    "text": chunk_text_data,
                    "metadata": doc_metadata or {},
                    "user_id": str(user_id), # required for tenancy checks
                },
            )
            self.db.add(outbox_event)

        # 4. Atomic Commit
        try:
            await self.db.commit()
            await self.db.refresh(document)
            
            logger.info(
                "ingestion_queued",
                document_id=str(document.id),
                outbox_events_created=chunk_count,
            )
        except Exception as e:
            await self.db.rollback()
            logger.error(
                "ingestion_failed_db_commit",
                document_id=str(document.id),
                error=str(e),
            )
            raise

        return document

    async def list_documents(self, user_id: uuid.UUID) -> list[Document]:
        result = await self.db.execute(
            select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_document(self, user_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        result = await self.db.execute(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
        )
        document = result.scalar_one_or_none()
        if document is None:
            raise NotFoundError("Document")
        return document

    async def delete_document(self, user_id: uuid.UUID, document_id: uuid.UUID) -> int:
        """
        Outbox Delete Flow:
        Instead of calling Qdrant/ES here, we mark the document as 'deleting'
        and queue a DELETE_CHUNKS outbox event. The worker handles the actual
        remote API calls safely.
        """
        document = await self.get_document(user_id, document_id)
        
        # If it's already deleting or deleted, do nothing
        if document.status in ("deleting", "deleted"):
            return document.chunk_count

        chunks_removed = document.chunk_count
        
        # 1. Mark status as 'deleting'
        document.status = "deleting"
        
        # 2. Queue the delete event for the worker
        delete_event = Outbox(
            aggregate_type="document",
            aggregate_id=document.id,
            document_id=document.id,
            operation_type="DELETE_CHUNKS",
            payload={
                "user_id": str(user_id),
            },
        )
        self.db.add(delete_event)

        # 3. Commit the intent
        await self.db.commit()

        logger.info(
            "document_delete_queued",
            document_id=str(document_id),
            user_id=str(user_id),
            chunks_to_remove=chunks_removed,
        )

        return chunks_removed