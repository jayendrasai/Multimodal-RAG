"""
app/services/ingestion_service.py

Business logic for document ingestion. The route handler stays thin —
all orchestration logic lives here.

Idempotency: ingestion is NOT idempotent by document content in Phase 1.
Uploading the same PDF twice creates two separate Document rows with two
separate chunk sets. True content-based idempotency (hash the file,
reject/merge duplicates) is a reasonable Phase 2 addition but is not in
the Day 4 scope. If you need idempotent retries from a flaky client,
use the document_id-based idempotency key pattern shown in
PATCH /documents/{id}/retry (not yet implemented in this stub).

KNOWN PHASE 1 LIMITATION (write this in your risk register):
  If the pipeline fails after Qdrant write but before Elasticsearch
  write (or vice versa), the two stores can become inconsistent for
  that document. Phase 1 does not implement a saga/compensation pattern.
  Mitigation: document.status = "failed" makes this visible, and
  DELETE /documents/{id} cleans up both stores regardless of partial
  state (delete is idempotent — deleting from a store with no matching
  data is a no-op, not an error).
"""

import uuid

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.core.file_validation import detect_file_type, validate_file_size
from app.models.document import Document
from app.pipelines.ingest_pipeline import run_ingest_pipeline
from app.search.es_client import delete_document_chunks as es_delete
from app.vector_store.qdrant_client import delete_document_chunks as qdrant_delete
from app.config import get_settings

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
        Full ingestion flow:
          1. Validate size + magic-number type detection
          2. Create Document row with status=processing
          3. Run pipeline (parse → chunk → embed → store)
          4. Update status=ready + chunk_count, or status=failed on error
        """
        validate_file_size(file_bytes)
        file_type = detect_file_type(file_bytes, filename)

        mime_map = {
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "eml": "message/rfc822",
        }

        document = Document(
            id=uuid.uuid4(),
            user_id=user_id,
            filename=filename,
            mime_type=mime_map[file_type],
            size_bytes=len(file_bytes),
            status="processing",
            embedding_model=settings.EMBEDDING_MODEL_VERSION,
            doc_metadata=doc_metadata,
        )
        self.db.add(document)
        await self.db.commit()
        await self.db.refresh(document)

        logger.info(
            "ingestion_started",
            document_id=str(document.id),
            user_id=str(user_id),
            file_type=file_type,
            size_bytes=len(file_bytes),
        )

        try:
            chunk_count = await run_ingest_pipeline(
                user_id=user_id,
                document_id=document.id,
                file_type=file_type,
                file_bytes=file_bytes,
                filename=filename,
                doc_metadata=doc_metadata,
            )

            document.status = "ready"
            document.chunk_count = chunk_count
            await self.db.commit()
            await self.db.refresh(document)

            logger.info(
                "ingestion_complete",
                document_id=str(document.id),
                chunk_count=chunk_count,
            )

        except Exception as e:
            document.status = "failed"
            await self.db.commit()
            logger.error(
                "ingestion_failed",
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
        Deletes the document and all associated chunks from Qdrant and
        Elasticsearch. Idempotent — deleting an already-deleted document's
        vectors is a no-op in both stores, not an error.
        """
        document = await self.get_document(user_id, document_id)
        chunks_removed = document.chunk_count

        await qdrant_delete(user_id, document_id)
        await es_delete(user_id, document_id)

        await self.db.delete(document)
        await self.db.commit()

        logger.info(
            "document_deleted",
            document_id=str(document_id),
            user_id=str(user_id),
            chunks_removed=chunks_removed,
        )

        return chunks_removed