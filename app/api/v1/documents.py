"""
app/api/v1/documents.py

POST   /v1/documents/ingest          — upload and process a document
GET    /v1/documents                 — list all documents for the user
DELETE /v1/documents/{document_id}   — queue document and its chunks for deletion

Rate limiting: ingestion is CPU/embedding-cost heavy. Limited to 10/min
per user — much stricter than the general 60/min API limit.

All routes use get_db_with_rls — RLS is enforced before any query runs.
"""

import json
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationError
from app.core.rate_limiter import get_rate_limiter
from app.dependencies import get_current_user, get_db_with_rls
from app.models.user import User
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.services.ingestion_service import IngestionService

router = APIRouter()

_ingest_limit = get_rate_limiter(limit=10, window=60)


@router.post(
    "/ingest",
    response_model=DocumentResponse,
    status_code=202,  # CHANGED: 202 Accepted because processing happens asynchronously
    summary="Upload and queue a document for processing",
)
async def ingest_document(
    current_user: Annotated[User, Depends(get_current_user)],
    file: UploadFile = File(..., description="PDF, DOCX, or .eml. Max 50MB."),
    metadata: str | None = Form(None, description='Optional JSON: {"source":..., "date":..., "category":...}'),
    db: AsyncSession = Depends(get_db_with_rls),
    _rate: None = Depends(_ingest_limit),
) -> DocumentResponse:
    """
    Parses the uploaded file, chunks it, and queues it in the Outbox.
    The Celery relay worker handles the actual embedding and syncing to
    Qdrant and Elasticsearch asynchronously.
    """
    doc_metadata = None
    if metadata:
        try:
            doc_metadata = json.loads(metadata)
            if not isinstance(doc_metadata, dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            raise ValidationError("metadata must be a valid JSON object.", field="metadata")

    file_bytes = await file.read()

    service = IngestionService(db=db)
    
    # This now commits to Postgres and returns immediately.
    # The returned document will have status="PROCESSING"
    document = await service.ingest_document(
        user_id=current_user.id,
        filename=file.filename or "unnamed",
        file_bytes=file_bytes,
        doc_metadata=doc_metadata,
    )

    return DocumentResponse.model_validate(document)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents for the authenticated user",
)
async def list_documents(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db_with_rls),
) -> DocumentListResponse:
    service = IngestionService(db=db)
    documents = await service.list_documents(user_id=current_user.id)
    return DocumentListResponse(
        documents=[doc for doc in documents],
        total=len(documents),
    )


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=202,  # CHANGED: 202 Accepted, delete happens via Outbox worker
    summary="Queue a document and all its vector chunks for deletion",
)
async def delete_document(
    document_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db_with_rls),
) -> DocumentDeleteResponse:
    """
    Marks the document as DELETING and queues a DELETE_CHUNKS outbox event.
    The worker handles the external teardown in Qdrant and ES.
    """
    service = IngestionService(db=db)
    
    # Handles 404s and queuing the Outbox event
    chunks_removed = await service.delete_document(
        user_id=current_user.id,
        document_id=document_id,
    )
    
    return DocumentDeleteResponse(
        deleted=True,
        chunks_removed=chunks_removed,
        document_id=document_id,
    )