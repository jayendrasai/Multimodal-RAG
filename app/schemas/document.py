"""
app/schemas/document.py
"""

import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    document_id: uuid.UUID = Field(validation_alias="id", serialization_alias="document_id")
    filename: str
    chunk_count: int
    status: str
    embedding_model: str
    size_bytes: int

    model_config = {"from_attributes": True, "populate_by_name": True}


class DocumentListItem(BaseModel):
    id: uuid.UUID
    filename: str
    chunk_count: int
    size_bytes: int
    created_at: datetime
    doc_metadata: dict | None = None

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    total: int


class DocumentDeleteResponse(BaseModel):
    deleted: bool
    chunks_removed: int
    document_id: uuid.UUID