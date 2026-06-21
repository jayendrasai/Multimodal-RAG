"""
app/schemas/query.py
"""

import uuid
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: uuid.UUID | None = Field(
        None,
        description="If provided, conversation history is loaded and this turn is appended.",
    )


class SourceItem(BaseModel):
    chunk_id: str
    document_id: str | None
    filename: str | None
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    critic_result: str
    retry_count: int
    latency_ms: int
    session_id: uuid.UUID | None