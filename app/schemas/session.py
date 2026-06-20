"""
app/schemas/session.py
"""

import uuid
from datetime import datetime
from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    context_hint: str | None = Field(
        None, max_length=500, description="Optional topic to pre-load relevant memory."
    )


class CreateSessionResponse(BaseModel):
    session_id: uuid.UUID
    created_at: datetime
    # Phase 1: always 0 — cross-session memory loading is Phase 2.
    # Field is in the response now so the API contract doesn't change later.
    memory_loaded_count: int = 0
    working_memory_tokens: int = 0


class SessionListItem(BaseModel):
    id: uuid.UUID
    created_at: datetime
    ended_at: datetime | None
    turn_count: int
    summary_id: uuid.UUID | None

    model_config = {"from_attributes": True}


class SessionListResponse(BaseModel):
    sessions: list[SessionListItem]


class TurnItem(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(validation_alias="created_at")
    sources: list | None = None
    critic_passed: bool | None = None

    model_config = {"from_attributes": True, "populate_by_name": True}


class SessionHistoryResponse(BaseModel):
    turns: list[TurnItem]
    turn_count: int


class EndSessionResponse(BaseModel):
    summary_id: uuid.UUID | None
    # Phase 1: summarization is not implemented yet (Phase 2 feature).
    # These fields are always empty/zero now but present in the contract
    # so Phase 2 doesn't require a breaking API change.
    topics_extracted: list[str] = Field(default_factory=list)
    entities_extracted: int = 0
    memory_stored: bool = False
    unresolved_questions: list[str] = Field(default_factory=list)