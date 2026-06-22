"""
app/schemas/audit.py
"""

import uuid
from datetime import datetime
from pydantic import BaseModel


class AuditLogItem(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID | None
    query_hash: str
    sources_used: list | None
    critic_result: str | None
    critic_corrections: int
    retry_count: int
    latency_ms: int | None
    created_at: datetime

    model_config = {"from_attributes": True}


class AuditLogResponse(BaseModel):
    logs: list[AuditLogItem]
    total: int
    limit: int
    offset: int