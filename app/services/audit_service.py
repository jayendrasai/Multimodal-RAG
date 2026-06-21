"""
app/services/audit_service.py

Writes the compliance audit record for every query, per blueprint spec:
"Every query, retrieved chunks, and critic result logged to Postgres."

query_hash, not raw query text, is stored — see models/audit.py docstring.
If your compliance requirements need the raw query text, that's a
deliberate schema change to make with encryption-at-rest in place, not
a default. Storing raw user queries indefinitely is itself a privacy
liability when those queries may reference sensitive document content.
"""

import hashlib
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit import AuditLog

logger = structlog.get_logger(__name__)


class AuditService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def log_query(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID | None,
        query: str,
        sources_used: list[dict],
        critic_result: str | None,
        critic_corrections: int,
        retry_count: int,
        latency_ms: int,
    ) -> AuditLog:
        query_hash = hashlib.sha256(query.encode()).hexdigest()

        # Store only chunk_id + document_id in the audit trail, not full
        # chunk text — the audit log proves WHICH sources were used for
        # compliance review, not a duplicate copy of the document content.
        sources_summary = [
            {"chunk_id": c.get("chunk_id"), "document_id": c.get("document_id")}
            for c in sources_used
        ]

        record = AuditLog(
            id=uuid.uuid4(),
            user_id=user_id,
            session_id=session_id,
            query_hash=query_hash,
            sources_used=sources_summary,
            critic_result=critic_result,
            critic_corrections=critic_corrections,
            retry_count=retry_count,
            latency_ms=latency_ms,
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(record)
        await self.db.commit()

        logger.info(
            "audit_logged",
            user_id=str(user_id),
            critic_result=critic_result,
            retry_count=retry_count,
            latency_ms=latency_ms,
        )

        return record