"""
app/db/rls.py

Row-Level Security enforcement.

Every authenticated database operation MUST call set_rls_user_id()
before querying tables that have RLS policies (episodic_memory,
semantic_memory, audit_logs, documents, sessions).

Defense-in-depth model:
  Layer 1 — ORM filter: every query includes WHERE user_id = :user_id
  Layer 2 — Postgres RLS: database rejects rows that don't match
             current_setting('app.current_user_id') even if the ORM
             filter is accidentally omitted

If Layer 1 fails (a developer forgets the filter), Layer 2 catches it.
If only Layer 1 exists, a single missed filter = data breach.

Usage pattern:
    async with get_db() as db:
        await set_rls_user_id(db, current_user.id)
        result = await db.execute(select(Document).where(...))

Never call this with an untrusted or unvalidated user_id.
The user_id must come from a verified JWT — see app/core/security.py.
"""

import uuid
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

import structlog

logger = structlog.get_logger(__name__)


async def set_rls_user_id(db: AsyncSession, user_id: uuid.UUID) -> None:
    """
    Set the Postgres session variable that RLS policies read.

    Uses SET LOCAL (transaction-scoped), not SET SESSION.
    SET SESSION would persist across connection pool reuse — catastrophic
    if connection is returned to pool mid-request and reused by another user.
    SET LOCAL is automatically cleared when the transaction ends.

    This MUST be called at the start of every authenticated DB operation.
    """
    # Validate it's actually a UUID — never interpolate raw strings
    user_id_str = str(uuid.UUID(str(user_id)))  # raises ValueError if malformed

    await db.execute(
        text("SELECT set_config('app.current_user_id', :user_id, true)"),
        {"user_id": user_id_str},
    )

    logger.debug("rls_user_id_set", user_id=user_id_str)