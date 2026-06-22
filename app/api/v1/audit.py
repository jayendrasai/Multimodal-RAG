"""
app/api/v1/audit.py

GET /v1/audit/logs

Blueprint requirement: "Every query, retrieved chunks, and critic result
logged to Postgres. Compliance trail." A compliance trail nobody can
read is not a compliance trail -- this endpoint is what makes the audit
data Day 6 already writes actually usable.

Users see only their OWN audit logs (RLS-scoped, same pattern as every
other resource in this API). There is no admin-wide "see everyone's
queries" endpoint in Phase 1 -- that's a deliberate scope cut. If a
genuine compliance/admin review need arises, it should be a separate,
explicitly-audited admin capability, not a side effect of this endpoint.

No special rate limit -- this is a read of your own existing data, not
a new compute-heavy operation.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db_with_rls
from app.models.audit import AuditLog
from app.models.user import User
from app.schemas.audit import AuditLogResponse

router = APIRouter()


@router.get(
    "/logs",
    response_model=AuditLogResponse,
    summary="List the authenticated user's own audit log entries",
)
async def list_audit_logs(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db_with_rls),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> AuditLogResponse:
    """
    Returns the caller's own audit trail, newest first.
    RLS scopes this to current_user.id automatically -- the explicit
    filter below is defense-in-depth and keeps intent self-documenting,
    same pattern as every other service in this codebase.
    """
    count_result = await db.execute(
        select(func.count()).select_from(AuditLog).where(AuditLog.user_id == current_user.id)
    )
    total = count_result.scalar_one()

    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.user_id == current_user.id)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    logs = list(result.scalars().all())

    return AuditLogResponse(logs=logs, total=total, limit=limit, offset=offset)