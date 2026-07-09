"""
app/api/v1/sessions.py

POST   /v1/sessions               — create a new session
GET    /v1/sessions               — list user's sessions
GET    /v1/sessions/{id}/history  — full turn history for a session
POST   /v1/sessions/{id}/end      — end session (Phase 2 will add summarization here)

No special rate limit beyond the general 60/min — sessions are cheap
operations (no LLM calls, no embedding) unlike ingestion or query.
"""

import uuid
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends , HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db_with_rls, get_redis
from app.models.user import User
from app.schemas.session import (
    CreateSessionRequest,
    CreateSessionResponse,
    EndSessionResponse,
    SessionHistoryResponse,
    SessionListResponse,
)
from app.services.session_service import SessionService
from app.core.exceptions import ConflictError, NotFoundError
from app.tasks.summarization import summarize_session

router = APIRouter()


@router.post(
    "",
    response_model=CreateSessionResponse,
    status_code=201,
    summary="Create a new session",
)
async def create_session(
    body: CreateSessionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db_with_rls),
    redis: aioredis.Redis = Depends(get_redis),
) -> CreateSessionResponse:
    """
    context_hint is accepted for API-contract stability but unused until
    Phase 2 (cross-session memory pre-loading). memory_loaded_count and
    working_memory_tokens are always 0 in Phase 1.
    """
    service = SessionService(db=db, redis=redis)
    session = await service.create_session(
        user_id=current_user.id,
        context_hint=body.context_hint,
    )
    return CreateSessionResponse(
        session_id=session.id,
        created_at=session.created_at,
        memory_loaded_count=0,
        working_memory_tokens=0,
    )


@router.get(
    "",
    response_model=SessionListResponse,
    summary="List the authenticated user's sessions",
)
async def list_sessions(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db_with_rls),
    redis: aioredis.Redis = Depends(get_redis),
) -> SessionListResponse:
    service = SessionService(db=db, redis=redis)
    sessions = await service.list_sessions(user_id=current_user.id)
    return SessionListResponse(sessions=sessions)


@router.get(
    "/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Get full turn history for a session",
)
async def get_session_history(
    session_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db_with_rls),
    redis: aioredis.Redis = Depends(get_redis),
) -> SessionHistoryResponse:
    """
    Returns 404 if the session doesn't exist OR belongs to another user —
    RLS makes another user's session row invisible, not merely forbidden.
    """
    service = SessionService(db=db, redis=redis)
    turns = await service.get_history(user_id=current_user.id, session_id=session_id)
    return SessionHistoryResponse(turns=turns, turn_count=len(turns))


@router.post(
    "/{session_id}/end",
    response_model=EndSessionResponse,
    summary="End a session",
)
async def end_session(
    session_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db_with_rls),
    redis: aioredis.Redis = Depends(get_redis),
) -> EndSessionResponse:
    """
    Ownership + atomic transition in one query. A non-owner's request matches 
    zero rows, identical response shape to "already ended".
    
    Idempotent: returns 202 whether it was just closed or already closed.
    """
    # 1. Attempt atomic update
    result = await db.execute(
        text("""
            UPDATE sessions
            SET status = 'summarization_pending', updated_at = now()
            WHERE id = :session_id AND user_id = :user_id AND status = 'active'
            RETURNING id
        """),
        {"session_id": str(session_id), "user_id": str(current_user.id)}
    )
    row = result.fetchone()

    # 2. Handle cases where the update missed (already ended, missing, or unauthorized)
    if row is None:
        existing = await db.execute(
            text("SELECT status FROM sessions WHERE id = :session_id AND user_id = :user_id"),
            {"session_id": str(session_id), "user_id": str(current_user.id)}
        )
        existing_row = existing.fetchone()
        
        if existing_row is None:
            # Either it doesn't exist, or it belongs to someone else
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="session not found"
            )
            
        # Already pending or summarized — do NOT re-enqueue.
        return {"session_id": str(session_id), "status": existing_row[0]}

    # Commit the transaction so the worker sees the updated status
    await db.commit()

    # 3. Enqueue the background task
    summarize_session.delay(str(session_id))
    
    #return {"session_id": str(session_id), "status": "summarization_pending"}
    return EndSessionResponse(
        session_id=session_id,
        status="summarization_pending",
        summary_id=None,
        topics_extracted=[],
        entities_extracted=0,
        memory_stored=False,
        unresolved_questions=[],
    )