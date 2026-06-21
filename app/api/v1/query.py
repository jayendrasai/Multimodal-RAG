"""
app/api/v1/query.py

POST /v1/query

Rate limited at the general 60/min - this is the endpoint everything
else exists to support, so it gets the standard limit, not a special
stricter one like ingestion (10/min, compute-heavy) or auth (10/min,
brute-force concern).

Session integration: if session_id is provided, working memory (last
6 turns) is loaded for context, and both the user query and the
assistant answer are written back as turns after the pipeline completes.
If session_id is omitted, the query runs stateless with no history -
useful for one-off queries that don't need conversation continuity.
"""

from typing import Annotated

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limiter import get_rate_limiter
from app.dependencies import get_current_user, get_db_with_rls, get_redis
from app.models.user import User
from app.pipelines.query_pipeline import run_query_pipeline
from app.schemas.query import QueryRequest, QueryResponse, SourceItem
from app.services.audit_service import AuditService
from app.services.session_service import SessionService

router = APIRouter()

_query_limit = get_rate_limiter(limit=60, window=60)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question against the user's ingested documents",
)
async def query(
    body: QueryRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db_with_rls),
    redis: aioredis.Redis = Depends(get_redis),
    _rate: None = Depends(_query_limit),
) -> QueryResponse:
    """
    Pipeline: hybrid retrieve (BM25 + dense, RRF) -> rerank -> generate
    -> critic check -> retry on FAIL (bounded) -> respond.

    Every call is audit-logged regardless of outcome - query hash,
    sources used, critic result, retry count, latency. This is not
    optional and does not depend on session_id being provided.
    """
    session_service = SessionService(db=db, redis=redis)

    history: list[dict] = []
    if body.session_id is not None:
        history = await session_service.get_working_memory(
            user_id=current_user.id,
            session_id=body.session_id,
        )

    result = await run_query_pipeline(
        user_id=current_user.id,
        query=body.query,
        history=history,
    )

    audit_service = AuditService(db=db)
    await audit_service.log_query(
        user_id=current_user.id,
        session_id=body.session_id,
        query=body.query,
        sources_used=result["sources"],
        critic_result=result["critic_result"],
        critic_corrections=result["critic_corrections"],
        retry_count=result["retry_count"],
        latency_ms=result["latency_ms"],
    )

    if body.session_id is not None:
        await session_service.add_turn(
            user_id=current_user.id,
            session_id=body.session_id,
            role="user",
            content=body.query,
        )
        await session_service.add_turn(
            user_id=current_user.id,
            session_id=body.session_id,
            role="assistant",
            content=result["answer"],
            sources=[{"chunk_id": s.get("chunk_id")} for s in result["sources"]],
            critic_passed=(result["critic_result"] == "PASS"),
            latency_ms=result["latency_ms"],
        )

    return QueryResponse(
        answer=result["answer"],
        sources=[
            SourceItem(
                chunk_id=s.get("chunk_id"),
                document_id=s.get("document_id"),
                filename=s.get("filename"),
                rerank_score=s.get("rerank_score"),
            )
            for s in result["sources"]
        ],
        critic_result=result["critic_result"],
        retry_count=result["retry_count"],
        latency_ms=result["latency_ms"],
        session_id=body.session_id,
    )