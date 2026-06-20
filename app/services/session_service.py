"""
app/services/session_service.py

Session lifecycle: create, list, get history, end.

Design note on Postgres vs Redis roles:
  Postgres (sessions, session_turns) = durable record. Source of truth.
  Redis (session_cache) = working memory window for the query pipeline.
                            Disposable — can be rebuilt from Postgres at
                            any time via warm_cache_from_turns.

Phase 1 does NOT implement summarization (POST /sessions/{id}/end returns
empty topics/entities/unresolved_questions). That's Phase 2 — episodic
memory writing happens there. Phase 1's /end just marks ended_at and
clears the Redis working-memory cache; the session_turns rows remain in
Postgres permanently as the audit-trail-adjacent record.
"""

import uuid
from datetime import datetime, timezone

import redis.asyncio as aioredis
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.session_cache import clear_session_cache, get_recent_turns, warm_cache_from_turns
from app.core.exceptions import ConflictError, NotFoundError
from app.models.session import Session, SessionTurn

logger = structlog.get_logger(__name__)


class SessionService:
    def __init__(self, db: AsyncSession, redis: aioredis.Redis) -> None:
        self.db = db
        self.redis = redis

    async def create_session(self, user_id: uuid.UUID, context_hint: str | None = None) -> Session:
        """
        context_hint is accepted but unused in Phase 1 — Phase 2's memory
        service will use it to pre-load relevant episodic summaries.
        Accepting it now means the API contract is stable across phases.
        """
        session = Session(
            id=uuid.uuid4(),
            user_id=user_id,
            turn_count=0,
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)

        logger.info("session_created", session_id=str(session.id), user_id=str(user_id))
        return session

    async def list_sessions(self, user_id: uuid.UUID) -> list[Session]:
        result = await self.db.execute(
            select(Session).where(Session.user_id == user_id).order_by(Session.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_session(self, user_id: uuid.UUID, session_id: uuid.UUID) -> Session:
        """
        RLS already scopes this to the caller's own rows at the DB layer,
        but the explicit user_id filter here is defense-in-depth and also
        makes the query's intent self-documenting.
        """
        result = await self.db.execute(
            select(Session).where(
                Session.id == session_id,
                Session.user_id == user_id,
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise NotFoundError("Session")
        return session

    async def get_history(self, user_id: uuid.UUID, session_id: uuid.UUID) -> list[SessionTurn]:
        # Confirms ownership and existence — raises NotFoundError otherwise
        await self.get_session(user_id, session_id)

        result = await self.db.execute(
            select(SessionTurn)
            .where(SessionTurn.session_id == session_id)
            .order_by(SessionTurn.created_at.asc())
        )
        return list(result.scalars().all())

    async def add_turn(
        self,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        role: str,
        content: str,
        sources: list | None = None,
        critic_passed: bool | None = None,
        latency_ms: int | None = None,
    ) -> SessionTurn:
        """
        Writes the turn to Postgres (durable) AND pushes it into the Redis
        working-memory cache (fast access for the query pipeline).

        Called by the query pipeline (Day 6+) after each user message and
        each assistant response — two calls per query/answer pair.
        """
        session = await self.get_session(user_id, session_id)

        if session.ended_at is not None:
            raise ConflictError("Cannot add a turn to an ended session.")

        turn = SessionTurn(
            id=uuid.uuid4(),
            session_id=session_id,
            user_id=user_id,
            role=role,
            content=content,
            sources=sources,
            critic_passed=critic_passed,
            latency_ms=latency_ms,
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(turn)

        session.turn_count += 1
        await self.db.commit()
        await self.db.refresh(turn)

        # Update the Redis working-memory window — best-effort, not
        # transactional with the Postgres write. If Redis is briefly down,
        # the durable record still succeeds; working memory just falls
        # back to a Postgres read on next access (see get_working_memory).
        try:
            from app.cache.session_cache import push_turn
            await push_turn(self.redis, session_id, role, content)
        except Exception as e:
            logger.warning("session_cache_push_failed", session_id=str(session_id), error=str(e))

        return turn

    async def get_working_memory(self, user_id: uuid.UUID, session_id: uuid.UUID) -> list[dict]:
        """
        Returns the last N turns for context injection into the query
        pipeline. Tries Redis first (fast path). If the cache is cold
        (expired, or Redis was down when turns were written), falls back
        to Postgres and warms the cache for next time.
        """
        await self.get_session(user_id, session_id)  # ownership check

        cached = await get_recent_turns(self.redis, session_id)
        if cached:
            return cached

        # Cache miss — rebuild from Postgres
        turns = await self.get_history(user_id, session_id)
        turn_dicts = [
            {
                "role": t.role,
                "content": t.content,
                "timestamp": t.created_at.isoformat(),
            }
            for t in turns
        ]
        await warm_cache_from_turns(self.redis, session_id, turn_dicts)

        from app.config import get_settings
        settings = get_settings()
        return turn_dicts[-settings.CONTEXT_WINDOW_TURNS:]

    async def end_session(self, user_id: uuid.UUID, session_id: uuid.UUID) -> Session:
        """
        Marks the session ended and clears its Redis working-memory cache.

        Phase 1: no summarization. summary_id remains None.
        Phase 2 will insert an LLM summarization call here, writing to
        episodic_memory and setting session.summary_id before returning.
        """
        session = await self.get_session(user_id, session_id)

        if session.ended_at is not None:
            raise ConflictError("Session has already ended.")

        session.ended_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(session)

        await clear_session_cache(self.redis, session_id)

        logger.info(
            "session_ended",
            session_id=str(session_id),
            user_id=str(user_id),
            turn_count=session.turn_count,
        )

        return session