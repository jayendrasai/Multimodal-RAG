"""
app/cache/session_cache.py

Working memory: the last N turns of a session, held in Redis for fast
access during an active conversation. This is NOT the persistent record —
Postgres session_turns table is the durable store (written by
session_service.py on every turn). Redis here is purely a low-latency
cache so the query pipeline doesn't hit Postgres for context on every
single query.

Data structure: Redis LIST, one per session.
  Key: session_turns:{session_id}
  Each list element: JSON-encoded {role, content, timestamp}
  Capped at CONTEXT_WINDOW_TURNS (6) via LTRIM after every push —
  older turns are evicted from the cache but remain in Postgres.

TTL: matches REDIS_SESSION_TTL_SECONDS (1 hour of inactivity expires the
cache entry — the session itself doesn't end, but its cached working
memory is rebuilt from Postgres on next access if the user returns later
within the same session).
"""

import json
import uuid
from datetime import datetime, timezone

import redis.asyncio as aioredis
import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


def _session_key(session_id: uuid.UUID | str) -> str:
    return f"session_turns:{session_id}"


async def push_turn(
    redis: aioredis.Redis,
    session_id: uuid.UUID | str,
    role: str,
    content: str,
) -> None:
    """
    Append a turn to the working memory cache.
    Trims to CONTEXT_WINDOW_TURNS immediately — the list never grows
    unbounded even under heavy use within a single session.
    """
    key = _session_key(session_id)
    turn = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    pipe = redis.pipeline()
    pipe.rpush(key, json.dumps(turn))
    pipe.ltrim(key, -settings.CONTEXT_WINDOW_TURNS, -1)
    pipe.expire(key, settings.REDIS_SESSION_TTL_SECONDS)
    await pipe.execute()

    logger.debug("turn_pushed_to_cache", session_id=str(session_id), role=role)


async def get_recent_turns(
    redis: aioredis.Redis,
    session_id: uuid.UUID | str,
) -> list[dict]:
    """
    Returns up to CONTEXT_WINDOW_TURNS most recent turns, oldest first.
    Returns an empty list if the cache is cold (expired or session is new) —
    caller (session_service) is responsible for falling back to Postgres
    if it needs full history beyond what's cached.
    """
    key = _session_key(session_id)
    raw_turns = await redis.lrange(key, 0, -1)
    return [json.loads(t) for t in raw_turns]


async def clear_session_cache(redis: aioredis.Redis, session_id: uuid.UUID | str) -> None:
    """Called when a session ends — working memory is no longer needed."""
    key = _session_key(session_id)
    await redis.delete(key)
    logger.debug("session_cache_cleared", session_id=str(session_id))


async def warm_cache_from_turns(
    redis: aioredis.Redis,
    session_id: uuid.UUID | str,
    turns: list[dict],
) -> None:
    """
    Rebuild the Redis cache from a list of turns loaded from Postgres.
    Used when a session is resumed and the cache has expired but the
    session hasn't formally ended — keeps working memory consistent
    with durable storage without forcing the caller to query Postgres
    on every subsequent turn.
    """
    key = _session_key(session_id)
    recent = turns[-settings.CONTEXT_WINDOW_TURNS:]

    if not recent:
        return

    pipe = redis.pipeline()
    pipe.delete(key)
    for turn in recent:
        pipe.rpush(key, json.dumps(turn))
    pipe.expire(key, settings.REDIS_SESSION_TTL_SECONDS)
    await pipe.execute()

    logger.debug("session_cache_warmed", session_id=str(session_id), turn_count=len(recent))