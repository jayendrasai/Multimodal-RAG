"""
app/core/rate_limiter.py

Redis-backed sliding window rate limiter.

Algorithm: sliding window counter using a sorted set (ZSET) per key.
Each request adds a timestamped member. Requests outside the window
are pruned. If the set size exceeds the limit, the request is rejected.

Why sliding window over fixed window:
  Fixed window allows 2x the limit at boundary (end of window N + start
  of window N+1). Sliding window distributes requests evenly. For an
  auth brute-force endpoint, 2x bursting is unacceptable.

Two rate limiters are provided:
  RateLimiter        — per-user, keyed by user_id (authenticated routes)
  IPRateLimiter      — per-IP, keyed by client IP (unauthenticated routes
                       like /auth/token — before user_id is known)

Usage as a FastAPI dependency:
  from app.core.rate_limiter import get_rate_limiter, get_ip_rate_limiter

  @router.post("/query")
  async def query(
      _: None = Depends(get_rate_limiter(limit=60, window=60)),
      current_user: User = Depends(get_current_user),
  ):
      ...
"""

import time
import uuid
from typing import Callable

import redis.asyncio as aioredis
import structlog
from fastapi import Depends, Request

from app.config import get_settings
from app.core.exceptions import RateLimitExceededError
from app.dependencies import get_redis

logger = structlog.get_logger(__name__)
settings = get_settings()


async def _sliding_window_check(
    redis: aioredis.Redis,
    key: str,
    limit: int,
    window_seconds: int,
) -> tuple[bool, int]:
    """
    Atomic sliding window check using a Redis ZSET + pipeline.

    Returns:
        (allowed: bool, retry_after: int)  — retry_after in seconds if denied

    Key structure: rate:<key>
    ZSET members: unique IDs (UUIDs), scores: Unix timestamps in milliseconds.

    The pipeline is atomic via multi-exec — no race condition between
    the COUNT check and the ZADD.
    """
    now_ms = int(time.time() * 1000)
    window_ms = window_seconds * 1000
    cutoff_ms = now_ms - window_ms
    redis_key = f"rate:{key}"

    pipe = redis.pipeline()
    pipe.zremrangebyscore(redis_key, "-inf", cutoff_ms)  # prune expired
    pipe.zcard(redis_key)                                 # count in window
    pipe.zadd(redis_key, {str(uuid.uuid4()): now_ms})    # add this request
    pipe.expire(redis_key, window_seconds + 1)            # TTL cleanup

    results = await pipe.execute()
    current_count = results[1]  # count BEFORE adding this request

    if current_count >= limit:
        # Undo the zadd — request denied, don't count it
        await redis.zpopmax(redis_key)
        oldest_ms = await redis.zscore(redis_key, (await redis.zrange(redis_key, 0, 0))[0]) if current_count > 0 else now_ms
        retry_after = max(1, int((oldest_ms + window_ms - now_ms) / 1000)) if oldest_ms else window_seconds
        return False, retry_after

    return True, 0


def get_rate_limiter(
    limit: int | None = None,
    window: int = 60,
) -> Callable:
    """
    Factory for per-user rate limiter dependency.
    Keyed by user_id from request.state (set by get_current_user).
    Falls back to settings.RATE_LIMIT_REQUESTS_PER_MINUTE if limit is None.

    Usage:
        @router.post("/query")
        async def query(
            _: None = Depends(get_rate_limiter()),  # 60/min default
        ):

        @router.post("/expensive")
        async def expensive(
            _: None = Depends(get_rate_limiter(limit=10, window=60)),
        ):
    """
    effective_limit = limit or settings.RATE_LIMIT_REQUESTS_PER_MINUTE

    async def _check(
        request: Request,
        redis: aioredis.Redis = Depends(get_redis),
    ) -> None:
        user_id = getattr(request.state, "user_id", None)
        if user_id is None:
            # Should never hit this on authenticated routes.
            # If it does, the auth middleware failed — reject to be safe.
            raise RateLimitExceededError()

        key = f"user:{user_id}"
        allowed, retry_after = await _sliding_window_check(
            redis, key, effective_limit, window
        )

        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                user_id=str(user_id),
                limit=effective_limit,
                window=window,
            )
            raise RateLimitExceededError(retry_after_seconds=retry_after)

    return _check


def get_ip_rate_limiter(
    limit: int | None = None,
    window: int = 60,
) -> Callable:
    """
    Factory for per-IP rate limiter dependency.
    Used on unauthenticated endpoints (/auth/token, /auth/refresh)
    where user_id is not yet known.

    Defaults to settings.RATE_LIMIT_AUTH_PER_MINUTE (10/min) which is
    stricter than the general limit to deter brute-force attacks.
    """
    effective_limit = limit or settings.RATE_LIMIT_AUTH_PER_MINUTE

    async def _check(
        request: Request,
        redis: aioredis.Redis = Depends(get_redis),
    ) -> None:
        # Prefer X-Forwarded-For (set by load balancer) over direct client IP.
        # In production, ensure the load balancer sets this header and the app
        # trusts only one proxy hop — otherwise IP spoofing is trivial.
        forwarded_for = request.headers.get("X-Forwarded-For")
        ip = forwarded_for.split(",")[0].strip() if forwarded_for else (request.client.host if request.client else "unknown")

        key = f"ip:{ip}"
        allowed, retry_after = await _sliding_window_check(
            redis, key, effective_limit, window
        )

        if not allowed:
            logger.warning(
                "ip_rate_limit_exceeded",
                ip=ip,
                limit=effective_limit,
                window=window,
            )
            raise RateLimitExceededError(retry_after_seconds=retry_after)

    return _check