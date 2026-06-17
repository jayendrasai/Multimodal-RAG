"""
app/api/v1/auth.py

Authentication endpoints.

POST /v1/auth/token    — login with username + password
POST /v1/auth/refresh  — exchange refresh token for new access token
POST /v1/auth/logout   — revoke current refresh token
POST /v1/auth/logout-all — revoke all refresh tokens (all devices)

Rate limiting:
  /auth/token and /auth/refresh: 10 req/min per IP (IP limiter, not user —
  user is not known yet). Prevents brute-force credential attacks.

Idempotency (/auth/token):
  Login is idempotent by design — submitting the same correct credentials
  twice issues two separate token pairs. This is correct behaviour for
  authentication. No idempotency key needed here.

  The idempotency pattern is applied to POST /documents/ingest and
  POST /v1/query (Day 4+) where duplicate requests would cause duplicate
  work or double-charges.
"""

from typing import Annotated

import redis.asyncio as aioredis
import structlog
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rate_limiter import get_ip_rate_limiter, get_rate_limiter
from app.dependencies import get_current_user, get_db_with_rls, get_redis
from app.db.session import get_db
from app.models.user import User
from app.schemas.auth import AccessTokenResponse, LoginRequest, RefreshRequest, TokenResponse
from app.services.auth_service import AuthService

logger = structlog.get_logger(__name__)
router = APIRouter()

# Auth endpoints use IP-based rate limiting (10/min) — stricter than general
# User is not authenticated yet so user_id is not available
_ip_limit = get_ip_rate_limiter(limit=10, window=60)


@router.post(
    "/token",
    response_model=TokenResponse,
    status_code=200,
    summary="Login",
    description="Exchange username and password for access and refresh tokens.",
)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    _rate: None = Depends(_ip_limit),
) -> TokenResponse:
    """
    Returns access_token (15min) and refresh_token (7 days).

    Security: same 401 response for wrong password AND non-existent user.
    Never reveal which condition triggered the error.
    bcrypt runs in both cases to prevent timing attacks.
    """
    service = AuthService(db=db, redis=redis)
    return await service.login(username=body.username, password=body.password)


@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
    status_code=200,
    summary="Refresh access token",
    description="Exchange a valid refresh token for a new access token.",
)
async def refresh_token(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
    _rate: None = Depends(_ip_limit),
) -> AccessTokenResponse:
    """
    Single-use refresh tokens. Each call rotates the token.
    Submitting an already-used refresh token returns 401.
    """
    service = AuthService(db=db, redis=redis)
    return await service.refresh(refresh_token=body.refresh_token)


@router.post(
    "/logout",
    status_code=204,
    summary="Logout",
    description="Revoke the current refresh token.",
)
async def logout(
    body: RefreshRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> None:
    """
    Revokes the refresh token provided in the body.
    Access token remains valid until its 15min TTL expires — this is
    acceptable for a short-lived token. For immediate revocation,
    use a token blacklist (adds Redis lookup per request — overkill here).
    """
    service = AuthService(db=db, redis=redis)
    await service.logout(
        user_id=str(current_user.id),
        refresh_token=body.refresh_token,
    )


@router.post(
    "/logout-all",
    status_code=204,
    summary="Logout all devices",
    description="Revoke all refresh tokens for the current user.",
)
async def logout_all(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> None:
    """Logs out from all devices by revoking every stored refresh token."""
    service = AuthService(db=db, redis=redis)
    await service.logout_all(user_id=str(current_user.id))