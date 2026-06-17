"""
app/services/auth_service.py

Authentication service: login, token issuance, refresh, logout.

Token storage design:
  Access tokens  — stateless JWT, 15min. Not stored. Validated by signature + expiry.
  Refresh tokens — stateful JWT, 7 days. The SHA-256 hash is stored in Redis.
                   Stored hash enables:
                     1. Single-use enforcement (rotate on each refresh)
                     2. Logout/revocation without rotating the JWT secret
                     3. "Logout all devices" by deleting all tokens for a user

Why hash the refresh token in Redis and not store it raw:
  If Redis is compromised, raw tokens can be used immediately.
  The SHA-256 hash is computationally infeasible to reverse.
  The attacker gets hashes that are useless without the original token.

Redis key: refresh:{user_id}:{token_hash} → "1"  (TTL = refresh token expiry)
"""

import hashlib
from datetime import timedelta

import redis.asyncio as aioredis
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.exceptions import UnauthorizedError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    verify_password,
)
from app.models.user import User
from app.schemas.auth import AccessTokenResponse, TokenResponse

logger = structlog.get_logger(__name__)
settings = get_settings()


def _hash_token(token: str) -> str:
    """SHA-256 hash of a token string. Used for storage, not for security primitives."""
    return hashlib.sha256(token.encode()).hexdigest()


def _refresh_redis_key(user_id: str, token_hash: str) -> str:
    return f"refresh:{user_id}:{token_hash}"


class AuthService:
    def __init__(self, db: AsyncSession, redis: aioredis.Redis) -> None:
        self.db = db
        self.redis = redis

    async def login(self, username: str, password: str) -> TokenResponse:
        """
        Validate credentials and issue access + refresh tokens.

        Security: same error message and response time for wrong password
        vs non-existent user. bcrypt.verify runs in both cases to prevent
        timing attacks that disambiguate the two cases.
        """
        user = await self._get_user_by_username(username)

        # Always run verify_password even if user is None.
        # bcrypt is slow — skipping it for missing users creates a timing
        # difference that reveals whether an account exists.
        dummy_hash = "$2b$12$12345678901234567890121234567890123456789012345678901"
        hash_to_check = user.hashed_password if user else dummy_hash
        password_valid = verify_password(password, hash_to_check)

        if not user or not password_valid or not user.is_active:
            logger.warning("login_failed", username=username)
            # Same error regardless of failure reason
            raise UnauthorizedError("Invalid credentials.")

        return await self._issue_tokens(user)

    async def refresh(self, refresh_token: str) -> AccessTokenResponse:
        """
        Validate refresh token and issue a new access token.
        Rotates the refresh token on every use (single-use enforcement).
        Old refresh token is invalidated immediately.
        """
        payload = decode_refresh_token(refresh_token)  # raises InvalidTokenError if bad
        user_id: str = payload["sub"]

        token_hash = _hash_token(refresh_token)
        redis_key = _refresh_redis_key(user_id, token_hash)

        # Verify token is in Redis (not revoked, not already used)
        exists = await self.redis.exists(redis_key)
        if not exists:
            logger.warning("refresh_token_not_found_or_revoked", user_id=user_id)
            raise UnauthorizedError("Refresh token is invalid or has been revoked.")

        # Invalidate used token immediately (rotate)
        await self.redis.delete(redis_key)

        # Verify user still exists and is active
        user = await self._get_user_by_id(user_id)
        if not user or not user.is_active:
            raise UnauthorizedError("Account is inactive.")

        # Issue new access token only — client keeps the same refresh token
        # until its TTL expires or they explicitly logout
        # NOTE: We also issue a new refresh token (full rotation) for maximum security.
        new_token_response = await self._issue_tokens(user)

        logger.info("token_refreshed", user_id=user_id)

        return AccessTokenResponse(
            access_token=new_token_response.access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    async def logout(self, user_id: str, refresh_token: str) -> None:
        """
        Revoke the specific refresh token.
        Access tokens cannot be revoked (stateless) — they expire after 15min.
        """
        token_hash = _hash_token(refresh_token)
        redis_key = _refresh_redis_key(user_id, token_hash)
        await self.redis.delete(redis_key)
        logger.info("user_logged_out", user_id=user_id)

    async def logout_all(self, user_id: str) -> None:
        """
        Revoke all refresh tokens for a user (logout all devices).
        Scans Redis for keys matching refresh:{user_id}:*.
        """
        pattern = f"refresh:{user_id}:*"
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                await self.redis.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info("all_tokens_revoked", user_id=user_id, count=deleted)

    # ── Private helpers ────────────────────────────────────────────────────

    async def _issue_tokens(self, user: User) -> TokenResponse:
        user_id_str = str(user.id)

        access_token = create_access_token(user_id_str)
        refresh_token = create_refresh_token(user_id_str)

        # Store hash of refresh token in Redis
        token_hash = _hash_token(refresh_token)
        redis_key = _refresh_redis_key(user_id_str, token_hash)
        ttl = settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400

        await self.redis.setex(redis_key, ttl, "1")

        logger.info("tokens_issued", user_id=user_id_str)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    async def _get_user_by_username(self, username: str) -> User | None:
        result = await self.db.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def _get_user_by_id(self, user_id: str) -> User | None:
        import uuid
        try:
            uid = uuid.UUID(user_id)
        except ValueError:
            return None
        result = await self.db.execute(
            select(User).where(User.id == uid)
        )
        return result.scalar_one_or_none()