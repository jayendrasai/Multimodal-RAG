"""
app/core/security.py

JWT encoding/decoding and bcrypt password hashing.
Nothing else lives here — keep this module focused.

Token design:
- Access token: 15min expiry, contains user_id (sub) and token type
- Refresh token: 7 day expiry, stored in DB as hashed value
  (so stolen refresh tokens can be revoked without invalidating the secret)
"""

from datetime import datetime, timedelta, timezone
from typing import Any
import uuid
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import get_settings
from app.core.exceptions import InvalidTokenError

settings = get_settings()

# bcrypt with 12 rounds — strong enough, not so slow it blocks the event loop
# Never use MD5, SHA-1, SHA-256 for passwords. Only bcrypt, Argon2, or scrypt.
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


# ── Passwords ──────────────────────────────────────────────────────────────

def hash_password(plain_password: str) -> str:
    """Hash a plaintext password. Never store the plain value."""
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time comparison — safe against timing attacks."""
    return _pwd_context.verify(plain_password, hashed_password)


# ── JWT ────────────────────────────────────────────────────────────────────

def create_access_token(user_id: str) -> str:
    """
    Issue a short-lived access token.
    Payload: sub (user_id), type (access), exp, iat
    """
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": user_id,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """
    Issue a long-lived refresh token.
    Store the HASH of this in the DB, not the raw value.
    """
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": user_id,
        "type": "refresh",
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Decode and validate an access token.
    Raises InvalidTokenError for any failure — expired, tampered, wrong type.
    Never leak decode failure reason to the client.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError:
        raise InvalidTokenError()

    if payload.get("type") != "access":
        raise InvalidTokenError()

    return payload


def decode_refresh_token(token: str) -> dict[str, Any]:
    """Decode and validate a refresh token."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError:
        raise InvalidTokenError()

    if payload.get("type") != "refresh":
        raise InvalidTokenError()

    return payload