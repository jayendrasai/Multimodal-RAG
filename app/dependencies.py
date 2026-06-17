# """
# app/dependencies.py

# FastAPI shared dependencies injected via Depends().

# get_current_user: validates JWT, returns User — used by every authenticated route
# get_db: yields AsyncSession with RLS pre-set for the current user
# get_redis: yields Redis client
# """

# import uuid
# from typing import Annotated

# import redis.asyncio as aioredis
# import structlog
# from fastapi import Depends, Request
# from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# from sqlalchemy.ext.asyncio import AsyncSession

# from app.config import get_settings
# from app.core.exceptions import UnauthorizedError
# from app.core.security import decode_access_token
# from app.db.rls import set_rls_user_id
# from app.db.session import get_db as _get_db
# from app.models.user import User
# from sqlalchemy import select

# logger = structlog.get_logger(__name__)
# settings = get_settings()

# bearer_scheme = HTTPBearer(auto_error=False)

# # ── Redis singleton ────────────────────────────────────────────────────────

# _redis_client: aioredis.Redis | None = None


# def get_redis_client() -> aioredis.Redis:
#     global _redis_client
#     if _redis_client is None:
#         _redis_client = aioredis.from_url(
#             str(settings.REDIS_URL),
#             decode_responses=True,
#             max_connections=20,
#         )
#     return _redis_client


# async def get_redis() -> aioredis.Redis:
#     """Dependency: Redis client."""
#     return get_redis_client()


# # ── Auth dependency ────────────────────────────────────────────────────────

# async def get_current_user(
#     request: Request,
#     credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
#     db: AsyncSession = Depends(_get_db),
# ) -> User:
#     """
#     1. Extract Bearer token from Authorization header
#     2. Decode and validate JWT signature + expiry
#     3. Load user from DB — verifies user still exists and is active
#     4. Bind user_id to request state for audit middleware

#     Raises UnauthorizedError for any failure — never leaks reason to client.
#     """
#     if credentials is None:
#         raise UnauthorizedError()

#     token = credentials.credentials
#     payload = decode_access_token(token)  # raises InvalidTokenError on failure

#     user_id_str: str | None = payload.get("sub")
#     if not user_id_str:
#         raise UnauthorizedError()

#     try:
#         user_id = uuid.UUID(user_id_str)
#     except ValueError:
#         raise UnauthorizedError()

#     # Load user — confirms account still exists and is active
#     result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
#     user = result.scalar_one_or_none()

#     if user is None:
#         raise UnauthorizedError()

#     # Bind to request state — used by AuditContextMiddleware and audit service
#     request.state.user_id = user.id

#     return user


# async def get_db_with_rls(
#     current_user: Annotated[User, Depends(get_current_user)],
#     db: AsyncSession = Depends(_get_db),
# ) -> AsyncSession:
#     """
#     Authenticated DB session with RLS pre-set.
#     Use this instead of get_db() in all authenticated routes.

#     This ensures the DB-level user isolation policy fires before
#     any query executes. Forgetting to call set_rls_user_id manually
#     is the #1 cause of cross-user data leakage in multi-tenant apps.
#     """
#     await set_rls_user_id(db, current_user.id)
#     return db


# # ── Admin dependency ───────────────────────────────────────────────────────

# async def require_admin(
#     current_user: Annotated[User, Depends(get_current_user)],
# ) -> User:
#     """Restrict endpoint to admin users only."""
#     if not current_user.is_admin:
#         from app.core.exceptions import ForbiddenError
#         raise ForbiddenError()
#     return current_user
"""
app/dependencies.py

FastAPI shared dependencies injected via Depends().

get_current_user: validates JWT, returns User — used by every authenticated route
get_db: yields AsyncSession with RLS pre-set for the current user
get_redis: yields Redis client
"""

import uuid
from typing import Annotated

import redis.asyncio as aioredis
import structlog
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.exceptions import UnauthorizedError
from app.core.security import decode_access_token
from app.db.rls import set_rls_user_id
from app.db.session import get_db as _get_db
# Import all model modules before any mapper is used.
# SQLAlchemy resolves string-based relationship targets at mapper
# init time — if a referenced model hasn't been imported yet it
# raises "failed to locate name 'Document'" (or Session, etc.).
from app.models import document, session, memory, audit  # noqa: F401
from app.models.user import User
from sqlalchemy import select

logger = structlog.get_logger(__name__)
settings = get_settings()

bearer_scheme = HTTPBearer(auto_error=False)

# ── Redis singleton ────────────────────────────────────────────────────────

_redis_client: aioredis.Redis | None = None


def get_redis_client() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            str(settings.REDIS_URL),
            decode_responses=True,
            max_connections=20,
        )
    return _redis_client


async def get_redis() -> aioredis.Redis:
    """Dependency: Redis client."""
    return get_redis_client()


# ── Auth dependency ────────────────────────────────────────────────────────

async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: AsyncSession = Depends(_get_db),
) -> User:
    """
    1. Extract Bearer token from Authorization header
    2. Decode and validate JWT signature + expiry
    3. Load user from DB — verifies user still exists and is active
    4. Bind user_id to request state for audit middleware

    Raises UnauthorizedError for any failure — never leaks reason to client.
    """
    if credentials is None:
        raise UnauthorizedError()

    token = credentials.credentials
    payload = decode_access_token(token)  # raises InvalidTokenError on failure

    user_id_str: str | None = payload.get("sub")
    if not user_id_str:
        raise UnauthorizedError()

    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        raise UnauthorizedError()

    # Load user — confirms account still exists and is active
    result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
    user = result.scalar_one_or_none()

    if user is None:
        raise UnauthorizedError()

    # Bind to request state — used by AuditContextMiddleware and audit service
    request.state.user_id = user.id

    return user


async def get_db_with_rls(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(_get_db),
) -> AsyncSession:
    """
    Authenticated DB session with RLS pre-set.
    Use this instead of get_db() in all authenticated routes.

    This ensures the DB-level user isolation policy fires before
    any query executes. Forgetting to call set_rls_user_id manually
    is the #1 cause of cross-user data leakage in multi-tenant apps.
    """
    await set_rls_user_id(db, current_user.id)
    return db


# ── Admin dependency ───────────────────────────────────────────────────────

async def require_admin(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Restrict endpoint to admin users only."""
    if not current_user.is_admin:
        from app.core.exceptions import ForbiddenError
        raise ForbiddenError()
    return current_user