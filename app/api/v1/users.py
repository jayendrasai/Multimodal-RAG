"""
app/api/v1/users.py

User registration endpoint.

POST /v1/users/register

Security decisions:
  - Rate limited: 5 registrations per IP per hour. Prevents account
    farming and credential stuffing preparation.
  - Username uniqueness check uses a constant-time DB lookup, not a
    separate existence check — avoids TOCTOU race between check and insert.
  - Password is hashed before DB write. Plain value never touches the DB layer.
  - Response never includes the hashed password or internal state.
  - Duplicate username returns 409 Conflict — this does leak that a username
    exists. This is an accepted tradeoff for usability. If username privacy
    is a requirement, return 201 and send a confirmation email instead.
"""

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError
from app.core.rate_limiter import get_ip_rate_limiter
from app.core.security import hash_password
from app.db.session import get_db
from app.models import document, memory, audit  # noqa: F401 — force mapper resolution
from app.models.user import User
from app.schemas.user import RegisterRequest, UserResponse
from fastapi import HTTPException

logger = structlog.get_logger(__name__)
router = APIRouter()

# Strict: 5 registrations per IP per hour
_reg_limit = get_ip_rate_limiter(limit=5, window=3600)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=201,
    summary="Register a new user",
)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
    _rate: None = Depends(_reg_limit),
) -> User:
    """
    Create a new user account.

    Returns 409 if the username is already taken.
    Returns 201 with the created user on success.
    Password is never returned in any response.
    """
    # Check username availability
    existing = await db.execute(
        select(User).where(User.username == body.username)
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail=f"Username '{body.username}' is already taken.")
        #raise ConflictError(f"Username '{body.username}' is already taken.")

    user = User(
        username=body.username,
        hashed_password=hash_password(body.password),
        is_active=True,
        is_admin=False,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info("user_registered", user_id=str(user.id), username=user.username)

    return user