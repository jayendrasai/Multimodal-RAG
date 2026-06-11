"""
app/db/session.py

Async SQLAlchemy engine and session factory.

Connection pool is sized explicitly — never rely on SQLAlchemy defaults
for a production service. Pool timeout is set to fail fast rather than
queue requests behind a stuck connection.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings

settings = get_settings()

# NullPool is used in tests to avoid cross-test connection state.
# Production uses the default QueuePool via pool_size / max_overflow.
engine = create_async_engine(
    str(settings.DATABASE_URL),
    echo=settings.DEBUG,           # SQL query logging in dev only
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_pre_ping=True,            # Discard stale connections silently
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,        # Avoid lazy-load errors after commit
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency. Yields a session and guarantees close on exit.

    Usage:
        @router.post("/endpoint")
        async def handler(db: AsyncSession = Depends(get_db)):
            ...

    The session is NOT committed here. Services are responsible for
    explicit commits. This prevents partial writes on unhandled exceptions.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()