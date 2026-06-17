"""
tests/conftest.py

Pytest fixtures shared across all test modules.

Test DB strategy:
  Uses a separate test database (ragdb_test) to avoid polluting dev data.
  Each test function gets a clean DB state via transaction rollback.
  No data from one test leaks into another.

Test Redis strategy:
  Uses Redis DB 1 (dev uses DB 0) to isolate test rate limiter state.
"""

import asyncio
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import get_settings
from app.core.security import hash_password
from app.dependencies import get_redis
from app.db.session import get_db
from app.main import create_app
from app.models.base import Base
from app.models.user import User

settings = get_settings()

# Test DB — separate database, NullPool to prevent cross-test state
TEST_DB_URL = str(settings.DATABASE_URL).replace("/ragdb", "/ragdb_test")

test_engine = create_async_engine(TEST_DB_URL, poolclass=NullPool, echo=False)
TestSessionLocal = async_sessionmaker(
    bind=test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_db():
    """Create all tables once per test session. Drop after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()

@pytest_asyncio.fixture(autouse=True)
async def clear_database(db):
    """Automatically clear all data between tests to prevent IntegrityErrors."""
    # Let the test run
    yield 
    
    # After the test, delete everything from all tables
    for table in reversed(Base.metadata.sorted_tables):
        await db.execute(table.delete())
    await db.commit()

@pytest_asyncio.fixture
async def db():
    """Per-test DB session. Rolls back after each test — no state leakage."""
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def redis_client():
    """Test Redis client using DB 1 (isolated from dev DB 0)."""
    import redis.asyncio as aioredis
    test_redis_url = str(settings.REDIS_URL).rstrip("/0") + "/1"
    client = aioredis.from_url(test_redis_url, decode_responses=True)
    yield client
    # Flush test DB after each test
    await client.flushdb()
    await client.aclose()


@pytest_asyncio.fixture
async def client(db, redis_client):
    """
    AsyncClient with dependency overrides for test DB and Redis.
    Injects test fixtures so the app uses isolated infra during tests.
    """
    app = create_app()

    async def override_get_db():
        yield db

    async def override_get_redis():
        return redis_client

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis] = override_get_redis

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def test_user(db):
    """Create a test user and return credentials."""
    user = User(
        username="testuser",
        hashed_password=hash_password("Testpassword123!"),
        is_active=True,
        is_admin=False,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {"username": "testuser", "password": "Testpassword123!", "id": user.id}


@pytest_asyncio.fixture
async def inactive_test_user(db):
    user = User(
        username="inactiveuser",
        hashed_password=hash_password("Testpassword123!"),
        is_active=False,
        is_admin=False,
    )
    db.add(user)
    await db.commit()
    return {"username": "inactiveuser", "password": "Testpassword123!"}


@pytest_asyncio.fixture
async def admin_user(db):
    user = User(
        username="adminuser",
        hashed_password=hash_password("Adminpassword123!"),
        is_active=True,
        is_admin=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {"username": "adminuser", "password": "Adminpassword123!", "id": user.id}


@pytest_asyncio.fixture
async def auth_headers(client, test_user):
    """Return Authorization header dict for an authenticated test user."""
    resp = await client.post("/v1/auth/token", json={
        "username": test_user["username"],
        "password": test_user["password"],
    })
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}