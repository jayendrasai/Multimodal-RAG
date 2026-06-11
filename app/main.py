"""
app/main.py

FastAPI application factory.

Startup order matters:
1. Logging — must be first so every subsequent step is observable
2. Config validation — fail fast if env vars are missing or wrong
3. DB connection — verify Postgres is reachable
4. Redis connection — verify Redis is reachable
5. Qdrant connection — verify vector store is reachable
6. Elasticsearch connection — verify BM25 index is reachable
7. Middleware — CORS, request ID, audit context
8. Exception handlers — must be registered before routes
9. Routers — API surface last

On shutdown, connections are closed cleanly to avoid connection pool leaks.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.error_handlers import register_exception_handlers
from app.core.logging import setup_logging
from app.core.middleware import register_middleware
from app.db.session import engine

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager — replaces deprecated on_startup / on_shutdown events.
    Runs startup code before yield, shutdown code after.
    If any startup step fails, the app refuses to start (fail fast).
    """
    settings = get_settings()

    # ── 1. Logging ────────────────────────────────────────────────────────
    setup_logging()
    logger.info("startup_begin", env=settings.APP_ENV, version=settings.APP_VERSION)

    # ── 2. Config guard ───────────────────────────────────────────────────
    # Pydantic validators in Settings already ran at import time.
    # This is a double-check log — confirms config loaded successfully.
    logger.info("config_loaded", llm_model=settings.LLM_MODEL)

    # ── 3. Database ───────────────────────────────────────────────────────
    try:
        async with engine.connect() as conn:
            from sqlalchemy import text
            await conn.execute(text("SELECT 1"))
        logger.info("db_connected")
    except Exception as e:
        logger.error("db_connection_failed", error=str(e))
        raise RuntimeError("Cannot connect to Postgres — refusing to start.") from e

    # ── 4. Redis ──────────────────────────────────────────────────────────
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(str(settings.REDIS_URL), decode_responses=True)
        await redis_client.ping()
        await redis_client.aclose()
        logger.info("redis_connected")
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
        raise RuntimeError("Cannot connect to Redis — refusing to start.") from e

    # ── 5. Qdrant ─────────────────────────────────────────────────────────
    try:
        from qdrant_client import AsyncQdrantClient
        qdrant = AsyncQdrantClient(
            url=str(settings.QDRANT_URL),
            api_key=settings.QDRANT_API_KEY,
        )
        await qdrant.get_collections()
        await qdrant.close()
        logger.info("qdrant_connected")
    except Exception as e:
        logger.error("qdrant_connection_failed", error=str(e))
        raise RuntimeError("Cannot connect to Qdrant — refusing to start.") from e

    # ── 6. Elasticsearch ──────────────────────────────────────────────────
    try:
        from elasticsearch import AsyncElasticsearch
        es = AsyncElasticsearch(
            str(settings.ELASTIC_URL),
            basic_auth=(settings.ELASTIC_USER, settings.ELASTIC_PASSWORD),
        )
        await es.ping()
        await es.close()
        logger.info("elasticsearch_connected")
    except Exception as e:
        logger.error("elasticsearch_connection_failed", error=str(e))
        raise RuntimeError("Cannot connect to Elasticsearch — refusing to start.") from e

    logger.info("startup_complete")

    yield  # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("shutdown_begin")
    await engine.dispose()
    logger.info("shutdown_complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG Platform API",
        version=settings.APP_VERSION,
        # Disable docs in production — OpenAPI schema leaks endpoint structure
        docs_url="/docs" if settings.APP_ENV != "production" else None,
        redoc_url="/redoc" if settings.APP_ENV != "production" else None,
        openapi_url="/openapi.json" if settings.APP_ENV != "production" else None,
        lifespan=lifespan,
    )

    # Order matters: middleware → exception handlers → routers
    register_middleware(app)
    register_exception_handlers(app)
    _register_routers(app)

    return app


def _register_routers(app: FastAPI) -> None:
    from app.api.v1.router import v1_router
    app.include_router(v1_router, prefix="/v1")

    # Health check — no auth, no versioning, used by load balancers
    @app.get("/health", include_in_schema=False)
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})


# Entry point for uvicorn
app = create_app()