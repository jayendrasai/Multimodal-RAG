"""
app/core/middleware.py

Middleware stack. Registered in main.py in this order (outermost first):
1. RequestIDMiddleware  — injects unique request_id into every request
2. AuditContextMiddleware — binds user_id + request_id to structlog context
3. CORSMiddleware — strict origin allowlist, no wildcards

CORS note: CORSMiddleware must be the OUTERMOST middleware so preflight
OPTIONS requests are handled before any auth checks fire.
"""

import uuid
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from app.config import get_settings

logger = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Inject a unique request_id on every request.
    Stored in request.state and returned in response headers.
    Enables tracing a single request across logs without a full APM tool.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Bind to structlog context — every log line within this request includes it
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        # Clear context after request completes — prevents context bleed between requests
        structlog.contextvars.clear_contextvars()

        return response


class AuditContextMiddleware(BaseHTTPMiddleware):
    """
    After JWT auth runs, bind user_id to the structlog context.
    This means every log line for an authenticated request includes user_id
    without the application code needing to pass it explicitly.

    Note: user_id is available on request.state only after the auth dependency
    runs. This middleware runs after RequestIDMiddleware but user_id binding
    happens post-auth — see comments inline.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        # user_id is set by the get_current_user dependency after JWT validation
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            structlog.contextvars.bind_contextvars(user_id=str(user_id))

        return response


def register_middleware(app: FastAPI) -> None:
    """Register all middleware. Call once in app factory."""
    settings = get_settings()

    # ── CORS — must be first (outermost) ───────────────────────────────────
    # Explicit allowlist only. No wildcards.
    # In production, ALLOWED_ORIGINS must be set in env or startup fails.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.ALLOWED_ORIGINS],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
        max_age=600,  # Cache preflight for 10 minutes
    )

    # ── Request ID — second ────────────────────────────────────────────────
    app.add_middleware(RequestIDMiddleware)

    # ── Audit context — third ──────────────────────────────────────────────
    app.add_middleware(AuditContextMiddleware)