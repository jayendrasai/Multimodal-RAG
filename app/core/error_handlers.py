"""
app/core/error_handlers.py

Global FastAPI exception handlers.
This is the ONLY place where exceptions are mapped to HTTP responses.
Business logic raises AppError subclasses — never HTTPException directly.

Security rule: clients never see stack traces, internal error messages,
DB errors, or service names. All internal detail is logged server-side only.
"""

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import (
    AppError,
    ForbiddenError,
    InvalidTokenError,
    NotFoundError,
    RateLimitExceededError,
    UnauthorizedError,
    ValidationError,
)

logger = structlog.get_logger(__name__)

# Map custom exception types to HTTP status codes
_STATUS_MAP: dict[type[AppError], int] = {
    ValidationError: 400,
    InvalidTokenError: 401,
    UnauthorizedError: 401,
    ForbiddenError: 403,
    NotFoundError: 404,
    RateLimitExceededError: 429,
}


def _error_response(status: int, code: str, message: str) -> JSONResponse:
    """Standardised error envelope. Shape never changes — clients can rely on it."""
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Call this once in app factory. Registers all handlers on the FastAPI app."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        status = _STATUS_MAP.get(type(exc), 500)

        # Log with full context server-side
        log = logger.bind(
            request_id=getattr(request.state, "request_id", None),
            path=request.url.path,
            error_code=exc.code,
        )

        if status >= 500:
            # Server errors: log full exception — NEVER send details to client
            log.error("internal_error", error=str(exc))
            return _error_response(500, "INTERNAL_ERROR", "An unexpected error occurred.")

        # Client errors: log at warning level
        log.warning("client_error", status=status, error=str(exc))

        # RateLimitExceededError gets a Retry-After header
        if isinstance(exc, RateLimitExceededError):
            response = _error_response(429, exc.code, exc.message)
            response.headers["Retry-After"] = str(exc.retry_after_seconds)
            return response

        return _error_response(status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_pydantic_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Pydantic v2 request validation errors — return field-level detail."""
        logger.warning(
            "request_validation_error",
            path=request.url.path,
            errors=exc.errors(),
        )
        # Safe to return field names to the client — never return raw values
        details = [
            {"field": ".".join(str(loc) for loc in err["loc"]), "message": err["msg"]}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "VALIDATION_ERROR", "details": details}},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_starlette_http(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Catch Starlette 404/405/etc. and normalise to our error envelope."""
        logger.info(
            "http_exception",
            path=request.url.path,
            status=exc.status_code,
        )
        return _error_response(exc.status_code, "HTTP_ERROR", exc.detail or "Request error.")

    @app.exception_handler(Exception)
    async def handle_unhandled(request: Request, exc: Exception) -> JSONResponse:
        """
        Last-resort handler. Should never fire in production — if it does,
        something raised a raw Exception instead of AppError.
        Log full traceback internally. Return nothing useful to client.
        """
        logger.exception(
            "unhandled_exception",
            path=request.url.path,
            exc_info=exc,
        )
        return _error_response(500, "INTERNAL_ERROR", "An unexpected error occurred.")