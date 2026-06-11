"""
app/core/logging.py

Structured JSON logging via structlog.

Security requirements enforced here:
1. PII and secrets are redacted BEFORE the log record is serialised.
   Redaction happens at the processor level — even if application code
   accidentally logs a password field, this processor removes it.
2. In production, output is newline-delimited JSON for log aggregators.
3. In development, output is human-readable coloured console output.

NEVER log: passwords, JWT tokens, API keys, raw user queries containing
PII, file contents, or Postgres connection strings.
"""

import logging
import sys
import re
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from app.config import get_settings

# ── Sensitive field names — matched case-insensitively ─────────────────────
_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "jwt",
        "api_key",
        "apikey",
        "authorization",
        "x-api-key",
        "anthropic_api_key",
        "cohere_api_key",
        "jina_api_key",
        "elastic_password",
        "redis_password",
        "database_url",
        "connection_string",
        "private_key",
        "secret_key",
    }
)

# Pattern to catch JWT-shaped strings in log values even if key isn't sensitive
_JWT_PATTERN = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")


def _redact_sensitive_fields(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """
    Structlog processor: walk the event dict and redact sensitive values.
    Runs before serialisation — operates on the raw dict, not the string.
    """

    def _redact_value(key: str, value: Any) -> Any:
        if isinstance(key, str) and key.lower() in _SENSITIVE_KEYS:
            return "[REDACTED]"
        if isinstance(value, str) and _JWT_PATTERN.search(value):
            return "[REDACTED:JWT]"
        if isinstance(value, dict):
            return {k: _redact_value(k, v) for k, v in value.items()}
        if isinstance(value, list):
            return [_redact_value("", item) for item in value]
        return value

    return {k: _redact_value(k, v) for k, v in event_dict.items()}


def setup_logging() -> None:
    """
    Configure structlog. Call once at application startup in main.py lifespan.
    """
    settings = get_settings()
    is_production = settings.APP_ENV == "production"

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact_sensitive_fields,  # MUST run before any serialiser
    ]

    if is_production:
        # Production: JSON output for log aggregators (Datadog, CloudWatch, etc.)
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Development: readable console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Also silence noisy third-party loggers in production
    if is_production:
        for noisy_logger in ["uvicorn.access", "httpx", "httpcore"]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)