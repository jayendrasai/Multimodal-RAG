"""
app/config.py

Single source of truth for all configuration.
Reads exclusively from environment variables — never from hardcoded defaults
for secrets. If a required secret is missing, the app refuses to start.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars — don't crash on unrelated vars
    )

    # ── App ────────────────────────────────────────────────────────────────
    APP_ENV: Literal["development", "staging", "production"] = "development"
    APP_NAME: str = "rag-platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # ── Security ───────────────────────────────────────────────────────────
    # No default — app refuses to start if SECRET_KEY is unset or still the placeholder
    SECRET_KEY: str = Field(..., min_length=32)

    # Strict CORS — no wildcard ever. Comma-separated origins in env.
    # Example: ALLOWED_ORIGINS=https://app.yourdomain.com,https://admin.yourdomain.com
    ALLOWED_ORIGINS: list[AnyHttpUrl] = Field(default_factory=list)

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list) -> list:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("SECRET_KEY")
    @classmethod
    def secret_key_not_placeholder(cls, v: str) -> str:
        if v.startswith("CHANGE_ME"):
            raise ValueError(
                "SECRET_KEY is still the placeholder value. "
                "Generate with: openssl rand -hex 32"
            )
        return v

    # ── JWT ────────────────────────────────────────────────────────────────
    JWT_SECRET: str = Field(..., min_length=32)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    @field_validator("JWT_SECRET")
    @classmethod
    def jwt_secret_not_placeholder(cls, v: str) -> str:
        if v.startswith("CHANGE_ME"):
            raise ValueError(
                "JWT_SECRET is still the placeholder value. "
                "Generate with: openssl rand -hex 32"
            )
        return v

    # ── Database ───────────────────────────────────────────────────────────
    DATABASE_URL: PostgresDsn = Field(...)
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30

    # ── Redis ──────────────────────────────────────────────────────────────
    REDIS_URL: RedisDsn = Field(...)
    REDIS_SESSION_TTL_SECONDS: int = 3600       # 1 hour working memory TTL
    REDIS_QUERY_CACHE_TTL_SECONDS: int = 3600   # 1 hour query result cache

    # ── Rate Limiting ──────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_AUTH_PER_MINUTE: int = 10        # Stricter for auth endpoints

    # ── Qdrant ─────────────────────────────────────────────────────────────
    QDRANT_URL: AnyHttpUrl = Field(...)
    QDRANT_API_KEY: str = Field(...)
    QDRANT_COLLECTION_VECTOR_SIZE: int = 1024   # BGE-M3 output dimensions
    QDRANT_HNSW_EF_CONSTRUCT: int = 200
    QDRANT_HNSW_M: int = 16

    # ── Elasticsearch ──────────────────────────────────────────────────────
    ELASTIC_URL: AnyHttpUrl = Field(...)
    ELASTIC_PASSWORD: str = Field(...)
    ELASTIC_USER: str = "elastic"

    # ── LLM ────────────────────────────────────────────────────────────────
    # ANTHROPIC_API_KEY: str = Field(...)
    # LLM_MODEL: str = "claude-sonnet-4-20250514"
    # LLM_MAX_TOKENS: int = 2048
    # LLM_TEMPERATURE: float = 0.1    # Low temp for factual RAG answers

    OPENROUTER_API_KEY: str = Field(...)
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "google/gemma-4-31b-it:free"
    CRITIC_LLM_MODEL: str = "openai/gpt-oss-120b:free"
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.1    # Low temp for factual RAG answers

    # ── Reranker ───────────────────────────────────────────────────────────
    COHERE_API_KEY: str | None = None
    USE_LOCAL_RERANKER: bool = True            # True = BGE-Reranker-Large local
    RERANKER_TOP_K: int = 10                    # Top-100 recall → Top-10 precision
    RERANKER_RECALL_K: int = 100

    # ── Embedding ──────────────────────────────────────────────────────────
    USE_LOCAL_EMBEDDING: bool = True            # True = BGE-M3 local
    JINA_API_KEY: str | None = None             # Required if USE_LOCAL_EMBEDDING=False
    # CRITICAL: pin exact model version — changing this after data is indexed
    # requires a full reindex of every user's Qdrant collection.
    EMBEDDING_MODEL_VERSION: str = "BAAI/bge-m3"

    # ── Document Ingestion ─────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 50
    CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 64
    ALLOWED_MIME_TYPES: list[str] = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "message/rfc822",                           # .eml
        "application/vnd.ms-outlook",               # .msg
    ]

    # ── Query Pipeline ─────────────────────────────────────────────────────
    CONTEXT_WINDOW_TURNS: int = 6               # Last N turns injected into context
    MAX_CRITIC_RETRIES: int = 2                 # Retry limit before returning best answer
    RRF_K: int = 60                             # Reciprocal Rank Fusion constant

    # Token budget (must sum to less than LLM context window)
    TOKEN_BUDGET_SYSTEM: int = 400
    TOKEN_BUDGET_MEMORY: int = 600              # Phase 2 — reserved but not used yet
    TOKEN_BUDGET_SEMANTIC: int = 200            # Phase 2 — reserved but not used yet
    TOKEN_BUDGET_CHUNKS: int = 2000
    TOKEN_BUDGET_HISTORY: int = 800

    @field_validator("COHERE_API_KEY")
    @classmethod
    def validate_reranker_config(cls, v: str | None, info) -> str | None:
        # Called after USE_LOCAL_RERANKER is set — if not local, key is required
        # Note: cross-field validation is handled at service init time
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached singleton. Import and call this everywhere.
    The lru_cache means Settings() is only instantiated once —
    env vars are read once at startup, not on every request.

    Usage:
        from app.config import get_settings
        settings = get_settings()
    """
    return Settings()