"""
app/core/exceptions.py

Custom exception hierarchy. Every exception the application raises
should be one of these — never raise raw Exception or HTTPException
from business logic. HTTP mapping happens in error_handlers.py only.
"""


class AppError(Exception):
    """Base class for all application errors."""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


# ── 400 ────────────────────────────────────────────────────────────────────
class ValidationError(AppError):
    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message, code="VALIDATION_ERROR")
        self.field = field


class FileTooLargeError(AppError):
    def __init__(self, max_mb: int) -> None:
        super().__init__(
            f"File exceeds maximum allowed size of {max_mb}MB.",
            code="FILE_TOO_LARGE",
        )


class UnsupportedFileTypeError(AppError):
    def __init__(self) -> None:
        super().__init__(
            "File type is not supported. Allowed: PDF, DOCX, EML, MSG.",
            code="UNSUPPORTED_FILE_TYPE",
        )


# ── 401 ────────────────────────────────────────────────────────────────────
class UnauthorizedError(AppError):
    def __init__(self, message: str = "Authentication required.") -> None:
        super().__init__(message, code="UNAUTHORIZED")


class InvalidTokenError(AppError):
    def __init__(self) -> None:
        super().__init__("Token is invalid or expired.", code="INVALID_TOKEN")


# ── 403 ────────────────────────────────────────────────────────────────────
class ForbiddenError(AppError):
    def __init__(self, message: str = "You do not have permission to perform this action.") -> None:
        super().__init__(message, code="FORBIDDEN")


# ── 404 ────────────────────────────────────────────────────────────────────
class NotFoundError(AppError):
    def __init__(self, resource: str = "Resource") -> None:
        super().__init__(f"{resource} not found.", code="NOT_FOUND")


# ── 409 ────────────────────────────────────────────────────────────────────
class ConflictError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="CONFLICT")


# ── 429 ────────────────────────────────────────────────────────────────────
class RateLimitExceededError(AppError):
    def __init__(self, retry_after_seconds: int = 60) -> None:
        super().__init__(
            "Rate limit exceeded. Please slow down.",
            code="RATE_LIMIT_EXCEEDED",
        )
        self.retry_after_seconds = retry_after_seconds


# ── 500 ────────────────────────────────────────────────────────────────────
class InternalError(AppError):
    def __init__(self, message: str = "An unexpected error occurred.") -> None:
        # NEVER pass internal details to the client.
        # Log the real error separately before raising this.
        super().__init__(message, code="INTERNAL_ERROR")


class EmbeddingError(InternalError):
    """Raised when the embedding model or API fails."""
    def __init__(self) -> None:
        super().__init__("Document embedding failed. Please retry.")


class RetrievalError(InternalError):
    """Raised when vector or keyword search fails."""
    def __init__(self) -> None:
        super().__init__("Retrieval failed. Please retry.")


class GenerationError(InternalError):
    """Raised when LLM generation fails after retries."""
    def __init__(self) -> None:
        super().__init__("Answer generation failed. Please retry.")

class ParserError(InternalError):
    def __init__(self, message: str = "Document could not be parsed. The file may be corrupted or password-protected.") -> None:
        super().__init__(message)

class RerankerError(InternalError):
    """Raised when the cross-encoder reranking fails or is misconfigured."""
    def __init__(self, message: str = "Document reranking failed. Please retry.") -> None:
        super().__init__(message)