"""
app/core/file_validation.py

File type validation via magic numbers (actual file header bytes),
never via filename extension. A .exe renamed to .pdf must be rejected.

This is a hard security requirement — extension-based validation is
trivially bypassed by renaming any file.
"""

import magic
import structlog

from app.config import get_settings
from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError

logger = structlog.get_logger(__name__)
settings = get_settings()

# Map detected MIME type → internal file type identifier
_MIME_TO_TYPE = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "message/rfc822": "eml",
    # python-magic sometimes detects .eml as text/plain if headers are minimal —
    # handled by a content-sniff fallback in detect_file_type below
}


def validate_file_size(file_bytes: bytes) -> None:
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise FileTooLargeError(settings.MAX_UPLOAD_SIZE_MB)
    if len(file_bytes) == 0:
        raise UnsupportedFileTypeError()


def detect_file_type(file_bytes: bytes, declared_filename: str) -> str:
    """
    Detect the real file type from magic numbers (header bytes).
    The declared_filename is used ONLY for the .eml content-sniff
    fallback below — never trusted for the primary type decision.

    Returns one of: "pdf", "docx", "eml"
    Raises UnsupportedFileTypeError for anything else.
    """
    detected_mime = magic.from_buffer(file_bytes, mime=True)

    file_type = _MIME_TO_TYPE.get(detected_mime)

    if file_type is None:
        # .eml files are plain text with RFC 822 headers. libmagic sometimes
        # reports them as text/plain rather than message/rfc822. Sniff the
        # first few hundred bytes for email header patterns as a fallback —
        # this still inspects content, not the filename, so it doesn't
        # weaken the magic-number requirement.
        if detected_mime in ("text/plain", "text/x-mail"):
            head = file_bytes[:1000].decode("utf-8", errors="ignore").lower()
            if any(h in head for h in ("from:", "to:", "subject:", "date:", "message-id:")):
                file_type = "eml"

    if file_type is None:
        logger.warning(
            "unsupported_file_type_rejected",
            detected_mime=detected_mime,
            declared_filename=declared_filename,
        )
        raise UnsupportedFileTypeError()

    if file_type not in {"pdf", "docx", "eml"}:
        raise UnsupportedFileTypeError()

    return file_type