"""
app/ingestion/email_parser.py

Email (.eml) parsing via mailparser.
Extracts subject, sender, recipients, date, and body text.
Attachments are NOT extracted in Phase 1 — out of scope until
multimodal ingestion (Phase 3).
"""

import asyncio

import mailparser
import structlog

from app.core.exceptions import ParserError
from app.ingestion.base_parser import BaseParser, ParsedDocument

logger = structlog.get_logger(__name__)


class EmailParser(BaseParser):
    async def parse(self, file_bytes: bytes) -> ParsedDocument:
        try:
            return await asyncio.to_thread(self._parse_sync, file_bytes)
        except ParserError:
            raise
        except Exception as e:
            logger.error("email_parse_failed", error=str(e))
            raise ParserError("Could not parse email file. File may be malformed.") from e

    def _parse_sync(self, file_bytes: bytes) -> ParsedDocument:
        mail = mailparser.parse_from_bytes(file_bytes)

        header_block = (
            f"Subject: {mail.subject}\n"
            f"From: {mail.from_}\n"
            f"To: {mail.to}\n"
            f"Date: {mail.date}\n"
            f"---\n"
        )
        body = mail.body or ""
        full_text = header_block + body

        if not body.strip():
            raise ParserError("Email contains no extractable body text.")

        return ParsedDocument(
            text=full_text,
            page_count=1,
            pages=[full_text],
            metadata={
                "parser": "mailparser",
                "subject": mail.subject,
                "from": str(mail.from_),
                "date": str(mail.date),
            },
        )