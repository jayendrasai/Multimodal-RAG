"""
app/ingestion/pdf_parser.py

PDF text extraction. pdfplumber is primary — it handles tables better
than pypdf. pypdf is the fallback for PDFs that pdfplumber chokes on
(corrupted streams, unusual encodings).

Runs extraction in a thread pool — pdfplumber is synchronous/CPU-bound
and would block the event loop if called directly in an async function.
"""

import asyncio
import io

import pdfplumber
import structlog
from pypdf import PdfReader

from app.core.exceptions import ParserError
from app.ingestion.base_parser import BaseParser, ParsedDocument

logger = structlog.get_logger(__name__)


class PDFParser(BaseParser):
    async def parse(self, file_bytes: bytes) -> ParsedDocument:
        try:
            return await asyncio.to_thread(self._parse_with_pdfplumber, file_bytes)
        except Exception as e:
            logger.warning("pdfplumber_failed_trying_pypdf", error=str(e))
            try:
                return await asyncio.to_thread(self._parse_with_pypdf, file_bytes)
            except Exception as e2:
                logger.error("pdf_parse_failed_both_methods", error=str(e2))
                raise ParserError("Could not extract text from PDF. File may be corrupted or scanned/image-only.") from e2

    def _parse_with_pdfplumber(self, file_bytes: bytes) -> ParsedDocument:
        pages: list[str] = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                # Extract tables separately and append as structured text —
                # raw extract_text() often mangles table layout
                tables = page.extract_tables()
                table_text = ""
                for table in tables:
                    rows = [" | ".join(str(cell or "") for cell in row) for row in table]
                    table_text += "\n".join(rows) + "\n"
                pages.append((page_text + "\n" + table_text).strip())

        full_text = "\n\n".join(pages)
        if not full_text.strip():
            raise ParserError("PDF contains no extractable text. It may be a scanned image without OCR.")

        return ParsedDocument(
            text=full_text,
            page_count=len(pages),
            pages=pages,
            metadata={"parser": "pdfplumber"},
        )

    def _parse_with_pypdf(self, file_bytes: bytes) -> ParsedDocument:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n\n".join(pages)

        if not full_text.strip():
            raise ParserError("PDF contains no extractable text.")

        return ParsedDocument(
            text=full_text,
            page_count=len(pages),
            pages=pages,
            metadata={"parser": "pypdf_fallback"},
        )