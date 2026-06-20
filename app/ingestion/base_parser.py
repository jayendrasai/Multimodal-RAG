"""
app/ingestion/base_parser.py

Abstract interface every document parser implements.
One parser per file type — isolated so a broken PDF parser cannot
crash DOCX ingestion (blueprint requirement).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ParsedDocument:
    """
    Normalised output from any parser. Downstream chunking and embedding
    code only ever deals with this shape — never the raw file format.
    """
    text: str
    page_count: int = 1
    # Optional per-page text for parsers that support it (PDF).
    # Used to attribute chunks back to a page number for citations.
    pages: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseParser(ABC):
    """All parsers raise ParserError on failure — never a raw exception."""

    @abstractmethod
    async def parse(self, file_bytes: bytes) -> ParsedDocument:
        ...