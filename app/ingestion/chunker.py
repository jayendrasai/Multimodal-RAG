"""
app/ingestion/chunker.py

Token-accurate chunking using tiktoken — matches the LLM tokenizer so
chunk boundaries are predictable in terms of context window usage.

Spec (blueprint): 512 tokens per chunk, 64 token overlap.

Chunking strategy: split on token count, but snap chunk boundaries to
the nearest sentence end within a small lookback window where possible.
Splitting mid-sentence is the most common cause of degraded retrieval
quality — a chunk ending mid-thought loses semantic coherence.
"""

import re

import structlog
import tiktoken

from app.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# cl100k_base is used by GPT-4/Claude-family tokenizers' approximate scheme.
# Exact Claude tokenization differs slightly but cl100k is close enough for
# chunk-size budgeting purposes — we are not counting billing tokens here.
_encoding = tiktoken.get_encoding("cl100k_base")

_SENTENCE_END_RE = re.compile(r"[.!?]\s")


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """
    Split text into overlapping chunks by token count.

    Returns a list of text chunks. Each chunk (except possibly the last)
    targets `chunk_size` tokens, with `overlap` tokens repeated at the
    start of the next chunk for context continuity across boundaries.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE_TOKENS
    overlap = overlap or settings.CHUNK_OVERLAP_TOKENS

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = _encoding.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    total = len(tokens)
    stride = chunk_size - overlap

    while start < total:
        end = min(start + chunk_size, total)
        chunk_tokens = tokens[start:end]
        chunk_str = _encoding.decode(chunk_tokens)

        # Try to snap the END boundary back to a sentence end, but only
        # if we're not at the very end of the document and the snap
        # doesn't shrink the chunk by more than 15% of chunk_size —
        # otherwise we'd produce tiny chunks near sentence-dense text.
        if end < total:
            chunk_str = _snap_to_sentence_boundary(chunk_str, chunk_size)

        chunks.append(chunk_str.strip())

        if end >= total:
            break
        start += stride

    logger.debug("chunked_text", total_tokens=total, chunk_count=len(chunks))
    return [c for c in chunks if c]  # drop any empty chunks from snapping


def _snap_to_sentence_boundary(chunk_str: str, chunk_size: int) -> str:
    """
    Look for the last sentence-ending punctuation in the final 15% of the
    chunk and truncate there if found. Prevents chunks ending mid-sentence
    without meaningfully reducing chunk size.
    """
    lookback_chars = max(50, int(len(chunk_str) * 0.15))
    tail = chunk_str[-lookback_chars:]
    matches = list(_SENTENCE_END_RE.finditer(tail))
    if not matches:
        return chunk_str

    last_match = matches[-1]
    cutoff = len(chunk_str) - lookback_chars + last_match.end()
    return chunk_str[:cutoff]


def count_tokens(text: str) -> int:
    """Utility — used by token budget enforcement in the query pipeline."""
    return len(_encoding.encode(text))