"""
app/pipelines/ingest_pipeline.py

Orchestrates the full ingestion flow:
  validate file → parse → chunk → embed → write to Qdrant → write to Elasticsearch

Plain Python orchestration (no LangGraph) — explicit per blueprint Phase 1 scope.

Failure handling: if any step fails partway through, partial writes to
Qdrant/Elasticsearch are NOT automatically rolled back in Phase 1 (no
distributed transaction across Postgres + Qdrant + Elasticsearch is in
scope yet). The document's Postgres status is set to "failed" so it's
visible and can be deleted/retried. This is a known Phase 1 limitation —
see the risk register note in the docstring of ingestion_service.py.
"""

import uuid

import structlog

from app.core.exceptions import ParserError
from app.ingestion.chunker import chunk_text
from app.ingestion.docx_parser import DOCXParser
from app.ingestion.email_parser import EmailParser
from app.ingestion.pdf_parser import PDFParser
from app.search.es_client import index_chunks
from app.vector_store.embedder import embed_texts
from app.vector_store.qdrant_client import upsert_chunks

logger = structlog.get_logger(__name__)

_PARSERS = {
    "pdf": PDFParser(),
    "docx": DOCXParser(),
    "eml": EmailParser(),
}


async def run_ingest_pipeline(
    user_id: uuid.UUID,
    document_id: uuid.UUID,
    file_type: str,
    file_bytes: bytes,
    filename: str,
    doc_metadata: dict | None = None,
) -> int:
    """
    Runs the full pipeline. Returns the number of chunks created.
    Raises ParserError, EmbeddingError, or RetrievalError on failure —
    caller (ingestion_service) is responsible for updating Postgres
    document status accordingly.
    """
    parser = _PARSERS.get(file_type)
    if parser is None:
        raise ParserError(f"No parser registered for file type: {file_type}")

    # ── 1. Parse ──────────────────────────────────────────────────────────
    parsed = await parser.parse(file_bytes)
    logger.info(
        "document_parsed",
        document_id=str(document_id),
        file_type=file_type,
        page_count=parsed.page_count,
        text_length=len(parsed.text),
    )

    # ── 2. Chunk ──────────────────────────────────────────────────────────
    chunks = chunk_text(parsed.text)
    if not chunks:
        raise ParserError("Document produced no chunks after parsing — file may be empty.")

    logger.info("document_chunked", document_id=str(document_id), chunk_count=len(chunks))

    # ── 3. Embed ──────────────────────────────────────────────────────────
    embeddings = await embed_texts(chunks)

    # ── 4. Generate chunk IDs (deterministic per document+index for idempotent re-ingestion) ──
    chunk_ids = [str(uuid.uuid5(document_id, str(i))) for i in range(len(chunks))]

    chunk_metadata_list = [
        {
            "filename": filename,
            **(doc_metadata or {}),
        }
        for _ in chunks
    ]

    # ── 5. Write to Qdrant (dense vectors) ───────────────────────────────
    await upsert_chunks(
        user_id=user_id,
        document_id=document_id,
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        chunk_texts=chunks,
        chunk_metadata=chunk_metadata_list,
    )

    # ── 6. Write to Elasticsearch (BM25 lexical index) ───────────────────
    await index_chunks(
        user_id=user_id,
        document_id=document_id,
        chunk_ids=chunk_ids,
        chunk_texts=chunks,
        chunk_metadata=chunk_metadata_list,
    )

    logger.info(
        "ingest_pipeline_complete",
        document_id=str(document_id),
        chunk_count=len(chunks),
    )

    return len(chunks)