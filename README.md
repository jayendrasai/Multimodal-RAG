# RAG Platform — Phase 1

A multi-user, self-healing Retrieval-Augmented Generation backend. Users upload documents, ask questions in natural conversation with follow-up context, and receive grounded, source-cited answers — with every claim checked against the source material before it reaches the user, and a full compliance audit trail of every query.

## What This Is

A single FastAPI service backed by four specialized data stores, each chosen for a distinct job:

- **PostgreSQL 16 + pgvector** — system of record: users, document metadata, sessions, conversation turns, audit trail. `pgvector` is installed but unused in Phase 1 (reserved for Phase 2 episodic memory).
- **Qdrant** — dense vector storage for semantic search, one isolated collection per user.
- **Elasticsearch** — BM25 lexical/keyword search, one isolated index per user.
- **Redis** — working-memory cache for active conversation turns, and sliding-window rate-limiter counters.

## Architecture

```
Routes (app/api/v1/)      → thin HTTP handlers, no business logic
Services (app/services/)  → business logic, orchestrates models + external calls
Pipelines (app/pipelines/) → multi-step workflows (ingest, query)
Models (app/models/)      → SQLAlchemy ORM / Postgres schema
Vector/Search clients     → Qdrant and Elasticsearch read/write wrappers
Cache                      → Redis working memory + rate limiting
Core                       → security, logging, error handling, middleware
```

Each layer talks only to the layer directly below it. Routes never touch the database directly. This separation is what kept a mid-project LLM provider migration (Anthropic → OpenRouter/Gemma) confined to two files, since provider-specific code was isolated behind `generation_service.py` and `critic_service.py` from the start.

## Per-User Data Isolation

Isolation is enforced at multiple independent layers, so a single bug at any one layer doesn't leak data across users:

- **Postgres**: Row-Level Security with `FORCE ROW LEVEL SECURITY` — even the database owner role cannot bypass it without an explicit, auditable override. Enforced via a transaction-local session variable set per request (never session-scoped, which would leak across pooled connections).
- **Qdrant**: one collection per user, named by a SHA-256 hash of the user ID rather than the raw UUID, to resist enumeration if the admin API is ever exposed.
- **Elasticsearch**: identical per-user index isolation scheme.
- **API layer**: every protected route resolves the user from a verified JWT and threads that ID through every downstream call. Accessing another user's resource by ID returns `404`, not `403` — existence is never confirmed to a non-owner.

## RAG Pipeline

### Ingestion

```
Upload (PDF / DOCX / EML)
  → Magic-number file-type detection (libmagic — never trusts filename
    extension or declared Content-Type; a renamed .exe is rejected even
    if labeled .pdf)
  → Size validation (50MB max)
  → Parse (pdfplumber + pypdf fallback for PDF, python-docx for DOCX,
    mailparser for .eml)
  → Chunk (512 tokens, 64 token overlap, sentence-boundary-aware)
  → Embed each chunk (BGE-M3, 1024-dim dense vectors, local CPU inference)
  → Write to Qdrant (dense vectors) and Elasticsearch (raw text for BM25)
  → Document status: ready (or failed — visible and deletable/retriable)
```

### Query

```
User question (+ optional session_id for conversation continuity)
  → Load last 6 turns from Redis, or rebuild from Postgres on cold cache
  → HYBRID RETRIEVAL (concurrent):
      - BM25 lexical search (Elasticsearch), top 100
      - Dense semantic search (Qdrant), top 100
  → Reciprocal Rank Fusion (RRF, k=60) — merges by rank position, since
    BM25 and cosine similarity scores are on incomparable scales
  → Cross-encoder reranking — fused top-100 reranked by a model that
    reads query + candidate together, narrowed to top 10
  → Context assembly within a hard token budget — never truncates a
    chunk mid-sentence; drops lowest-ranked whole chunks if needed
  → Generation — LLM answers using only the assembled context
  → Groundedness critic — a SEPARATE LLM call verifies every claim
    against the retrieved chunks. Fails CLOSED: an unparseable critic
    response counts as FAIL, never a silent PASS.
  → PASS → return answer + sources
    FAIL → build a targeted sub-query from the unsupported claim,
           increase retrieval recall, retry (max 2) → otherwise return
           the best attempt with a visible caveat
  → Audit log written unconditionally (hashed query, sources, critic
    result, retry count, latency)
  → Turns written back to Postgres (durable) and Redis (cache)
```

### Design Rationale

- **Hybrid retrieval, not one method**: BM25 catches exact term matches dense embeddings can blur; dense search catches semantically-equivalent phrasing BM25 would miss. Fusing both outperforms either alone.
- **Two-stage retrieval**: broad, cheap recall (BM25 + dense) followed by narrow, expensive precision (cross-encoder rerank) — standard IR practice, since reranking the full corpus would be too slow.
- **A separate critic call, not self-grading**: a model checking its own same-pass output is a weaker check than a fresh call that only sees the candidate answer and the sources.
- **Plain Python orchestration, not a graph framework**: the retry loop is an explicit, bounded `while` loop. Graph-based state machines are deferred until query decomposition or multi-hop logic actually require them.

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| API framework | FastAPI + async SQLAlchemy | Async-native, automatic OpenAPI docs, Pydantic typing |
| Relational database | PostgreSQL 16 + pgvector | RLS for hard multi-tenant isolation; pgvector pre-installed for Phase 2 |
| Vector store | Qdrant | Per-user collections, HNSW indexing, async client |
| Lexical search | Elasticsearch | Mature BM25 implementation, per-user index isolation |
| Cache / sessions | Redis | Sub-millisecond reads, sliding-window rate limiting |
| Embedding model | BGE-M3 (local, CPU) | Strong multilingual dense retrieval, no per-call cost, no data leaves infra |
| Reranker | BGE-Reranker-Large (local) or Cohere Rerank v3 | Two-stage retrieval precision |
| LLM (generation + critic) | OpenRouter → Gemma-4-31B | Migrated from Anthropic Claude for cost — see Known Limitations |
| Auth | JWT (15min access / 7-day refresh, rotated on use) + bcrypt | Stateless access tokens, revocable refresh tokens |
| Containerization | Docker + Docker Compose, multi-stage build | CPU-only PyTorch pinned to avoid multi-GB CUDA images |
| File-type validation | libmagic / python-magic | Detects real file type from header bytes, immune to extension spoofing |
| Chunking | tiktoken | Token-accurate chunk sizing matching LLM context budgeting |
| Testing | pytest + pytest-asyncio + httpx | Async-native integration tests against real test Postgres + Redis |

## API Reference

All endpoints versioned under `/v1`. All except `/health`, `/auth/token`, `/auth/refresh`, and `/users/register` require a Bearer JWT.

**Authentication**

| Method | Path | Purpose | Notes |
|---|---|---|---|
| POST | `/v1/auth/token` | Login | Returns access + refresh token. Same error for wrong password and unknown user. |
| POST | `/v1/auth/refresh` | Exchange refresh token for new access token | Single-use — each refresh token works once, then rotates. |
| POST | `/v1/auth/logout` | Revoke one refresh token | `204 No Content`. |
| POST | `/v1/auth/logout-all` | Revoke every refresh token for the user | "Logout all devices." |

**Users**

| Method | Path | Purpose | Notes |
|---|---|---|---|
| POST | `/v1/users/register` | Create a new account | Password policy enforced (12+ chars, upper+digit). Rate-limited 5/hour/IP. |

**Documents**

| Method | Path | Purpose | Notes |
|---|---|---|---|
| POST | `/v1/documents/ingest` | Upload and process a document | Multipart upload, optional JSON metadata. Rate-limited 10/min/user. |
| GET | `/v1/documents` | List the caller's documents | RLS-scoped automatically. |
| DELETE | `/v1/documents/{id}` | Delete a document and its chunks | Removes from Postgres, Qdrant, and Elasticsearch. |

**Sessions**

| Method | Path | Purpose | Notes |
|---|---|---|---|
| POST | `/v1/sessions` | Start a new conversation session | |
| GET | `/v1/sessions` | List the caller's sessions | |
| GET | `/v1/sessions/{id}/history` | Full turn-by-turn history | |
| POST | `/v1/sessions/{id}/end` | End a session | Not idempotent — calling twice returns `409`. |

**Query**

| Method | Path | Purpose | Notes |
|---|---|---|---|
| POST | `/v1/query` | Ask a question against ingested documents | `session_id` optional. Rate-limited 60/min/user. |

**Audit**

| Method | Path | Purpose | Notes |
|---|---|---|---|
| GET | `/v1/audit/logs` | View the caller's own compliance audit trail | Paginated. Query text is hashed, never stored raw. |

**System**

| Method | Path | Purpose | Notes |
|---|---|---|---|
| GET | `/health` | Liveness check | No auth. Used by load balancers / orchestrators. |

## Known Limitations & Carried-Forward Risks

- **No saga/compensation pattern** across Postgres + Qdrant + Elasticsearch during ingestion. A failure between the vector/search writes can leave a document's data split across stores until it's deleted and re-ingested. Document status is always visible (`failed`), and delete is unconditionally safe against partial state.
- **Ingestion is synchronous within the request** — large documents block the HTTP response until embedding finishes. Acceptable at current scale; a background task queue is the natural next step if document size or volume grows.
- **No content-based ingestion idempotency** — uploading the same file twice creates two separate documents and chunk sets.
- **Generator and critic share the same model family** (Gemma-4-31B, free tier, via OpenRouter) following a cost-driven provider migration. This is architecturally weaker than an independent or larger critic model — smaller models are more prone to confidently approving their own plausible-but-ungrounded output, which is exactly the failure mode the critic exists to catch. The fail-closed-on-parse-error protection holds regardless of model, but does not catch a confidently wrong PASS verdict on a genuinely hallucinated claim. This should be periodically re-tested with deliberately unanswerable questions, not verified once and assumed permanent.
- **`end_session` is not idempotent by design** (`409` on repeat call) — worth revisiting if any client retry logic could legitimately double-call it after a timeout.

## What's Next (Phase 2 — Not Yet Built)

- Session summarization into structured episodic memory
- Cross-session memory retrieval
- Semantic entity extraction (spaCy NER)
- Memory scoping and decay (recency-weighted relevance)
- Dedicated token budget manager for memory context slots
- User-facing memory management API (list/delete/export, GDPR-style)
- Prometheus + Grafana monitoring dashboard

None of this was started in Phase 1, by design.