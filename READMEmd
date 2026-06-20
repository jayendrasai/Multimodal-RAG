# Multimodal-RAG Platform — Phase 1

Self-healing, multi-user RAG backend. FastAPI + Postgres + Qdrant + Elasticsearch + Redis.

Phase 1 scope: ingest documents (PDF/DOCX/EML), hybrid retrieval (BM25 + dense),
cross-encoder reranking, grounded generation with a hallucination critic, per-user
data isolation enforced at the database and vector-store level. No cross-session
memory yet — that's Phase 2.

## Stack

| Component | Technology |
|---|---|
| API | FastAPI, async SQLAlchemy |
| Relational DB | PostgreSQL 16 + pgvector |
| Vector store | Qdrant (one collection per user) |
| Lexical search | Elasticsearch (BM25, one index per user) |
| Cache / sessions | Redis |
| Embedding | BGE-M3 (local, CPU) |
| Reranking | Cohere Rerank v3 (or local BGE-Reranker fallback) |
| LLM | Claude (Anthropic API) |
| Auth | JWT (15min access / 7 day refresh, rotated on use) |

## Project Layout

```
app/
├── api/v1/          Route handlers — thin, no business logic
├── core/            Security, logging, exceptions, middleware, rate limiting
├── models/          SQLAlchemy ORM models
├── schemas/         Pydantic request/response contracts
├── services/         Business logic
├── pipelines/        Multi-step orchestration (ingest, query)
├── ingestion/        Per-file-type parsers + chunker
├── vector_store/      Qdrant client + embedder
├── search/           Elasticsearch client
├── db/               Session factory + Row-Level Security enforcement
└── main.py            App factory, startup health checks

tests/
├── unit/
└── integration/       Includes adversarial cross-user isolation tests

infra/postgres/init.sql   RLS policies, pgvector extension, indexes
alembic/                  DB migrations
scripts/                  One-off ops scripts (create_admin_user.py, etc.)
```

## Security Model (non-negotiable, read before changing anything)

- **Per-user isolation, two layers.** Postgres Row-Level Security (`FORCE ROW LEVEL
  SECURITY`) is the second line of defense behind ORM-level filtering. Qdrant and
  Elasticsearch use one collection/index per user, named `sha256(user_id)` — not the
  raw UUID — so isolation holds even if a query forgets to filter.
- **`SET LOCAL`, never `SET SESSION`**, for the RLS user context (`app/db/rls.py`).
  `SET SESSION` would leak one user's context to the next request reusing the same
  pooled connection.
- **File uploads validated by magic number**, not extension or declared
  Content-Type (`app/core/file_validation.py`). Requires `libmagic1` at the OS level.
- **Passwords**: bcrypt, 12 rounds. **Refresh tokens**: SHA-256 hash stored in Redis,
  single-use, rotated on every refresh.
- **Same error for "wrong password" and "user doesn't exist."** bcrypt always runs,
  even for a non-existent user, to avoid a timing oracle.
- **5xx responses never leak internals.** Stack traces and DB errors are logged
  server-side only; the client gets a generic message.
- **Secrets never have defaults.** The app refuses to start if `SECRET_KEY` or
  `JWT_SECRET` are missing or still placeholder values.

## Running Locally

```bash
cp .env.example .env
# generate secrets:
openssl rand -hex 32   # → SECRET_KEY
openssl rand -hex 32   # → JWT_SECRET
# fill in .env, then:

docker compose up -d
docker compose exec api alembic upgrade head
docker compose exec api python scripts/create_admin_user.py --username admin
```

Full manual testing checklist: see `day2_manual_testing.md`.
Docker command reference (images, containers, rebuilds, logs): see
`docker_command_reference.txt`.

## Irreversible Decisions — Do Not Change Without a Migration Plan

1. **`EMBEDDING_MODEL_VERSION`** (currently BGE-M3). Changing this after documents
   are indexed requires re-embedding every chunk for every user. See ADR-001.
2. **Collection/index naming scheme** (`sha256(user_id)`). Changing this after
   go-live requires renaming every user's Qdrant collection and ES index.
3. **`FORCE ROW LEVEL SECURITY`** in `init.sql`. Downgrading to plain `ENABLE`
   lets the table owner role bypass RLS silently.

## What's NOT Built Yet (by design)

| Feature | Phase |
|---|---|
| Cross-session memory (episodic/semantic) | Phase 2 |
| Query decomposition, multi-hop | Phase 3 |
| Knowledge graph (Neo4j) | Phase 3 |
| Multimodal ingestion (images, tables-as-structure) | Phase 3 |
| Prometheus/Grafana monitoring | Phase 2 |

## Day-by-Day Build Log

| Day | Delivered |
|---|---|
| 1 | Folder structure, dependencies, Docker Compose stack |
| 2 | Config, logging, exceptions, middleware, RLS, ORM models, app factory |
| 3 | Rate limiter, auth endpoints (login/refresh/logout), admin script |
| 4 | Document ingestion: parsers, chunker, embedder, Qdrant/ES writes, ingest API |
| 5 | Sessions: working memory (Redis), session lifecycle endpoints |

Git commit messages for each day are kept in `dayN_git_commits.md`.

## Known Phase 1 Limitations

- No distributed transaction across Postgres + Qdrant + Elasticsearch during
  ingestion. A failure between the two store writes can leave them briefly
  inconsistent for one document. Mitigation: `DELETE /documents/{id}` is
  idempotent against both stores regardless of partial state.
- Ingestion runs synchronously in the request — large documents block the
  upload response until embedding finishes. Acceptable for Phase 1 testing;
  revisit with a background task queue if document sizes grow.
- No content-based ingestion idempotency — re-uploading the same file creates
  a new `document_id` and a new chunk set rather than detecting the duplicate.