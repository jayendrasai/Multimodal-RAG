-- -- infra/postgres/init.sql
-- -- Runs once at Postgres container first boot.
-- -- Alembic manages table DDL. This file handles:
-- --   1. Extensions
-- --   2. RLS policies (security layer — not managed by Alembic)
-- --   3. Performance indexes that don't belong in migration files

-- -- ── Extensions ──────────────────────────────────────────────────────────────
-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- CREATE EXTENSION IF NOT EXISTS "vector";         -- pgvector — used in Phase 2
-- CREATE EXTENSION IF NOT EXISTS "pg_trgm";        -- trigram index for text search


-- -- ── Row-Level Security ──────────────────────────────────────────────────────
-- -- RLS enforcement model:
-- -- Every query that touches a user-partitioned table must first call:
-- --   SET LOCAL app.current_user_id = '<uuid>';
-- -- This is enforced by app/db/rls.py at the application layer (Layer 1).
-- -- The policies below enforce it at the database layer (Layer 2).
-- -- If Layer 1 is bypassed (direct DB access, migration script, etc.),
-- -- Layer 2 prevents cross-user data access.

-- -- episodic_memory
-- ALTER TABLE episodic_memory ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE episodic_memory FORCE ROW LEVEL SECURITY;

-- CREATE POLICY rls_episodic_select ON episodic_memory
--     FOR SELECT
--     USING (user_id = current_setting('app.current_user_id', true)::UUID);

-- CREATE POLICY rls_episodic_insert ON episodic_memory
--     FOR INSERT
--     WITH CHECK (user_id = current_setting('app.current_user_id', true)::UUID);

-- CREATE POLICY rls_episodic_delete ON episodic_memory
--     FOR DELETE
--     USING (user_id = current_setting('app.current_user_id', true)::UUID);


-- -- semantic_memory
-- ALTER TABLE semantic_memory ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE semantic_memory FORCE ROW LEVEL SECURITY;

-- CREATE POLICY rls_semantic_select ON semantic_memory
--     FOR SELECT
--     USING (user_id = current_setting('app.current_user_id', true)::UUID);

-- CREATE POLICY rls_semantic_insert ON semantic_memory
--     FOR INSERT
--     WITH CHECK (user_id = current_setting('app.current_user_id', true)::UUID);

-- CREATE POLICY rls_semantic_delete ON semantic_memory
--     FOR DELETE
--     USING (user_id = current_setting('app.current_user_id', true)::UUID);


-- -- documents
-- ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE documents FORCE ROW LEVEL SECURITY;

-- CREATE POLICY rls_documents ON documents
--     FOR ALL
--     USING (user_id = current_setting('app.current_user_id', true)::UUID)
--     WITH CHECK (user_id = current_setting('app.current_user_id', true)::UUID);


-- -- sessions
-- ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE sessions FORCE ROW LEVEL SECURITY;

-- CREATE POLICY rls_sessions ON sessions
--     FOR ALL
--     USING (user_id = current_setting('app.current_user_id', true)::UUID)
--     WITH CHECK (user_id = current_setting('app.current_user_id', true)::UUID);


-- -- session_turns
-- ALTER TABLE session_turns ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE session_turns FORCE ROW LEVEL SECURITY;

-- CREATE POLICY rls_session_turns ON session_turns
--     FOR ALL
--     USING (user_id = current_setting('app.current_user_id', true)::UUID)
--     WITH CHECK (user_id = current_setting('app.current_user_id', true)::UUID);


-- -- audit_logs: users can read their own; only app service role can insert
-- ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE audit_logs FORCE ROW LEVEL SECURITY;

-- CREATE POLICY rls_audit_select ON audit_logs
--     FOR SELECT
--     USING (user_id = current_setting('app.current_user_id', true)::UUID);

-- -- INSERT on audit_logs is NOT RLS-gated — the application service role
-- -- writes audit records on behalf of users. The app enforces user_id correctly.
-- -- FORCE ROW LEVEL SECURITY ensures even superusers respect policies
-- -- unless they explicitly SET row_security = off (require explicit bypass, not silent).


-- -- ── Performance Indexes ─────────────────────────────────────────────────────
-- -- These complement the ORM-defined indexes on FK columns.

-- -- Episodic memory: retrieve recent summaries per user (Phase 2 cross-session retrieval)
-- CREATE INDEX IF NOT EXISTS idx_episodic_user_created
--     ON episodic_memory (user_id, created_at DESC);

-- -- Episodic memory: expire old records efficiently
-- CREATE INDEX IF NOT EXISTS idx_episodic_expires
--     ON episodic_memory (expires_at)
--     WHERE expires_at IS NOT NULL;

-- -- Documents: filter by user + status (ingestion pipeline status checks)
-- CREATE INDEX IF NOT EXISTS idx_documents_user_status
--     ON documents (user_id, status);

-- -- Audit logs: compliance queries by user + time range
-- CREATE INDEX IF NOT EXISTS idx_audit_user_created
--     ON audit_logs (user_id, created_at DESC);

-- -- Session turns: fast history retrieval for working memory
-- CREATE INDEX IF NOT EXISTS idx_turns_session_created
--     ON session_turns (session_id, created_at ASC);


-- -- ── Superuser bypass note ───────────────────────────────────────────────────
-- -- FORCE ROW LEVEL SECURITY means even the table owner (raguser) respects
-- -- RLS policies. To bypass for migrations or admin scripts, use:
-- --   SET row_security = off;
-- -- This must be explicit — never implicit. Document every such bypass.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";