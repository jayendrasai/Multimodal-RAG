"""
tests/integration/test_phase1_exit_criterion.py

Blueprint Phase 1 exit criterion, verbatim:
  "A user can upload documents, ask questions with follow-up context,
   and get grounded answers. Nothing else."

This test exists to verify exactly that sentence as ONE continuous
journey, not as a collection of already-tested individual endpoints.
Every assertion below maps to a clause in the exit criterion:

  "upload documents"          -> ingest succeeds, status=ready
  "ask questions"              -> query returns a grounded answer
  "with follow-up context"     -> second query correctly uses session history
  "grounded answers"           -> critic_result is present and sources are non-empty

All external calls (embedder, retrieval, reranker, LLM, critic) are
mocked, consistent with the rest of the pytest suite -- the REAL version
of this exact journey is what manual_docker_testing_rag_pipeline_full.txt
Sections 2 and 4 already verified against live infra. This test's job
is to make sure the journey doesn't silently break in CI as the codebase
changes, not to re-prove the ML stack works.
"""

import io
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.asyncio

_TEST_DOC_BYTES = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 60>>stream\nBT /F1 12 Tf 100 700 Td (The liability cap is five hundred thousand dollars.) Tj ET\nendstream endobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"xref\n0 6\n0000000000 65535 f \n"
    b"trailer<</Size 6/Root 1 0 R>>\n%%EOF"
)


def _chunk(text="The liability cap is $500,000.", cid="c1"):
    return {
        "chunk_id": cid,
        "text": text,
        "document_id": "doc-1",
        "filename": "test.pdf",
        "rerank_score": 0.95,
    }


class TestPhase1ExitCriterion:
    async def test_full_user_journey_upload_ask_followup_grounded(
        self, client, auth_headers
    ):
        """
        One continuous journey: register/login (fixture-provided) ->
        upload a document -> create a session -> ask a question ->
        ask a follow-up that depends on the first answer -> verify
        groundedness signals are present throughout.
        """
        # ── "A user can upload documents" ────────────────────────────────
        with patch("app.pipelines.ingest_pipeline.embed_texts", new=AsyncMock(return_value=[[0.1] * 1024])), \
             patch("app.pipelines.ingest_pipeline.upsert_chunks", new=AsyncMock()), \
             patch("app.pipelines.ingest_pipeline.index_chunks", new=AsyncMock()):

            files = {"file": ("test.pdf", io.BytesIO(_TEST_DOC_BYTES), "application/pdf")}
            ingest_resp = await client.post(
                "/v1/documents/ingest", files=files, headers=auth_headers
            )

        assert ingest_resp.status_code == 201, "Upload must succeed"
        assert ingest_resp.json()["status"] == "ready", "Document must finish processing, not fail"
        document_id = ingest_resp.json()["document_id"]

        # Confirm it's actually listed -- "uploaded" means retrievable, not just 201'd
        list_resp = await client.get("/v1/documents", headers=auth_headers)
        assert list_resp.json()["total"] == 1
        assert list_resp.json()["documents"][0]["id"] == document_id

        # Create a session for the conversational part of the journey
        session_resp = await client.post("/v1/sessions", json={}, headers=auth_headers)
        assert session_resp.status_code == 201
        session_id = session_resp.json()["session_id"]

        # ── "ask questions ... and get grounded answers" (first turn) ────
        with patch(
            "app.pipelines.query_pipeline.hybrid_retrieve",
            new=AsyncMock(return_value=[_chunk()])
            ), \
            patch(
                "app.pipelines.query_pipeline.rerank",
                new=AsyncMock(return_value=[_chunk()])
            ), \
            patch(
                "app.pipelines.query_pipeline.generate_answer",
                new=AsyncMock(
                    return_value=("The liability cap is $500,000.", [_chunk()])
                ),
            ), \
            patch(
                "app.pipelines.query_pipeline.check_groundedness",
                new=AsyncMock(
                    return_value={
                        "result": "PASS",
                        "unsupported_claims": [],
                        "reasoning": "grounded",
                    }
                ),
            ):
            first_resp = await client.post(
                "/v1/query",
                json={"query": "What is the liability cap?", "session_id": session_id},
                headers=auth_headers,
            )

        assert first_resp.status_code == 200
        first_data = first_resp.json()
        assert first_data["critic_result"] == "PASS", "First answer must be grounded"
        assert len(first_data["sources"]) > 0, "Grounded answer must cite sources"
        assert "500,000" in first_data["answer"] or "500000" in first_data["answer"]

        # ── "ask questions with follow-up context" (second turn) ─────────
        # The mock can't actually demonstrate the LLM USING history (that's
        # the real model's job, verified manually) -- but this confirms the
        # pipeline correctly LOADS and FORWARDS history, and that the turn
        # gets recorded, which is the structural guarantee pytest can make.
        with patch(
            "app.pipelines.query_pipeline.hybrid_retrieve",
            new=AsyncMock(return_value=[_chunk()])
        ), \
            patch(
                "app.pipelines.query_pipeline.rerank",
                new=AsyncMock(return_value=[_chunk()])
            ), \
            patch(
                "app.pipelines.query_pipeline.generate_answer",
                new=AsyncMock(
                    return_value=(
                        "Yes, the same $500,000 cap applies to indemnification.",
                        [_chunk()]
                    )
                ),
            ) as mock_generate, \
            patch(
                "app.pipelines.query_pipeline.check_groundedness",
                new=AsyncMock(
                    return_value={
                        "result": "PASS",
                        "unsupported_claims": [],
                        "reasoning": "grounded",
                    }
                ),
            ):
            followup_resp = await client.post(
                "/v1/query",
                json={"query": "Does that cap also apply to indemnification?", "session_id": session_id},
                headers=auth_headers,
            )

        assert followup_resp.status_code == 200
        followup_data = followup_resp.json()
        assert followup_data["critic_result"] == "PASS"

        # Structural proof that follow-up context was actually loaded and
        # passed to generation -- the history argument generate_answer
        # received must contain the FIRST turn's content.
        call_kwargs = mock_generate.call_args.kwargs
        history_passed = call_kwargs.get("history", [])
        history_contents = [t.get("content", "") for t in history_passed]
        assert any("liability cap" in c.lower() for c in history_contents), (
            "Follow-up query must receive prior turn as context -- "
            "this is the 'with follow-up context' clause of the exit criterion"
        )

        # ── Verify the full conversation is durably recorded ──────────────
        history_resp = await client.get(
            f"/v1/sessions/{session_id}/history", headers=auth_headers
        )
        assert history_resp.json()["turn_count"] == 4  # 2 user + 2 assistant

        # ── Verify the compliance trail (audit log) was written ───────────
        audit_resp = await client.get("/v1/audit/logs", headers=auth_headers)
        assert audit_resp.json()["total"] == 2  # one per query call
        for entry in audit_resp.json()["logs"]:
            assert entry["critic_result"] == "PASS"
            assert len(entry["query_hash"]) == 64  # hashed, never raw text

        # ── End the session cleanly ────────────────────────────────────────
        end_resp = await client.post(f"/v1/sessions/{session_id}/end", headers=auth_headers)
        assert end_resp.status_code == 200

        # ── Clean up the document ──────────────────────────────────────────
        with patch("app.services.ingestion_service.qdrant_delete", new=AsyncMock()), \
             patch("app.services.ingestion_service.es_delete", new=AsyncMock()):
            delete_resp = await client.delete(
                f"/v1/documents/{document_id}", headers=auth_headers
            )
        assert delete_resp.status_code == 200

        # If every assertion above passed, the Phase 1 exit criterion,
        # as literally stated in the blueprint, is met for this journey.