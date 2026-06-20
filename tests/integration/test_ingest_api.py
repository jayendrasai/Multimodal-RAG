"""
tests/integration/test_ingest_api.py

Document ingestion tests.

Note: these tests mock the embedder and Qdrant/Elasticsearch clients
since loading BGE-M3 and hitting real infra in unit/integration tests
is slow and environment-dependent. A separate smoke test (not pytest)
should run the full pipeline against real Docker infra before deploy —
see day2_manual_testing.md pattern extended for Day 4.
"""

import io
import uuid
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.asyncio

# Minimal valid PDF bytes — just enough for pdfplumber to open without error
_MINIMAL_PDF = (
    b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 58>>stream\nBT /F1 12 Tf 100 700 Td (This is test content for parsing.) Tj ET\nendstream endobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"xref\n0 6\n0000000000 65535 f \n"
    b"trailer<</Size 6/Root 1 0 R>>\n%%EOF"
)


@pytest.fixture
def mock_embedding_pipeline():
    """Mock embedder, Qdrant, and Elasticsearch — test the route/service logic, not the ML stack."""
    with patch("app.pipelines.ingest_pipeline.embed_texts", new=AsyncMock(return_value=[[0.1] * 1024])), \
         patch("app.pipelines.ingest_pipeline.upsert_chunks", new=AsyncMock()), \
         patch("app.pipelines.ingest_pipeline.index_chunks", new=AsyncMock()):
        yield


class TestIngestEndpoint:
    async def test_ingest_pdf_success(self, client, auth_headers, mock_embedding_pipeline):
        files = {"file": ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")}
        resp = await client.post("/v1/documents/ingest", files=files, headers=auth_headers)

        assert resp.status_code == 201
        data = resp.json()
        assert data["filename"] == "test.pdf"
        assert data["status"] == "ready"
        assert data["chunk_count"] >= 1

    async def test_ingest_requires_auth(self, client):
        files = {"file": ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")}
        resp = await client.post("/v1/documents/ingest", files=files)
        assert resp.status_code == 401

    async def test_ingest_rejects_oversized_file(self, client, auth_headers):
        # Construct a file just over 50MB
        big_content = b"%PDF-1.4\n" + (b"0" * (51 * 1024 * 1024))
        files = {"file": ("huge.pdf", io.BytesIO(big_content), "application/pdf")}
        resp = await client.post("/v1/documents/ingest", files=files, headers=auth_headers)
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "FILE_TOO_LARGE"

    async def test_ingest_rejects_disguised_exe(self, client, auth_headers):
        """An .exe renamed to .pdf must be rejected by magic-number check."""
        exe_header = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff"  # PE header
        files = {"file": ("totally_a_pdf.pdf", io.BytesIO(exe_header), "application/pdf")}
        resp = await client.post("/v1/documents/ingest", files=files, headers=auth_headers)
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"

    async def test_ingest_rejects_empty_file(self, client, auth_headers):
        files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
        resp = await client.post("/v1/documents/ingest", files=files, headers=auth_headers)
        assert resp.status_code == 400

    async def test_ingest_with_valid_metadata(self, client, auth_headers, mock_embedding_pipeline):
        files = {"file": ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")}
        data = {"metadata": '{"source": "upload", "category": "contract"}'}
        resp = await client.post(
            "/v1/documents/ingest", files=files, data=data, headers=auth_headers
        )
        assert resp.status_code == 201

    async def test_ingest_with_invalid_metadata_json(self, client, auth_headers):
        files = {"file": ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")}
        data = {"metadata": "not valid json{{{"}
        resp = await client.post(
            "/v1/documents/ingest", files=files, data=data, headers=auth_headers
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "VALIDATION_ERROR"

    async def test_ingest_rate_limit(self, client, auth_headers, mock_embedding_pipeline):
        """11th ingest request within 60s for the same user should 429."""
        statuses = []
        for _ in range(11):
            files = {"file": ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")}
            resp = await client.post("/v1/documents/ingest", files=files, headers=auth_headers)
            statuses.append(resp.status_code)
        assert 429 in statuses


class TestListDocuments:
    async def test_list_empty(self, client, auth_headers):
        resp = await client.get("/v1/documents", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    async def test_list_after_ingest(self, client, auth_headers, mock_embedding_pipeline):
        files = {"file": ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")}
        await client.post("/v1/documents/ingest", files=files, headers=auth_headers)

        resp = await client.get("/v1/documents", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["documents"][0]["filename"] == "test.pdf"

    async def test_list_requires_auth(self, client):
        resp = await client.get("/v1/documents")
        assert resp.status_code == 401


class TestDeleteDocument:
    async def test_delete_nonexistent_document(self, client, auth_headers):
        fake_id = str(uuid.uuid4())
        resp = await client.delete(f"/v1/documents/{fake_id}", headers=auth_headers)
        assert resp.status_code == 404

    async def test_delete_requires_auth(self, client):
        fake_id = str(uuid.uuid4())
        resp = await client.delete(f"/v1/documents/{fake_id}")
        assert resp.status_code == 401

    async def test_delete_success(self, client, auth_headers, mock_embedding_pipeline):
        with patch("app.services.ingestion_service.qdrant_delete", new=AsyncMock()), \
             patch("app.services.ingestion_service.es_delete", new=AsyncMock()):
            files = {"file": ("test.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")}
            ingest_resp = await client.post("/v1/documents/ingest", files=files, headers=auth_headers)
            document_id = ingest_resp.json()["document_id"]

            resp = await client.delete(f"/v1/documents/{document_id}", headers=auth_headers)
            assert resp.status_code == 200
            assert resp.json()["deleted"] is True


class TestUserIsolation:
    """
    Adversarial test: User A's documents must never be visible or
    deletable by User B. This is the most important test in this file.
    """

    async def test_user_cannot_see_other_users_documents(
        self, client, test_user, admin_user, mock_embedding_pipeline
    ):
        # test_user uploads a document
        login_a = await client.post("/v1/auth/token", json={
            "username": test_user["username"], "password": test_user["password"]
        })
        headers_a = {"Authorization": f"Bearer {login_a.json()['access_token']}"}

        files = {"file": ("secret.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")}
        await client.post("/v1/documents/ingest", files=files, headers=headers_a)

        # admin_user (different account) lists documents — must see 0
        login_b = await client.post("/v1/auth/token", json={
            "username": admin_user["username"], "password": admin_user["password"]
        })
        headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

        resp_b = await client.get("/v1/documents", headers=headers_b)
        assert resp_b.json()["total"] == 0

    async def test_user_cannot_delete_other_users_document(
        self, client, test_user, admin_user, mock_embedding_pipeline
    ):
        login_a = await client.post("/v1/auth/token", json={
            "username": test_user["username"], "password": test_user["password"]
        })
        headers_a = {"Authorization": f"Bearer {login_a.json()['access_token']}"}

        files = {"file": ("secret.pdf", io.BytesIO(_MINIMAL_PDF), "application/pdf")}
        ingest_resp = await client.post("/v1/documents/ingest", files=files, headers=headers_a)
        document_id = ingest_resp.json()["document_id"]

        login_b = await client.post("/v1/auth/token", json={
            "username": admin_user["username"], "password": admin_user["password"]
        })
        headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

        resp = await client.delete(f"/v1/documents/{document_id}", headers=headers_b)
        # Must be 404, not 403 — confirming existence to a non-owner is
        # itself a minor information leak. RLS makes the row invisible,
        # not merely forbidden.
        assert resp.status_code == 404