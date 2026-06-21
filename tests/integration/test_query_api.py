"""
tests/integration/test_query_api.py

Query pipeline tests. All external calls (embedder, Qdrant, Elasticsearch,
Cohere/Anthropic) are mocked -- these test orchestration, retry logic,
audit logging, and session integration, not the ML/LLM stack itself.

A real end-to-end query test against live infra belongs in the manual
Docker testing guide, not here -- LLM calls are slow, cost money, and
non-deterministic, which makes them a poor fit for a fast pytest suite
that needs to be safe to run repeatedly.
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.asyncio


def _mock_chunk(chunk_id="c1", text="The liability cap is $500,000.", score=0.9):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "document_id": str(uuid.uuid4()),
        "filename": "contract.pdf",
        "rerank_score": score,
    }


# @pytest.fixture
# def mock_pipeline_pass():
#     """Full pipeline mocked to a clean PASS on the first attempt."""
#     with patch("app.services.retrieval_service.embed_query", new=AsyncMock(return_value=[0.1] * 1024)), \
#          patch("app.search.es_search.bm25_search", new=AsyncMock(return_value=[_mock_chunk()])), \
#          patch("app.vector_store.qdrant_search.dense_search", new=AsyncMock(return_value=[_mock_chunk()])), \
#          patch("app.services.reranker_service.rerank", new=AsyncMock(return_value=[_mock_chunk()])), \
#          patch(
#              "app.services.generation_service.generate_answer",
#              new=AsyncMock(return_value=("The liability cap is $500,000.", [_mock_chunk()])),
#          ), \
#          patch(
#              "app.services.critic_service.check_groundedness",
#              new=AsyncMock(return_value={"result": "PASS", "unsupported_claims": [], "reasoning": "ok"}),
#          ):
#         yield


# @pytest.fixture
# def mock_pipeline_fail_then_pass():
#     """First critic check FAILs, retry succeeds with PASS."""
#     verdicts = [
#         {"result": "FAIL", "unsupported_claims": ["the $500,000 figure"], "reasoning": "not found"},
#         {"result": "PASS", "unsupported_claims": [], "reasoning": "ok now"},
#     ]
#     with patch("app.services.retrieval_service.embed_query", new=AsyncMock(return_value=[0.1] * 1024)), \
#          patch("app.search.es_search.bm25_search", new=AsyncMock(return_value=[_mock_chunk()])), \
#          patch("app.vector_store.qdrant_search.dense_search", new=AsyncMock(return_value=[_mock_chunk()])), \
#          patch("app.services.reranker_service.rerank", new=AsyncMock(return_value=[_mock_chunk()])), \
#          patch(
#              "app.services.generation_service.generate_answer",
#              new=AsyncMock(return_value=("The cap is $500,000.", [_mock_chunk()])),
#          ), \
#          patch("app.services.critic_service.check_groundedness", new=AsyncMock(side_effect=verdicts)):
#         yield


# @pytest.fixture
# def mock_pipeline_always_fail():
#     """Critic always FAILs -- exhausts retries."""
#     with patch("app.services.retrieval_service.embed_query", new=AsyncMock(return_value=[0.1] * 1024)), \
#          patch("app.search.es_search.bm25_search", new=AsyncMock(return_value=[_mock_chunk()])), \
#          patch("app.vector_store.qdrant_search.dense_search", new=AsyncMock(return_value=[_mock_chunk()])), \
#          patch("app.services.reranker_service.rerank", new=AsyncMock(return_value=[_mock_chunk()])), \
#          patch(
#              "app.services.generation_service.generate_answer",
#              new=AsyncMock(return_value=("Some unverifiable claim.", [_mock_chunk()])),
#          ), \
#          patch(
#              "app.services.critic_service.check_groundedness",
#              new=AsyncMock(return_value={
#                  "result": "FAIL", "unsupported_claims": ["unverifiable claim"], "reasoning": "no"
#              }),
#          ):
#         yield

@pytest.fixture
def mock_pipeline_pass():
    """Full pipeline mocked to a clean PASS on the first attempt."""
    with patch("app.pipelines.query_pipeline.hybrid_retrieve", new=AsyncMock(return_value=[_mock_chunk()])), \
         patch("app.pipelines.query_pipeline.rerank", new=AsyncMock(return_value=[_mock_chunk()])), \
         patch(
             "app.pipelines.query_pipeline.generate_answer",
             new=AsyncMock(return_value=("The liability cap is $500,000.", [_mock_chunk()])),
         ), \
         patch(
             "app.pipelines.query_pipeline.check_groundedness",
             new=AsyncMock(return_value={"result": "PASS", "unsupported_claims": [], "reasoning": "ok"}),
         ):
        yield


@pytest.fixture
def mock_pipeline_fail_then_pass():
    """First critic check FAILs, retry succeeds with PASS."""
    verdicts = [
        {"result": "FAIL", "unsupported_claims": ["the $500,000 figure"], "reasoning": "not found"},
        {"result": "PASS", "unsupported_claims": [], "reasoning": "ok now"},
    ]
    with patch("app.pipelines.query_pipeline.hybrid_retrieve", new=AsyncMock(return_value=[_mock_chunk()])), \
         patch("app.pipelines.query_pipeline.rerank", new=AsyncMock(return_value=[_mock_chunk()])), \
         patch(
             "app.pipelines.query_pipeline.generate_answer",
             new=AsyncMock(return_value=("The cap is $500,000.", [_mock_chunk()])),
         ), \
         patch("app.pipelines.query_pipeline.check_groundedness", new=AsyncMock(side_effect=verdicts)):
        yield


@pytest.fixture
def mock_pipeline_always_fail():
    """Critic always FAILs -- exhausts retries."""
    with patch("app.pipelines.query_pipeline.hybrid_retrieve", new=AsyncMock(return_value=[_mock_chunk()])), \
         patch("app.pipelines.query_pipeline.rerank", new=AsyncMock(return_value=[_mock_chunk()])), \
         patch(
             "app.pipelines.query_pipeline.generate_answer",
             new=AsyncMock(return_value=("Some unverifiable claim.", [_mock_chunk()])),
         ), \
         patch(
             "app.pipelines.query_pipeline.check_groundedness",
             new=AsyncMock(return_value={
                 "result": "FAIL", "unsupported_claims": ["unverifiable claim"], "reasoning": "no"
             }),
         ):
        yield


class TestQueryEndpoint:
    async def test_query_success_no_session(self, client, auth_headers, mock_pipeline_pass):
        resp = await client.post(
            "/v1/query", json={"query": "What is the liability cap?"}, headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["critic_result"] == "PASS"
        assert data["retry_count"] == 0
        assert data["session_id"] is None
        assert len(data["sources"]) >= 1

    async def test_query_requires_auth(self, client):
        resp = await client.post("/v1/query", json={"query": "test"})
        assert resp.status_code == 401

    async def test_query_empty_string_rejected(self, client, auth_headers):
        resp = await client.post("/v1/query", json={"query": ""}, headers=auth_headers)
        assert resp.status_code == 422

    async def test_query_retries_on_critic_fail(self, client, auth_headers, mock_pipeline_fail_then_pass):
        resp = await client.post(
            "/v1/query", json={"query": "What is the cap?"}, headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["critic_result"] == "PASS"
        assert data["retry_count"] == 1

    async def test_query_returns_caveat_after_exhausting_retries(
        self, client, auth_headers, mock_pipeline_always_fail
    ):
        resp = await client.post(
            "/v1/query", json={"query": "What is the cap?"}, headers=auth_headers
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["critic_result"] == "FAIL"
        assert "could not be fully verified" in data["answer"]

    async def test_query_with_nonexistent_session_404s(self, client, auth_headers, mock_pipeline_pass):
        fake_session = str(uuid.uuid4())
        resp = await client.post(
            "/v1/query",
            json={"query": "test", "session_id": fake_session},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    async def test_query_with_valid_session_writes_turns(self, client, auth_headers, mock_pipeline_pass):
        create = await client.post("/v1/sessions", json={}, headers=auth_headers)
        session_id = create.json()["session_id"]

        resp = await client.post(
            "/v1/query",
            json={"query": "What is the liability cap?", "session_id": session_id},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session_id

        history = await client.get(f"/v1/sessions/{session_id}/history", headers=auth_headers)
        assert history.json()["turn_count"] == 2  # user turn + assistant turn

    async def test_query_rate_limit(self, client, auth_headers, mock_pipeline_pass):
        statuses = []
        for _ in range(61):
            resp = await client.post(
                "/v1/query", json={"query": "test"}, headers=auth_headers
            )
            statuses.append(resp.status_code)
        assert 429 in statuses


class TestUserIsolation:
    async def test_user_cannot_query_with_other_users_session(
        self, client, test_user, admin_user, mock_pipeline_pass
    ):
        login_a = await client.post("/v1/auth/token", json={
            "username": test_user["username"], "password": test_user["password"]
        })
        headers_a = {"Authorization": f"Bearer {login_a.json()['access_token']}"}
        create = await client.post("/v1/sessions", json={}, headers=headers_a)
        session_id = create.json()["session_id"]

        login_b = await client.post("/v1/auth/token", json={
            "username": admin_user["username"], "password": admin_user["password"]
        })
        headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

        resp = await client.post(
            "/v1/query",
            json={"query": "test", "session_id": session_id},
            headers=headers_b,
        )
        assert resp.status_code == 404