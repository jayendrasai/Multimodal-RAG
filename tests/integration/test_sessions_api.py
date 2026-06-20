"""
tests/integration/test_sessions_api.py

Session lifecycle tests.

Coverage:
  - create, list, history, end — happy paths
  - end_session is not idempotent (409 on second call)
  - cannot add turns to an ended session
  - working memory cache: cold-cache fallback to Postgres
  - working memory respects CONTEXT_WINDOW_TURNS cap
  - adversarial: user cannot view or end another user's session
"""

import uuid
import pytest

pytestmark = pytest.mark.asyncio


class TestCreateSession:
    async def test_create_session_success(self, client, auth_headers):
        resp = await client.post("/v1/sessions", json={}, headers=auth_headers)
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert data["memory_loaded_count"] == 0

    async def test_create_session_with_context_hint(self, client, auth_headers):
        resp = await client.post(
            "/v1/sessions",
            json={"context_hint": "contract review"},
            headers=auth_headers,
        )
        assert resp.status_code == 201

    async def test_create_session_requires_auth(self, client):
        resp = await client.post("/v1/sessions", json={})
        assert resp.status_code == 401


class TestListSessions:
    async def test_list_empty(self, client, auth_headers):
        resp = await client.get("/v1/sessions", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["sessions"] == []

    async def test_list_after_create(self, client, auth_headers):
        await client.post("/v1/sessions", json={}, headers=auth_headers)
        await client.post("/v1/sessions", json={}, headers=auth_headers)

        resp = await client.get("/v1/sessions", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["sessions"]) == 2


class TestSessionHistory:
    async def test_history_empty_for_new_session(self, client, auth_headers):
        create = await client.post("/v1/sessions", json={}, headers=auth_headers)
        session_id = create.json()["session_id"]

        resp = await client.get(f"/v1/sessions/{session_id}/history", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["turn_count"] == 0

    async def test_history_nonexistent_session(self, client, auth_headers):
        fake_id = str(uuid.uuid4())
        resp = await client.get(f"/v1/sessions/{fake_id}/history", headers=auth_headers)
        assert resp.status_code == 404


class TestEndSession:
    async def test_end_session_success(self, client, auth_headers):
        create = await client.post("/v1/sessions", json={}, headers=auth_headers)
        session_id = create.json()["session_id"]

        resp = await client.post(f"/v1/sessions/{session_id}/end", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary_id"] is None  # Phase 1 — no summarization yet
        assert data["memory_stored"] is False

    async def test_end_session_twice_returns_conflict(self, client, auth_headers):
        create = await client.post("/v1/sessions", json={}, headers=auth_headers)
        session_id = create.json()["session_id"]

        r1 = await client.post(f"/v1/sessions/{session_id}/end", headers=auth_headers)
        assert r1.status_code == 200

        r2 = await client.post(f"/v1/sessions/{session_id}/end", headers=auth_headers)
        assert r2.status_code == 409

    async def test_end_nonexistent_session(self, client, auth_headers):
        fake_id = str(uuid.uuid4())
        resp = await client.post(f"/v1/sessions/{fake_id}/end", headers=auth_headers)
        assert resp.status_code == 404


class TestWorkingMemory:
    """
    These exercise SessionService directly (not via HTTP) since there's
    no public endpoint yet for adding turns — that's wired up by the
    query pipeline in Day 6. Testing the service layer now means Day 6
    integration is just plumbing, not new logic.
    """

    async def test_add_turn_and_retrieve_working_memory(self, db, redis_client, test_user):
        from app.services.session_service import SessionService

        service = SessionService(db=db, redis=redis_client)
        session = await service.create_session(user_id=test_user["id"])

        await service.add_turn(test_user["id"], session.id, "user", "What is the cap?")
        await service.add_turn(test_user["id"], session.id, "assistant", "The cap is $500,000.")

        memory = await service.get_working_memory(test_user["id"], session.id)
        assert len(memory) == 2
        assert memory[0]["role"] == "user"
        assert memory[1]["role"] == "assistant"

    async def test_working_memory_caps_at_context_window(self, db, redis_client, test_user):
        from app.services.session_service import SessionService
        from app.config import get_settings

        settings = get_settings()
        service = SessionService(db=db, redis=redis_client)
        session = await service.create_session(user_id=test_user["id"])

        for i in range(settings.CONTEXT_WINDOW_TURNS + 5):
            await service.add_turn(test_user["id"], session.id, "user", f"message {i}")

        memory = await service.get_working_memory(test_user["id"], session.id)
        assert len(memory) == settings.CONTEXT_WINDOW_TURNS
        # Most recent turns retained, oldest evicted
        assert memory[-1]["content"] == f"message {settings.CONTEXT_WINDOW_TURNS + 4}"

    async def test_working_memory_cold_cache_falls_back_to_postgres(
        self, db, redis_client, test_user
    ):
        """
        Simulates Redis cache expiry: clear the cache directly, then
        confirm working memory is correctly rebuilt from Postgres.
        """
        from app.services.session_service import SessionService
        from app.cache.session_cache import clear_session_cache

        service = SessionService(db=db, redis=redis_client)
        session = await service.create_session(user_id=test_user["id"])
        await service.add_turn(test_user["id"], session.id, "user", "first message")

        # Simulate cache expiry
        await clear_session_cache(redis_client, session.id)

        memory = await service.get_working_memory(test_user["id"], session.id)
        assert len(memory) == 1
        assert memory[0]["content"] == "first message"

    async def test_cannot_add_turn_to_ended_session(self, db, redis_client, test_user):
        from app.services.session_service import SessionService
        from app.core.exceptions import ConflictError

        service = SessionService(db=db, redis=redis_client)
        session = await service.create_session(user_id=test_user["id"])
        await service.end_session(test_user["id"], session.id)

        with pytest.raises(ConflictError):
            await service.add_turn(test_user["id"], session.id, "user", "too late")


class TestUserIsolation:
    async def test_user_cannot_see_other_users_sessions(self, client, test_user, admin_user):
        login_a = await client.post("/v1/auth/token", json={
            "username": test_user["username"], "password": test_user["password"]
        })
        headers_a = {"Authorization": f"Bearer {login_a.json()['access_token']}"}
        await client.post("/v1/sessions", json={}, headers=headers_a)

        login_b = await client.post("/v1/auth/token", json={
            "username": admin_user["username"], "password": admin_user["password"]
        })
        headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

        resp = await client.get("/v1/sessions", headers=headers_b)
        assert resp.json()["sessions"] == []

    async def test_user_cannot_end_other_users_session(self, client, test_user, admin_user):
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

        resp = await client.post(f"/v1/sessions/{session_id}/end", headers=headers_b)
        assert resp.status_code == 404  # not 403 — row is invisible, not forbidden