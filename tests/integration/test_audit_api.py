"""
tests/integration/test_audit_api.py
"""

import uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.asyncio


async def _write_audit_row(db, user_id, critic_result="PASS"):
    from app.models.audit import AuditLog
    record = AuditLog(
        id=uuid.uuid4(),
        user_id=user_id,
        session_id=None,
        query_hash="a" * 64,
        sources_used=[{"chunk_id": "c1", "document_id": "d1"}],
        critic_result=critic_result,
        critic_corrections=0,
        retry_count=0,
        latency_ms=1234,
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    await db.commit()
    return record


class TestAuditLogs:
    async def test_list_empty(self, client, auth_headers):
        resp = await client.get("/v1/audit/logs", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    async def test_list_after_writes(self, client, auth_headers, db, test_user):
        await _write_audit_row(db, test_user["id"])
        await _write_audit_row(db, test_user["id"])

        resp = await client.get("/v1/audit/logs", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["logs"]) == 2
        # query_hash returned, never raw query text
        assert len(data["logs"][0]["query_hash"]) == 64

    async def test_pagination(self, client, auth_headers, db, test_user):
        for _ in range(5):
            await _write_audit_row(db, test_user["id"])

        resp = await client.get("/v1/audit/logs?limit=2&offset=0", headers=auth_headers)
        data = resp.json()
        assert data["total"] == 5
        assert len(data["logs"]) == 2

    async def test_requires_auth(self, client):
        resp = await client.get("/v1/audit/logs")
        assert resp.status_code == 401

    async def test_user_cannot_see_other_users_audit_logs(
        self, client, db, test_user, admin_user
    ):
        await _write_audit_row(db, admin_user["id"])

        login_a = await client.post("/v1/auth/token", json={
            "username": test_user["username"], "password": test_user["password"]
        })
        headers_a = {"Authorization": f"Bearer {login_a.json()['access_token']}"}

        resp = await client.get("/v1/audit/logs", headers=headers_a)
        assert resp.json()["total"] == 0