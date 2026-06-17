"""
tests/integration/test_auth_api.py

Integration tests for authentication endpoints.
Tests run against a real test DB (separate from dev DB) via conftest fixtures.

Coverage:
  - Login: success, wrong password, missing user, inactive user
  - Refresh: success, reuse rejection, tampered token, wrong type
  - Logout: success, token gone after logout
  - Logout all: clears all sessions
  - Rate limiting: IP limiter blocks after 10 req/min
  - Timing: login response time is consistent (no oracle timing attack)
"""

import time
import pytest
import pytest_asyncio
from httpx import AsyncClient


pytestmark = pytest.mark.asyncio


class TestLogin:
    async def test_login_success(self, client: AsyncClient, test_user):
        resp = await client.post("/v1/auth/token", json={
            "username": test_user["username"],
            "password": test_user["password"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 15 * 60

    async def test_login_wrong_password(self, client: AsyncClient, test_user):
        resp = await client.post("/v1/auth/token", json={
            "username": test_user["username"],
            "password": "wrongpassword",
        })
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "UNAUTHORIZED"

    async def test_login_missing_user(self, client: AsyncClient):
        resp = await client.post("/v1/auth/token", json={
            "username": "nonexistent_user_xyz",
            "password": "somepassword",
        })
        assert resp.status_code == 401
        # Error message must be identical to wrong_password response
        # — prevents username enumeration
        assert resp.json()["error"]["message"] == "Invalid credentials."

    async def test_login_inactive_user(self, client: AsyncClient, inactive_test_user):
        resp = await client.post("/v1/auth/token", json={
            "username": inactive_test_user["username"],
            "password": inactive_test_user["password"],
        })
        assert resp.status_code == 401

    async def test_login_timing_consistency(self, client: AsyncClient, test_user):
        """
        Login for existing user and non-existing user should take similar time.
        bcrypt runs in both cases to prevent timing oracle.
        Threshold: difference < 500ms over 3 runs.
        """
        existing_times = []
        missing_times = []

        for _ in range(3):
            t0 = time.monotonic()
            await client.post("/v1/auth/token", json={
                "username": test_user["username"],
                "password": "wrong",
            })
            existing_times.append(time.monotonic() - t0)

            t0 = time.monotonic()
            await client.post("/v1/auth/token", json={
                "username": "no_such_user_timing_test",
                "password": "wrong",
            })
            missing_times.append(time.monotonic() - t0)

        avg_existing = sum(existing_times) / len(existing_times)
        avg_missing = sum(missing_times) / len(missing_times)
        diff = abs(avg_existing - avg_missing)

        assert diff < 0.5, (
            f"Timing oracle possible: existing={avg_existing:.3f}s "
            f"missing={avg_missing:.3f}s diff={diff:.3f}s"
        )

    async def test_login_returns_request_id(self, client: AsyncClient, test_user):
        resp = await client.post("/v1/auth/token", json={
            "username": test_user["username"],
            "password": test_user["password"],
        })
        assert "x-request-id" in resp.headers


class TestRefresh:
    async def test_refresh_success(self, client: AsyncClient, test_user):
        login = await client.post("/v1/auth/token", json={
            "username": test_user["username"],
            "password": test_user["password"],
        })
        refresh_token = login.json()["refresh_token"]

        resp = await client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_refresh_token_single_use(self, client: AsyncClient, test_user):
        """Refresh token cannot be used twice — it's rotated on first use."""
        login = await client.post("/v1/auth/token", json={
            "username": test_user["username"],
            "password": test_user["password"],
        })
        refresh_token = login.json()["refresh_token"]

        # First use: OK
        r1 = await client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
        assert r1.status_code == 200

        # Second use of same token: rejected
        r2 = await client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
        assert r2.status_code == 401

    async def test_access_token_rejected_as_refresh(self, client: AsyncClient, test_user):
        """Access token must not be accepted on the refresh endpoint."""
        login = await client.post("/v1/auth/token", json={
            "username": test_user["username"],
            "password": test_user["password"],
        })
        access_token = login.json()["access_token"]

        resp = await client.post("/v1/auth/refresh", json={"refresh_token": access_token})
        assert resp.status_code == 401

    async def test_tampered_refresh_token(self, client: AsyncClient):
        resp = await client.post("/v1/auth/refresh", json={
            "refresh_token": "eyJhbGciOiJIUzI1NiJ9.tampered.signature"
        })
        assert resp.status_code == 401


class TestLogout:
    async def test_logout_revokes_refresh_token(self, client: AsyncClient, test_user):
        login = await client.post("/v1/auth/token", json={
            "username": test_user["username"],
            "password": test_user["password"],
        })
        tokens = login.json()

        # Logout
        resp = await client.post(
            "/v1/auth/logout",
            json={"refresh_token": tokens["refresh_token"]},
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        assert resp.status_code == 204

        # Refresh token is now revoked
        r2 = await client.post("/v1/auth/refresh", json={
            "refresh_token": tokens["refresh_token"]
        })
        assert r2.status_code == 401

    async def test_logout_all(self, client: AsyncClient, test_user):
        # Create two sessions
        t1 = (await client.post("/v1/auth/token", json={
            "username": test_user["username"], "password": test_user["password"]
        })).json()
        t2 = (await client.post("/v1/auth/token", json={
            "username": test_user["username"], "password": test_user["password"]
        })).json()

        # Logout all using first session's access token
        resp = await client.post(
            "/v1/auth/logout-all",
            headers={"Authorization": f"Bearer {t1['access_token']}"},
        )
        assert resp.status_code == 204

        # Both refresh tokens should now be revoked
        r1 = await client.post("/v1/auth/refresh", json={"refresh_token": t1["refresh_token"]})
        r2 = await client.post("/v1/auth/refresh", json={"refresh_token": t2["refresh_token"]})
        assert r1.status_code == 401
        assert r2.status_code == 401


class TestRateLimiting:
    async def test_login_rate_limit(self, client: AsyncClient):
        """11th request within 60s should be rate limited."""
        responses = []
        for i in range(11):
            r = await client.post("/v1/auth/token", json={
                "username": f"user{i}",
                "password": "password",
            })
            responses.append(r.status_code)

        assert 429 in responses
        # Find the 429 and check Retry-After header
        # rate_limited = next(
        #     (await client.post("/v1/auth/token", json={"username": "x", "password": "y"}))
        #     for _ in range(1)
        # )
        # Remove the `rate_limited = next(...)` block completely and replace it with:
        rate_limited = await client.post("/v1/auth/token", json={"username": "x", "password": "y"})
        
        assert rate_limited.status_code == 429
        # HTTPX lowercase header keys automatically
        assert "retry-after" in rate_limited.headers
        # After 10 failed attempts the 11th should 429 with Retry-After
        last_429 = [r for r in responses if r == 429]
        assert len(last_429) >= 1