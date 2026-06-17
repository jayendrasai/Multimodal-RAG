"""
tests/integration/test_register_api.py

Registration endpoint tests.
"""

import pytest

pytestmark = pytest.mark.asyncio


class TestRegister:
    async def test_register_success(self, client):
        resp = await client.post("/v1/users/register", json={
            "username": "newuser1",
            "password": "Testpassword123!",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "newuser1"
        assert "id" in data
        assert "hashed_password" not in data  # never exposed
        assert "password" not in data

    async def test_register_duplicate_username(self, client):
        payload = {"username": "dupeuser", "password": "Testpassword123!"}
        r1 = await client.post("/v1/users/register", json=payload)
        
        assert r1.status_code == 201

        r2 = await client.post("/v1/users/register", json=payload)
        assert r2.status_code == 409
        assert r2.json()["error"]["message"] == "Username 'dupeuser' is already taken."

    async def test_register_weak_password_too_short(self, client):
        resp = await client.post("/v1/users/register", json={
            "username": "weakuser",
            "password": "Short1",
        })
        assert resp.status_code == 422

    async def test_register_password_no_uppercase(self, client):
        resp = await client.post("/v1/users/register", json={
            "username": "weakuser2",
            "password": "alllowercase1",
        })
        assert resp.status_code == 422

    async def test_register_password_no_digit(self, client):
        resp = await client.post("/v1/users/register", json={
            "username": "weakuser3",
            "password": "NoDigitPassword",
        })
        assert resp.status_code == 422

    async def test_register_invalid_username_chars(self, client):
        resp = await client.post("/v1/users/register", json={
            "username": "user name!",
            "password": "Testpassword123!",
        })
        assert resp.status_code == 422

    async def test_register_username_normalised_to_lowercase(self, client):
        resp = await client.post("/v1/users/register", json={
            "username": "MixedCase",
            "password": "Testpassword123!",
        })
        assert resp.status_code == 201
        assert resp.json()["username"] == "mixedcase"

    async def test_registered_user_can_login(self, client):
        await client.post("/v1/users/register", json={
            "username": "loginafter",
            "password": "Testpassword123!",
        })
        login = await client.post("/v1/auth/token", json={
            "username": "loginafter",
            "password": "Testpassword123!",
        })
        assert login.status_code == 200
        assert "access_token" in login.json()