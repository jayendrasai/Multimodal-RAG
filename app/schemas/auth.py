"""
app/schemas/auth.py

Pydantic v2 request and response models for the auth endpoints.

Security note: TokenResponse never includes the user's password hash,
internal IDs in raw form, or any field that could leak account existence.
The login endpoint returns the same 401 for "wrong password" and
"user not found" — never disambiguate these to the client.
"""

from pydantic import BaseModel, Field, field_validator


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=100)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("username")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # access token TTL in seconds


class AccessTokenResponse(BaseModel):
    """Returned by /auth/refresh — new access token only."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int