"""
app/schemas/user.py
"""

import uuid
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=12, max_length=128)

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        v = v.strip()
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username may only contain letters, numbers, hyphens, and underscores.")
        return v.lower()

    @field_validator("password")
    @classmethod
    def password_not_trivial(cls, v: str) -> str:
        if v.lower() in {"password", "12345678", "qwerty123", "password123"}:
            raise ValueError("Password is too common.")
        has_upper = any(c.isupper() for c in v)
        has_digit = any(c.isdigit() for c in v)
        if not (has_upper and has_digit):
            raise ValueError("Password must contain at least one uppercase letter and one digit.")
        return v


class UserResponse(BaseModel):
    id: uuid.UUID
    username: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}