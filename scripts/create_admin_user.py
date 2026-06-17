"""
scripts/create_admin_user.py

One-time script to create the first admin user.
Run once after Alembic migration creates the users table.

Usage:
    python scripts/create_admin_user.py --username admin --password <strong_password>

This script does NOT accept passwords via positional args to avoid
them appearing in shell history. Use --password flag or let the
script prompt interactively.
"""

import argparse
import asyncio
import getpass
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.config import get_settings
from app.core.security import hash_password
from app.models.base import Base
from app.models.user import User
from app.models.document import Document
from app.models.session import Session , SessionTurn
from app.models.audit import AuditLog


settings = get_settings()


async def create_admin(username: str, password: str) -> None:
    engine = create_async_engine(str(settings.DATABASE_URL), echo=False)
    SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as db:
        # Check if user already exists
        result = await db.execute(select(User).where(User.username == username))
        existing = result.scalar_one_or_none()

        if existing:
            print(f"ERROR: User '{username}' already exists.")
            await engine.dispose()
            sys.exit(1)

        if len(password) < 12:
            print("ERROR: Password must be at least 12 characters.")
            await engine.dispose()
            sys.exit(1)

        user = User(
            username=username,
            hashed_password=hash_password(password),
            is_active=True,
            is_admin=True,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

        print(f"Admin user created: {username} (id: {user.id})")

    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create admin user")
    parser.add_argument("--username", required=True, help="Admin username")
    parser.add_argument("--password", default=None, help="Admin password (prompted if omitted)")
    args = parser.parse_args()

    password = args.password
    if not password:
        password = getpass.getpass("Enter admin password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("ERROR: Passwords do not match.")
            sys.exit(1)

    asyncio.run(create_admin(args.username, password))


if __name__ == "__main__":
    main()