from fastapi import APIRouter

from app.api.v1 import auth, documents, query, sessions, audit

v1_router = APIRouter()

v1_router.include_router(auth.router, prefix="/auth", tags=["Auth"])
v1_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
v1_router.include_router(query.router, prefix="/query", tags=["Query"])
v1_router.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])
v1_router.include_router(audit.router, prefix="/audit", tags=["Audit"])