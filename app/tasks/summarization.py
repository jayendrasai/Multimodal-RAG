import asyncio
import logging
from sqlalchemy import text

from app.core.celery_app import celery_app
from app.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def _update_session_status(session_id: str):
    """Async helper to perform the database update."""
    async with AsyncSessionLocal() as db:
        # Note: If your RLS policies (in app/db/rls.py) strictly block background 
        # workers from updating rows without an active user_id, you will need to 
        # pass the user_id into this Celery task and call set_rls_user_id() here.
        await db.execute(
            text("""
                UPDATE sessions
                SET status = 'summarized', updated_at = now()
                WHERE id = :session_id AND status = 'summarization_pending'
            """),
            {"session_id": session_id},
        )
        await db.commit()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def summarize_session(self, session_id: str):
    """
    Sprint 1 stub: proves the enqueue-once guarantee end to end.
    Real summarization body (LLM call + outbox write) lands in Sprint 5.
    """
    logger.info("summarize_session started", extra={"session_id": session_id})

    try:
        # Bridge the synchronous Celery worker to your async SQLAlchemy setup
        asyncio.run(_update_session_status(session_id))
        
        logger.info("summarize_session completed", extra={"session_id": session_id})

    except Exception as exc:
        logger.error(
            "summarize_session failed, retrying", 
            extra={"session_id": session_id, "error": str(exc)}
        )
        # Triggers the max_retries and default_retry_delay configured in the decorator
        raise self.retry(exc=exc)