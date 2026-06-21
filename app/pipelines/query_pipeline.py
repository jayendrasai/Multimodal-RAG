"""
app/pipelines/query_pipeline.py

The full query flow, per blueprint diagram:
  1. Retrieve (BM25 + dense, RRF fusion)
  2. Rerank (top-100 -> top-10)
  3. Context build (token budgets)
  4. Generate
  5. Critic check
  6. FAIL -> targeted retry (isolate unsupported claim, re-retrieve, regenerate)
     PASS -> return response + sources + audit log

Retry strategy (blueprint: "isolate failed claim, generate targeted
sub-query, re-retrieve with increased recall"):
  On FAIL, we build a sub-query from the unsupported claim(s) the critic
  flagged, re-retrieve with a LARGER top_k (increased recall), rerank,
  and regenerate. This targets the specific gap rather than blindly
  re-running the identical query, which would likely produce the same
  unsupported claim again.

  MAX_CRITIC_RETRIES (default 2, from config) bounds the loop. If still
  FAIL after exhausting retries, we return the best available answer
  with a visible caveat rather than failing the request outright.

Plain Python state machine -- no LangGraph. Deliberate Phase 1 scope
decision; explicit RETRIEVE -> GENERATE -> CRITIQUE -> RETRY states are
Phase 3 (LangGraph state machine, per blueprint roadmap).
"""

import time
import uuid

import structlog

from app.config import get_settings
from app.services.critic_service import check_groundedness
from app.services.generation_service import generate_answer
from app.services.reranker_service import rerank
from app.services.retrieval_service import hybrid_retrieve

logger = structlog.get_logger(__name__)
settings = get_settings()


async def run_query_pipeline(
    user_id: uuid.UUID,
    query: str,
    history: list[dict],
) -> dict:
    """
    Returns:
      {
        "answer": str,
        "sources": list[dict],
        "critic_result": "PASS" | "FAIL",
        "critic_corrections": int,
        "retry_count": int,
        "latency_ms": int,
      }
    """
    start_time = time.monotonic()
    retry_count = 0
    critic_corrections = 0

    current_query = query
    recall_k = settings.RERANKER_RECALL_K

    answer = ""
    chunks_used: list[dict] = []
    verdict: dict = {"result": "FAIL", "unsupported_claims": [], "reasoning": ""}

    while retry_count <= settings.MAX_CRITIC_RETRIES:
        candidates = await hybrid_retrieve(user_id, current_query, top_k=recall_k)

        reranked = await rerank(current_query, candidates, top_k=settings.RERANKER_TOP_K)

        answer, chunks_used = await generate_answer(
            query=query,
            chunks=reranked,
            history=history,
        )

        verdict = await check_groundedness(answer, chunks_used)

        if verdict["result"] == "PASS":
            break

        retry_count += 1
        if retry_count > settings.MAX_CRITIC_RETRIES:
            logger.warning(
                "critic_retries_exhausted",
                user_id=str(user_id),
                retry_count=retry_count - 1,
                unsupported_claims=verdict.get("unsupported_claims", []),
            )
            break

        critic_corrections += 1
        recall_k = min(recall_k * 2, 500)

        unsupported = verdict.get("unsupported_claims", [])
        if unsupported:
            current_query = f"{query} -- specifically regarding: {'; '.join(unsupported)}"

        logger.info(
            "critic_failed_retrying",
            user_id=str(user_id),
            retry_count=retry_count,
            new_recall_k=recall_k,
        )

    latency_ms = int((time.monotonic() - start_time) * 1000)

    if verdict["result"] == "FAIL":
        answer = (
            f"{answer}\n\n"
            f"Note: some claims in this answer could not be fully verified "
            f"against the source documents. Please double-check before relying on it."
        )

    logger.info(
        "query_pipeline_complete",
        user_id=str(user_id),
        critic_result=verdict["result"],
        retry_count=retry_count,
        latency_ms=latency_ms,
    )

    return {
        "answer": answer,
        "sources": chunks_used,
        "critic_result": verdict["result"],
        "critic_corrections": critic_corrections,
        "retry_count": retry_count,
        "latency_ms": latency_ms,
    }