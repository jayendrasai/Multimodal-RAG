"""
app/services/generation_service.py

Context assembly + LLM generation.

Token budget enforcement (from config, set on Day 2 already):
  TOKEN_BUDGET_SYSTEM    400   — system prompt
  TOKEN_BUDGET_HISTORY   800   — last 6 turns of conversation
  TOKEN_BUDGET_CHUNKS   2000   — reranked source chunks
  (TOKEN_BUDGET_MEMORY and TOKEN_BUDGET_SEMANTIC are reserved for
   Phase 2 — unused here, but the budget math leaves room for them)

If chunks exceed their budget, lowest-ranked chunks (by rerank_score)
are dropped first — never truncated mid-chunk, since a half-sentence
of context is worse than one fewer complete chunk.
"""

import structlog
#from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import get_settings
from app.core.exceptions import GenerationError
from app.ingestion.chunker import count_tokens

logger = structlog.get_logger(__name__)
settings = get_settings()

#_client: AsyncAnthropic | None = None
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL
        )
        #_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


_SYSTEM_PROMPT = """You are a careful assistant that answers questions strictly \
using the provided source excerpts. Every factual claim in your answer must be \
directly supported by at least one excerpt. If the excerpts don't contain enough \
information to answer, say so explicitly rather than guessing or using outside \
knowledge. Cite which excerpt(s) support each claim by their [N] marker."""


def build_context(
    chunks: list[dict],
    history: list[dict],
) -> tuple[str, str, list[dict]]:
    """
    Assembles the prompt context within token budgets.
    Returns (system_prompt, user_message_with_context, chunks_actually_used).

    chunks_actually_used is returned separately because chunks may be
    dropped to fit the budget — the caller (and audit log) needs to know
    exactly which sources were actually presented to the LLM, not just
    which were retrieved.
    """
    # ── History within budget ────────────────────────────────────────────
    history_text = ""
    history_tokens_used = 0
    for turn in reversed(history):  # most recent first, to prioritize recency if trimming
        line = f"{turn['role'].capitalize()}: {turn['content']}\n"
        line_tokens = count_tokens(line)
        if history_tokens_used + line_tokens > settings.TOKEN_BUDGET_HISTORY:
            break
        history_text = line + history_text  # prepend to restore chronological order
        history_tokens_used += line_tokens

    # ── Chunks within budget ──────────────────────────────────────────────
    # Chunks are already ranked best-first by reranker — fill the budget
    # in that order, drop whatever doesn't fit rather than truncating.
    chunks_used: list[dict] = []
    chunks_text_parts: list[str] = []
    chunk_tokens_used = 0

    for i, chunk in enumerate(chunks, start=1):
        excerpt = f"[{i}] (source: {chunk.get('filename', 'unknown')})\n{chunk['text']}\n"
        excerpt_tokens = count_tokens(excerpt)
        if chunk_tokens_used + excerpt_tokens > settings.TOKEN_BUDGET_CHUNKS:
            continue  # skip this one, but keep checking later (shorter) chunks
        chunks_text_parts.append(excerpt)
        chunks_used.append(chunk)
        chunk_tokens_used += excerpt_tokens

    chunks_text = "\n".join(chunks_text_parts)

    user_message = (
        f"### Conversation so far:\n{history_text or '(no prior turns)'}\n\n"
        f"### Source excerpts:\n{chunks_text or '(no relevant sources found)'}\n\n"
        f"### Question:\n{{query}}"
    )

    return _SYSTEM_PROMPT, user_message, chunks_used


async def generate_answer(
    query: str,
    chunks: list[dict],
    history: list[dict],
) -> tuple[str, list[dict]]:
    """
    Returns (answer_text, chunks_used).
    chunks_used is needed by the caller for audit logging and for the
    critic step — the critic only checks claims against what was
    ACTUALLY shown to the model, not everything that was retrieved.
    """
    system_prompt, user_template, chunks_used = build_context(chunks, history)
    user_message = user_template.format(query=query)

    try:
        client = _get_client()
        # response = await client.messages.create(
        #     model=settings.LLM_MODEL,
        #     max_tokens=settings.LLM_MAX_TOKENS,
        #     temperature=settings.LLM_TEMPERATURE,
        #     system=system_prompt,
        #     messages=[{"role": "user", "content": user_message}],
        # )
        #answer = response.content[0].text
        response = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
        )
        answer = response.choices[0].message.content

        logger.info(
            "generation_complete",
            chunks_used=len(chunks_used),
            answer_length=len(answer),
        )

        return answer, chunks_used

    except Exception as e:
        logger.error("generation_failed", error=str(e))
        raise GenerationError() from e