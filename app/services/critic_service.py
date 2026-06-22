"""
app/services/critic_service.py

Basic LLM critic: does every claim in the generated answer appear in
a retrieved chunk? Per blueprint spec exactly — this is the
hallucination check, not a quality/style check.

The critic is a SEPARATE LLM call, not the same call that generated
the answer. Asking a model to grade its own output in the same
generation pass is a known weak self-check; a fresh call with only
the answer + sources (no awareness of "this is your own work") is a
meaningfully different and stronger check, even though it's the same
underlying model.

Output contract: the critic MUST return structured JSON so the retry
loop can parse it programmatically. We instruct the model explicitly
to output ONLY JSON, and we defensively strip markdown fences before
parsing, since models occasionally wrap JSON in ```json fences despite
instructions not to.
"""

import json

import structlog
#from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import get_settings
from app.core.exceptions import GenerationError

logger = structlog.get_logger(__name__)
settings = get_settings()

#_client: AsyncAnthropic | None = None
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY, 
            base_url="https://openrouter.ai/api/v1"
        )
        #_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


_CRITIC_SYSTEM_PROMPT = """You are a strict fact-checker. You will be given a set \
of source excerpts and an answer that claims to be grounded in those excerpts. \
Check whether EVERY factual claim in the answer is directly supported by at least \
one excerpt. A claim is unsupported if it adds information, numbers, names, or \
conclusions not present in the excerpts — including reasonable-sounding inferences \
the excerpts don't actually state.

Respond with ONLY a JSON object, no markdown fences, no other text:
{
  "result": "PASS" or "FAIL",
  "unsupported_claims": ["claim text", ...],
  "reasoning": "one sentence explaining the verdict"
}

If the answer explicitly says it cannot answer from the given sources, that is a PASS —
declining to answer is never a hallucination."""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


async def check_groundedness(
    answer: str,
    chunks_used: list[dict],
) -> dict:
    """
    Returns:
      {"result": "PASS" | "FAIL", "unsupported_claims": [...], "reasoning": str}

    On any failure to get a parseable verdict from the LLM, fails CLOSED —
    treats it as FAIL with reasoning noting the parse failure. An unparseable
    critic response must never be silently treated as a PASS; that would
    defeat the entire purpose of having a critic.
    """
    if not chunks_used:
        # Nothing was retrieved at all — the only way the answer can be
        # grounded is if it explicitly declined. Still run the check
        # rather than special-casing it, since the answer might falsely
        # claim something despite empty context.
        sources_text = "(no source excerpts were available)"
    else:
        sources_text = "\n\n".join(
            f"[{i}] {c['text']}" for i, c in enumerate(chunks_used, start=1)
        )

    user_message = (
        f"### Source excerpts:\n{sources_text}\n\n"
        f"### Answer to check:\n{answer}"
    )

    try:
        client = _get_client()
        # response = await client.messages.create(
        #     model=settings.LLM_MODEL,
        #     max_tokens=1024,
        #     temperature=0,  # deterministic — this is a check, not a creative task
        #     system=_CRITIC_SYSTEM_PROMPT,
        #     messages=[{"role": "user", "content": user_message}],
        # )
        # raw_text = response.content[0].text
        response = await client.chat.completions.create(
            model=settings.CRITIC_LLM_MODEL,
            max_tokens=1024,
            temperature=0,  # deterministic — this is a check, not a creative task
            messages=[
                {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
        )
        raw_text = response.choices[0].message.content
        cleaned = _strip_json_fences(raw_text)
        verdict = json.loads(cleaned)

        if verdict.get("result") not in ("PASS", "FAIL"):
            raise ValueError(f"Invalid result field: {verdict.get('result')}")

        logger.info(
            "critic_check_complete",
            result=verdict["result"],
            unsupported_count=len(verdict.get("unsupported_claims", [])),
        )

        return verdict

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # Fail CLOSED — an unparseable verdict is treated as FAIL, never PASS.
        logger.error("critic_response_unparseable", error=str(e))
        return {
            "result": "FAIL",
            "unsupported_claims": [],
            "reasoning": "Critic response could not be parsed — failing closed.",
        }
    except Exception as e:
        logger.error("critic_check_failed", error=str(e))
        raise GenerationError() from e