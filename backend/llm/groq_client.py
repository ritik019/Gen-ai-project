from __future__ import annotations

import json
import logging
from typing import Any

from groq import Groq

from .config import DEFAULT_LLM_CONFIG, LLMConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a restaurant recommendation engine. "
    "Given user preferences and a list of candidate restaurants, "
    "re-rank them from best to worst match and provide a short, "
    "friendly one-sentence explanation for each.\n\n"
    "Return ONLY valid JSON in this exact format:\n"
    '{"recommendations": [{"id": "<restaurant_id>", "reason": "<one sentence>"}]}\n'
    "Include only restaurants from the provided list. "
    "Order from best match to worst."
)


def _build_user_message(
    preferences: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> str:
    lines = ["## User Preferences"]
    if preferences.get("location"):
        lines.append(f"- Location: {preferences['location']}")
    if preferences.get("price_range"):
        lines.append(f"- Price range: {', '.join(preferences['price_range'])}")
    if preferences.get("min_rating"):
        lines.append(f"- Minimum rating: {preferences['min_rating']}")
    if preferences.get("cuisines"):
        lines.append(f"- Cuisines: {', '.join(preferences['cuisines'])}")
    if preferences.get("free_text_preferences"):
        lines.append(f"- Special preferences: {preferences['free_text_preferences']}")

    lines.append("\n## Candidate Restaurants")
    lines.append("| ID | Name | Price | Rating | Cuisines |")
    lines.append("|---|---|---|---|---|")
    for c in candidates:
        cuisines_str = ", ".join(c.get("cuisines", []))
        lines.append(
            f"| {c['id']} | {c['name']} | {c.get('price_bucket', '?')} "
            f"| {c.get('avg_rating', 'N/A')} | {cuisines_str} |"
        )

    return "\n".join(lines)


def rank_and_explain(
    preferences: dict[str, Any],
    candidates: list[dict[str, Any]],
    config: LLMConfig = DEFAULT_LLM_CONFIG,
) -> dict[str, str]:
    """
    Call Groq LLM to re-rank candidates and generate explanations.

    Returns a dict mapping restaurant id -> reason string.
    Returns empty dict on any failure (timeout, bad JSON, API error).
    """
    if not config.enabled or not config.api_key:
        return {}

    if not candidates:
        return {}

    try:
        client = Groq(api_key=config.api_key, timeout=config.timeout)
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_message(preferences, candidates),
                },
            ],
            max_tokens=config.max_tokens,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or ""
        parsed = json.loads(content)

        results: dict[str, str] = {}
        for item in parsed.get("recommendations", []):
            rid = str(item.get("id", ""))
            reason = item.get("reason", "")
            if rid and reason:
                results[rid] = reason

        return results

    except Exception:
        logger.warning("Groq LLM call failed, falling back to heuristic ranking", exc_info=True)
        return {}
