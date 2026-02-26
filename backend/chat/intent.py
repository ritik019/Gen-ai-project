from __future__ import annotations

import json
import logging
import re
from typing import Any

from groq import Groq

from ..llm.config import DEFAULT_LLM_CONFIG, LLMConfig
from ..recommendations.models import RecommendationRequest
from .models import ConversationState, ConversationTurn, ExtractedIntent

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.6
MAX_CLARIFICATIONS = 1
_MAX_TURNS = 6  # 3 exchanges

# ---------------------------------------------------------------------------
# LLM Prompts
# ---------------------------------------------------------------------------

INTENT_EXTRACTION_PROMPT = """\
You are a restaurant intent parser. Given a user message (and optionally prior \
conversation context), extract structured dining preferences as JSON.

Return ONLY valid JSON with these fields (omit fields you cannot infer):
{
  "location": "area or locality name",
  "cuisines": ["Cuisine1", "Cuisine2"],
  "price_sentiment": "raw price expression e.g. 'under 500', 'cheap', 'fine dining'",
  "min_rating": 4.0,
  "mood": "romantic / casual / lively / quiet / cozy",
  "occasion": "date night / birthday / work lunch / family dinner / friends hangout",
  "group_size": "couple / small group / large group / solo",
  "dietary": ["vegetarian", "vegan"],
  "vibe": "rooftop / outdoor / cafe / bar / fine dining / street food",
  "time_context": "tonight / lunch / weekend brunch",
  "confidence": 0.7,
  "missing_fields": ["location"]
}

Confidence scoring rules:
- Very vague query with no specifics → confidence 0.2–0.3, list missing_fields
- Location alone → confidence 0.5
- Location + one other signal (cuisine, mood, occasion) → confidence 0.7
- Location + two or more signals → confidence 0.8–0.9
- Highly specific with location + cuisine + mood/occasion → confidence 0.95

Normalize cuisines to Title Case. Keep raw price expressions as-is.
Always include confidence and missing_fields in your response."""

CLARIFICATION_PROMPT = """\
You are a friendly AI dining assistant. The user wants restaurant recommendations \
but their request is missing some key details.

Based on what we know so far, ask a brief, conversational follow-up question \
(under 50 words) to clarify at most 2 missing details. Prioritize asking about \
location first if missing, then cuisine or occasion.

Do NOT list options in bullet points. Keep it natural and warm."""


# ---------------------------------------------------------------------------
# Price Mapping
# ---------------------------------------------------------------------------

_PRICE_KEYWORDS: dict[str, list[str]] = {
    "cheap": ["$"],
    "budget": ["$"],
    "affordable": ["$", "$$"],
    "inexpensive": ["$"],
    "moderate": ["$$"],
    "mid-range": ["$$"],
    "mid range": ["$$"],
    "expensive": ["$$$", "$$$$"],
    "fine dining": ["$$$", "$$$$"],
    "premium": ["$$$", "$$$$"],
    "luxury": ["$$$$"],
    "splurge": ["$$$", "$$$$"],
}

_AMOUNT_RE = re.compile(
    r"(?:under|below|less than|max|upto|up to)\s*(?:₹|rs\.?|inr)?\s*(\d+)",
    re.IGNORECASE,
)
_PER_PERSON_RE = re.compile(r"per\s*person", re.IGNORECASE)


def _map_price_sentiment_to_buckets(sentiment: str | None) -> list[str] | None:
    if not sentiment:
        return None

    lower = sentiment.lower().strip()

    # Check keyword matches first
    for keyword, buckets in _PRICE_KEYWORDS.items():
        if keyword in lower:
            return buckets

    # Try numeric extraction: "under 500 per person" or "under 1000"
    match = _AMOUNT_RE.search(lower)
    if match:
        amount = int(match.group(1))
        # If "per person", multiply by 2 for avg_cost_for_two
        if _PER_PERSON_RE.search(lower):
            amount *= 2
        # Map amount (for two) to price buckets
        if amount <= 500:
            return ["$"]
        elif amount <= 1000:
            return ["$", "$$"]
        elif amount <= 2000:
            return ["$", "$$", "$$$"]
        else:
            return ["$", "$$", "$$$", "$$$$"]

    return None


# ---------------------------------------------------------------------------
# Intent → RecommendationRequest
# ---------------------------------------------------------------------------


def map_intent_to_request(
    intent: ExtractedIntent,
    original_message: str,
) -> RecommendationRequest:
    price_range = _map_price_sentiment_to_buckets(intent.price_sentiment)

    # Build free_text_preferences from mood/vibe/occasion and original message
    parts: list[str] = []
    if intent.mood:
        parts.append(f"mood: {intent.mood}")
    if intent.occasion:
        parts.append(f"occasion: {intent.occasion}")
    if intent.vibe:
        parts.append(f"vibe: {intent.vibe}")
    if intent.time_context:
        parts.append(f"time: {intent.time_context}")
    if intent.group_size:
        parts.append(f"group: {intent.group_size}")
    if intent.dietary:
        parts.append(f"dietary: {', '.join(intent.dietary)}")
    parts.append(original_message)

    free_text = " | ".join(parts)

    return RecommendationRequest(
        location=intent.location,
        price_range=price_range,
        min_rating=intent.min_rating or 0.0,
        cuisines=intent.cuisines,
        free_text_preferences=free_text,
    )


# ---------------------------------------------------------------------------
# Conversation Accumulation
# ---------------------------------------------------------------------------


def accumulate_intent(
    accumulated: dict[str, Any],
    new_intent: ExtractedIntent,
) -> dict[str, Any]:
    new_data = new_intent.model_dump(exclude_none=True, exclude={"confidence", "missing_fields"})

    for key, value in new_data.items():
        if isinstance(value, list) and isinstance(accumulated.get(key), list):
            # Union for lists (cuisines, dietary)
            existing = set(accumulated[key])
            existing.update(value)
            accumulated[key] = sorted(existing)
        elif value:  # Only overwrite with truthy values
            accumulated[key] = value

    return accumulated


def update_conversation_state(
    state: ConversationState,
    user_message: str,
    assistant_message: str,
    new_intent: ExtractedIntent,
    result_ids: list[str] | None = None,
) -> ConversationState:
    turns = list(state.turns)
    turns.append(ConversationTurn(role="user", content=user_message))
    turns.append(ConversationTurn(role="assistant", content=assistant_message))

    # Keep only last _MAX_TURNS messages
    if len(turns) > _MAX_TURNS:
        turns = turns[-_MAX_TURNS:]

    accumulated = accumulate_intent(dict(state.accumulated_intent), new_intent)

    return ConversationState(
        turns=turns,
        accumulated_intent=accumulated,
        clarification_count=state.clarification_count,
        last_results_ids=result_ids or state.last_results_ids,
    )


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


def _fallback_intent(message: str) -> ExtractedIntent:
    return ExtractedIntent(confidence=0.0, missing_fields=["location", "cuisines"])


_FALLBACK_QUESTIONS = [
    "Which area or neighborhood are you looking to eat in?",
    "Any particular cuisine you're in the mood for?",
    "What's the occasion — casual outing, date night, or something else?",
]


def _fallback_clarification(intent: ExtractedIntent) -> str:
    if not intent.location and not intent.cuisines:
        return "I'd love to help! Which area are you looking to eat in, and do you have a cuisine preference?"
    if not intent.location:
        return "Sounds great! Which area or neighborhood should I search in?"
    return "Nice choice! Any particular cuisine or vibe you're going for?"


# ---------------------------------------------------------------------------
# LLM Calls
# ---------------------------------------------------------------------------


def extract_intent(
    message: str,
    conversation_state: ConversationState | None = None,
    config: LLMConfig = DEFAULT_LLM_CONFIG,
) -> ExtractedIntent:
    if not config.enabled or not config.api_key:
        return _fallback_intent(message)

    try:
        # Build context from conversation history
        messages: list[dict[str, str]] = [
            {"role": "system", "content": INTENT_EXTRACTION_PROMPT},
        ]

        if conversation_state and conversation_state.turns:
            history_parts = []
            for turn in conversation_state.turns[-4:]:  # Last 2 exchanges
                history_parts.append(f"{turn.role}: {turn.content}")
            if conversation_state.accumulated_intent:
                history_parts.append(
                    f"Known preferences so far: {json.dumps(conversation_state.accumulated_intent)}"
                )
            context = "\n".join(history_parts)
            messages.append(
                {"role": "user", "content": f"Conversation context:\n{context}\n\nLatest message: {message}"}
            )
        else:
            messages.append({"role": "user", "content": message})

        client = Groq(api_key=config.api_key, timeout=config.timeout)
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        return ExtractedIntent(**parsed)

    except Exception:
        logger.warning("Intent extraction failed, using fallback", exc_info=True)
        return _fallback_intent(message)


def generate_clarification(
    intent: ExtractedIntent,
    config: LLMConfig = DEFAULT_LLM_CONFIG,
) -> str:
    if not config.enabled or not config.api_key:
        return _fallback_clarification(intent)

    try:
        known = intent.model_dump(exclude_none=True, exclude={"confidence", "missing_fields"})
        missing = intent.missing_fields or []

        user_content = (
            f"Known preferences: {json.dumps(known)}\n"
            f"Missing information: {', '.join(missing) if missing else 'unclear'}\n"
            "Generate a brief follow-up question."
        )

        client = Groq(api_key=config.api_key, timeout=config.timeout)
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": CLARIFICATION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=128,
            temperature=0.5,
        )

        question = (response.choices[0].message.content or "").strip()
        return question if question else _fallback_clarification(intent)

    except Exception:
        logger.warning("Clarification generation failed, using fallback", exc_info=True)
        return _fallback_clarification(intent)
