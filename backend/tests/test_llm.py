import json
from unittest.mock import MagicMock, patch

from backend.llm.config import LLMConfig
from backend.llm.groq_client import rank_and_explain

SAMPLE_CANDIDATES = [
    {"id": "1", "name": "Spice House", "price_bucket": "$$", "avg_rating": 4.2, "cuisines": ["north indian", "chinese"]},
    {"id": "2", "name": "Pasta Palace", "price_bucket": "$$$", "avg_rating": 4.5, "cuisines": ["italian"]},
    {"id": "3", "name": "Curry Leaf", "price_bucket": "$", "avg_rating": 3.9, "cuisines": ["south indian"]},
]

SAMPLE_PREFERENCES = {
    "location": "BTM",
    "cuisines": ["Italian"],
    "free_text_preferences": "cozy date night",
}

ENABLED_CONFIG = LLMConfig(api_key="test-key", enabled=True)
DISABLED_CONFIG = LLMConfig(api_key="test-key", enabled=False)


def _mock_groq_response(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("backend.llm.groq_client.Groq")
def test_rank_and_explain_returns_reasons(mock_groq_cls):
    llm_response = json.dumps({
        "recommendations": [
            {"id": "2", "reason": "Perfect Italian cuisine for a romantic evening."},
            {"id": "1", "reason": "Great Chinese and Indian options nearby."},
            {"id": "3", "reason": "Affordable South Indian with solid ratings."},
        ]
    })
    mock_groq_cls.return_value.chat.completions.create.return_value = _mock_groq_response(llm_response)

    result = rank_and_explain(SAMPLE_PREFERENCES, SAMPLE_CANDIDATES, config=ENABLED_CONFIG)

    assert result["2"] == "Perfect Italian cuisine for a romantic evening."
    assert result["1"] == "Great Chinese and Indian options nearby."
    assert result["3"] == "Affordable South Indian with solid ratings."


@patch("backend.llm.groq_client.Groq")
def test_rank_and_explain_fallback_on_api_error(mock_groq_cls):
    mock_groq_cls.return_value.chat.completions.create.side_effect = Exception("API timeout")

    result = rank_and_explain(SAMPLE_PREFERENCES, SAMPLE_CANDIDATES, config=ENABLED_CONFIG)

    assert result == {}


@patch("backend.llm.groq_client.Groq")
def test_rank_and_explain_fallback_on_bad_json(mock_groq_cls):
    mock_groq_cls.return_value.chat.completions.create.return_value = _mock_groq_response("not valid json{{{")

    result = rank_and_explain(SAMPLE_PREFERENCES, SAMPLE_CANDIDATES, config=ENABLED_CONFIG)

    assert result == {}


def test_rank_and_explain_disabled():
    result = rank_and_explain(SAMPLE_PREFERENCES, SAMPLE_CANDIDATES, config=DISABLED_CONFIG)

    assert result == {}


def test_rank_and_explain_empty_candidates():
    result = rank_and_explain(SAMPLE_PREFERENCES, [], config=ENABLED_CONFIG)

    assert result == {}
