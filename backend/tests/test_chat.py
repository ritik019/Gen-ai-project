from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.app import app
from backend.chat.intent import (
    _map_price_sentiment_to_buckets,
    accumulate_intent,
    extract_intent,
    generate_clarification,
    map_intent_to_request,
)
from backend.chat.models import ConversationState, ExtractedIntent

client = TestClient(app)


def _login(c):
    c.post("/auth/login", json={"username": "user", "password": "user123"})


# ── Price Mapping ────────────────────────────────────────────────────────


class TestPriceMapping:
    def test_cheap_keyword(self):
        assert _map_price_sentiment_to_buckets("cheap") == ["$"]

    def test_moderate_keyword(self):
        assert _map_price_sentiment_to_buckets("moderate") == ["$$"]

    def test_expensive_keyword(self):
        assert _map_price_sentiment_to_buckets("fine dining") == ["$$$", "$$$$"]

    def test_per_person_amount(self):
        result = _map_price_sentiment_to_buckets("under 500 per person")
        assert result == ["$", "$$"]  # 500*2=1000 for two → ≤1000 → $,$$

    def test_amount_for_two(self):
        result = _map_price_sentiment_to_buckets("under 500")
        assert result == ["$"]  # 500 for two → ≤500 → $

    def test_none_input(self):
        assert _map_price_sentiment_to_buckets(None) is None

    def test_unknown_expression(self):
        assert _map_price_sentiment_to_buckets("somewhere nice") is None


# ── Intent → RecommendationRequest ───────────────────────────────────────


class TestIntentMapping:
    def test_full_intent_mapping(self):
        intent = ExtractedIntent(
            location="Koramangala",
            cuisines=["Italian", "Chinese"],
            price_sentiment="moderate",
            min_rating=4.0,
            mood="romantic",
            occasion="date night",
            vibe="rooftop",
            confidence=0.9,
        )
        req = map_intent_to_request(intent, "romantic Italian dinner in Koramangala")
        assert req.location == "Koramangala"
        assert req.cuisines == ["Italian", "Chinese"]
        assert req.price_range == ["$$"]
        assert req.min_rating == 4.0
        assert "mood: romantic" in req.free_text_preferences
        assert "occasion: date night" in req.free_text_preferences
        assert "vibe: rooftop" in req.free_text_preferences

    def test_minimal_intent_uses_original_message(self):
        intent = ExtractedIntent(confidence=0.0)
        req = map_intent_to_request(intent, "good food nearby")
        assert req.location is None
        assert req.cuisines == []
        assert req.price_range is None
        assert "good food nearby" in req.free_text_preferences


# ── Accumulation ─────────────────────────────────────────────────────────


class TestAccumulation:
    def test_new_fields_added(self):
        old = {}
        intent = ExtractedIntent(location="BTM", cuisines=["Chinese"], confidence=0.5)
        result = accumulate_intent(old, intent)
        assert result["location"] == "BTM"
        assert result["cuisines"] == ["Chinese"]

    def test_scalar_overwrite(self):
        old = {"location": "BTM", "mood": "casual"}
        intent = ExtractedIntent(location="Koramangala", confidence=0.7)
        result = accumulate_intent(old, intent)
        assert result["location"] == "Koramangala"
        assert result["mood"] == "casual"  # Preserved from old

    def test_list_union(self):
        old = {"cuisines": ["Chinese", "Italian"]}
        intent = ExtractedIntent(cuisines=["Mexican", "Chinese"], confidence=0.6)
        result = accumulate_intent(old, intent)
        assert sorted(result["cuisines"]) == ["Chinese", "Italian", "Mexican"]


# ── Intent Extraction ────────────────────────────────────────────────────


class TestIntentExtraction:
    def test_fallback_when_disabled(self):
        from backend.llm.config import LLMConfig
        config = LLMConfig(enabled=False)
        intent = extract_intent("good food", config=config)
        assert intent.confidence == 0.0

    @patch("backend.chat.intent.Groq")
    def test_successful_extraction(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "location": "Koramangala",
            "cuisines": ["Italian"],
            "mood": "romantic",
            "confidence": 0.85,
            "missing_fields": [],
        })
        mock_client.chat.completions.create.return_value = mock_response

        from backend.llm.config import LLMConfig
        config = LLMConfig(api_key="test-key", enabled=True)
        intent = extract_intent("romantic Italian in Koramangala", config=config)
        assert intent.location == "Koramangala"
        assert intent.cuisines == ["Italian"]
        assert intent.mood == "romantic"
        assert intent.confidence == 0.85

    @patch("backend.chat.intent.Groq")
    def test_fallback_on_api_error(self, mock_groq_cls):
        mock_groq_cls.return_value.chat.completions.create.side_effect = Exception("API error")

        from backend.llm.config import LLMConfig
        config = LLMConfig(api_key="test-key", enabled=True)
        intent = extract_intent("good food", config=config)
        assert intent.confidence == 0.0


# ── Clarification ────────────────────────────────────────────────────────


class TestClarification:
    def test_fallback_no_location_no_cuisine(self):
        intent = ExtractedIntent(confidence=0.3, missing_fields=["location", "cuisines"])
        from backend.llm.config import LLMConfig
        config = LLMConfig(enabled=False)
        question = generate_clarification(intent, config=config)
        assert "area" in question.lower()

    def test_fallback_no_location_with_cuisine(self):
        intent = ExtractedIntent(
            cuisines=["Italian"],
            confidence=0.4,
            missing_fields=["location"],
        )
        from backend.llm.config import LLMConfig
        config = LLMConfig(enabled=False)
        question = generate_clarification(intent, config=config)
        assert "area" in question.lower() or "neighborhood" in question.lower()


# ── Chat Endpoint ────────────────────────────────────────────────────────


class TestChatEndpoint:
    def test_requires_login(self):
        c = TestClient(app)
        resp = c.post("/chat", json={"message": "good food"})
        assert resp.status_code == 401

    @patch("backend.chat.intent.Groq")
    def test_returns_results_for_specific_query(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "location": "BTM",
            "cuisines": ["Chinese"],
            "confidence": 0.8,
            "missing_fields": [],
        })
        mock_client.chat.completions.create.return_value = mock_response

        c = TestClient(app)
        _login(c)
        resp = c.post("/chat", json={"message": "Chinese food in BTM"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "results"
        assert body["results"] is not None
        assert body["parsed_intent"]["location"] == "BTM"

    @patch("backend.chat.intent.Groq")
    def test_returns_clarification_for_vague_query(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        # First call: intent extraction → low confidence
        intent_response = MagicMock()
        intent_response.choices = [MagicMock()]
        intent_response.choices[0].message.content = json.dumps({
            "mood": "good",
            "confidence": 0.3,
            "missing_fields": ["location", "cuisines"],
        })

        # Second call: clarification → question
        clarif_response = MagicMock()
        clarif_response.choices = [MagicMock()]
        clarif_response.choices[0].message.content = "What area would you like to eat in?"

        mock_client.chat.completions.create.side_effect = [
            intent_response,
            clarif_response,
        ]

        c = TestClient(app)
        _login(c)
        resp = c.post("/chat", json={"message": "good food"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "clarification"
        assert body["results"] is None
        assert "area" in body["message"].lower() or "eat" in body["message"].lower()

    @patch("backend.app.extract_intent")
    def test_fallback_on_total_failure(self, mock_extract):
        """When intent extraction fails (confidence=0.0), skip clarification → use raw message."""
        mock_extract.return_value = ExtractedIntent(confidence=0.0, missing_fields=["location"])

        c = TestClient(app)
        _login(c)
        resp = c.post("/chat", json={"message": "food in BTM"})
        assert resp.status_code == 200
        body = resp.json()
        # Should get results (not clarification) due to confidence=0.0 fallback
        assert body["type"] == "results"


# ── Multi-turn Conversation ──────────────────────────────────────────────


class TestConversationSession:
    @patch("backend.chat.intent.Groq")
    def test_multi_turn_accumulates_intent(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        # Turn 1: vague → clarification
        intent_1 = MagicMock()
        intent_1.choices = [MagicMock()]
        intent_1.choices[0].message.content = json.dumps({
            "mood": "romantic",
            "confidence": 0.3,
            "missing_fields": ["location", "cuisines"],
        })
        clarif_1 = MagicMock()
        clarif_1.choices = [MagicMock()]
        clarif_1.choices[0].message.content = "Which area are you looking at?"

        # Turn 2: specific → results
        intent_2 = MagicMock()
        intent_2.choices = [MagicMock()]
        intent_2.choices[0].message.content = json.dumps({
            "location": "BTM",
            "cuisines": ["Italian"],
            "confidence": 0.85,
            "missing_fields": [],
        })

        mock_client.chat.completions.create.side_effect = [
            intent_1,  # Turn 1 intent extraction
            clarif_1,  # Turn 1 clarification
            intent_2,  # Turn 2 intent extraction
        ]

        c = TestClient(app)
        _login(c)

        # Turn 1: vague query → clarification
        resp1 = c.post("/chat", json={"message": "something romantic"})
        assert resp1.json()["type"] == "clarification"

        # Turn 2: follow-up with specifics → results
        resp2 = c.post("/chat", json={"message": "Italian in BTM"})
        body2 = resp2.json()
        assert body2["type"] == "results"
        # Accumulated intent should have both mood and location
        assert body2["parsed_intent"].get("mood") == "romantic"
        assert body2["parsed_intent"].get("location") == "BTM"
