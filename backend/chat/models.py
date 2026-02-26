from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from ..recommendations.models import RecommendationResponse


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)


class ChatResponseType(str, Enum):
    results = "results"
    clarification = "clarification"


class ChatResponse(BaseModel):
    type: ChatResponseType
    message: str
    results: RecommendationResponse | None = None
    parsed_intent: dict = Field(default_factory=dict)


class ExtractedIntent(BaseModel):
    location: str | None = None
    cuisines: list[str] = Field(default_factory=list)
    price_sentiment: str | None = None
    min_rating: float | None = None
    mood: str | None = None
    occasion: str | None = None
    group_size: str | None = None
    dietary: list[str] = Field(default_factory=list)
    vibe: str | None = None
    time_context: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    role: str
    content: str


class ConversationState(BaseModel):
    turns: list[ConversationTurn] = Field(default_factory=list)
    accumulated_intent: dict = Field(default_factory=dict)
    clarification_count: int = 0
    last_results_ids: list[str] = Field(default_factory=list)
