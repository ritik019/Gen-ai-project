from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    location: str = Field(..., min_length=1, description="City area or locality name")
    price_range: list[str] | None = Field(
        default=None,
        description='Price buckets to include, e.g. ["$", "$$"]',
    )
    min_rating: float = Field(default=0.0, ge=0.0, le=5.0)
    cuisines: list[str] = Field(default_factory=list)
    free_text_preferences: str | None = Field(
        default=None, description="Free-text hints for Phase 3 LLM re-ranking"
    )
    limit: int = Field(default=10, ge=1, le=50)


class RestaurantOut(BaseModel):
    id: str
    name: str
    address: str
    city: str
    locality: str
    price_bucket: str
    avg_cost_for_two: float | None
    avg_rating: float | None
    cuisines: list[str]


class RecommendationItem(BaseModel):
    restaurant: RestaurantOut
    score: float
    reason: str | None = None


class RecommendationResponse(BaseModel):
    recommendations: list[RecommendationItem]
    total_candidates: int
    variant: str | None = None


class FeedbackRequest(BaseModel):
    restaurant_id: str = Field(..., min_length=1)
    query_location: str = Field(..., min_length=1)
    is_positive: bool
    variant: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    total_feedback: int
