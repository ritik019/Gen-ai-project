from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analytics.aggregator import compute_analytics
from .analytics.feedback import get_feedback, record_feedback
from .analytics.store import get_events
from .ab_testing.experiments import EXPERIMENTS, get_assignments
from .recommendations.cache import get_cache_stats
from .recommendations.data_store import get_dataframe
from .recommendations.models import (
    FeedbackRequest,
    FeedbackResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from .recommendations.retrieval import get_recommendations

app = FastAPI(title="Restaurant Recommendation API", version="1.0.0")

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict:
    df = get_dataframe()
    cities = sorted(df["city"].dropna().unique().tolist())
    cuisines: set[str] = set()
    for val in df["cuisines"].dropna():
        for c in str(val).split(","):
            c = c.strip()
            if c:
                cuisines.add(c)
    return {"cities": cities, "cuisines": sorted(cuisines)}


@app.post("/recommendations", response_model=RecommendationResponse)
def recommendations(request: RecommendationRequest) -> RecommendationResponse:
    return get_recommendations(request)


@app.get("/analytics")
def analytics() -> dict:
    return compute_analytics(get_events())


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    record_feedback(
        request.restaurant_id,
        request.query_location,
        request.is_positive,
        request.variant,
    )
    return FeedbackResponse(status="recorded", total_feedback=len(get_feedback()))


@app.get("/feedback/stats")
def feedback_stats() -> dict:
    fb = get_feedback()
    positive = sum(1 for f in fb if f["is_positive"])
    negative = len(fb) - positive
    return {
        "total": len(fb),
        "positive": positive,
        "negative": negative,
        "satisfaction_rate": round(positive / len(fb) * 100, 1) if fb else 0.0,
    }


@app.get("/cache/stats")
def cache_stats() -> dict:
    return get_cache_stats()


@app.get("/ab-test/results")
def ab_test_results() -> dict:
    fb = get_feedback()
    variant_a = [f for f in fb if f.get("variant") == "A"]
    variant_b = [f for f in fb if f.get("variant") == "B"]
    a_pos = sum(1 for f in variant_a if f["is_positive"])
    b_pos = sum(1 for f in variant_b if f["is_positive"])
    return {
        "experiment": EXPERIMENTS.get("scoring_weights", {}),
        "total_assignments": len(get_assignments()),
        "results": {
            "A": {
                "total_feedback": len(variant_a),
                "positive": a_pos,
                "satisfaction_rate": round(a_pos / len(variant_a) * 100, 1) if variant_a else 0.0,
            },
            "B": {
                "total_feedback": len(variant_b),
                "positive": b_pos,
                "satisfaction_rate": round(b_pos / len(variant_b) * 100, 1) if variant_b else 0.0,
            },
        },
    }


@app.get("/share")
def share():
    return FileResponse(str(_STATIC_DIR / "index.html"))


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(_STATIC_DIR / "index.html"))
