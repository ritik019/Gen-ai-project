from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .recommendations.data_store import get_dataframe
from .recommendations.models import RecommendationRequest, RecommendationResponse
from .recommendations.retrieval import get_recommendations

app = FastAPI(title="Restaurant Recommendation API", version="0.5.0")

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


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def root():
    return FileResponse(str(_STATIC_DIR / "index.html"))
