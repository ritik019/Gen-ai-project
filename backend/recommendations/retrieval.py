from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ..ab_testing.experiments import assign_variant, get_variant_weights
from ..analytics.store import record_event
from ..embeddings.encoder import encode_text
from ..llm.groq_client import rank_and_explain
from .cache import cache_get, cache_set
from .data_store import get_dataframe, get_embeddings
from .models import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    RestaurantOut,
)

PRICE_ORDER = ["$", "$$", "$$$", "$$$$"]


def _price_distance(a: str, b: str) -> int:
    """Return how many bucket steps apart two price buckets are."""
    try:
        return abs(PRICE_ORDER.index(a) - PRICE_ORDER.index(b))
    except ValueError:
        return 0


def _score_row(
    row: pd.Series,
    requested_cuisines_lower: set[str],
    requested_buckets: list[str] | None,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute a heuristic score for a single restaurant row."""
    w = weights or {"rating": 0.6, "cuisine": 0.3, "price": 0.1}

    rating = row.get("avg_rating")
    rating_score = (float(rating) / 5.0) if pd.notna(rating) else 0.0

    if requested_cuisines_lower:
        restaurant_cuisines: list[str] = row.get("cuisines_list", [])
        matches = sum(1 for c in restaurant_cuisines if c in requested_cuisines_lower)
        cuisine_score = matches / len(requested_cuisines_lower)
    else:
        cuisine_score = 1.0

    if requested_buckets:
        bucket = row.get("price_bucket", "")
        distances = [_price_distance(bucket, b) for b in requested_buckets]
        min_dist = min(distances) if distances else 0
        price_score = max(0.0, 1.0 - min_dist * 0.5)
    else:
        price_score = 1.0

    return w["rating"] * rating_score + w["cuisine"] * cuisine_score + w["price"] * price_score


def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    start_time = time.time()

    # --- A/B variant assignment ---
    variant = assign_variant()
    weights = get_variant_weights(variant)

    # --- Cache check ---
    request_dict = request.model_dump()
    request_dict["_variant"] = variant  # include variant in cache key
    cached = cache_get(request_dict)
    if cached is not None:
        elapsed_ms = round((time.time() - start_time) * 1000, 1)
        record_event("search", {
            "location": request.location,
            "cuisines": request.cuisines,
            "price_range": request.price_range,
            "min_rating": request.min_rating,
            "free_text": request.free_text_preferences,
            "total_candidates": cached.total_candidates,
            "results_returned": len(cached.recommendations),
            "response_time_ms": elapsed_ms,
            "cache_hit": True,
        })
        return cached

    df = get_dataframe()

    # --- Hard filters ---
    if request.location:
        location_lower = request.location.strip().lower()
        mask = df["city_lower"].str.contains(location_lower, na=False) | df[
            "locality_lower"
        ].str.contains(location_lower, na=False)
    else:
        mask = pd.Series(True, index=df.index)

    if request.price_range:
        mask = mask & df["price_bucket"].isin(request.price_range)

    if request.min_rating > 0:
        mask = mask & (df["avg_rating"] >= request.min_rating)

    if request.cuisines:
        req_cuisines_lower = {c.strip().lower() for c in request.cuisines}
        mask = mask & df["cuisines_list"].apply(
            lambda cl: bool(req_cuisines_lower & set(cl))
        )
    else:
        req_cuisines_lower = set()

    candidates = df.loc[mask].copy()
    total_candidates = len(candidates)

    if candidates.empty:
        elapsed_ms = round((time.time() - start_time) * 1000, 1)
        record_event("search", {
            "location": request.location,
            "cuisines": request.cuisines,
            "price_range": request.price_range,
            "min_rating": request.min_rating,
            "free_text": request.free_text_preferences,
            "total_candidates": 0,
            "results_returned": 0,
            "response_time_ms": elapsed_ms,
            "cache_hit": False,
        })
        return RecommendationResponse(
            recommendations=[], total_candidates=0, variant=variant,
        )

    # --- Scoring ---
    candidates["_score"] = candidates.apply(
        _score_row,
        axis=1,
        requested_cuisines_lower=req_cuisines_lower,
        requested_buckets=request.price_range,
        weights=weights,
    )

    # --- Semantic scoring (when free_text_preferences provided) ---
    all_embeddings = get_embeddings()
    if request.free_text_preferences and all_embeddings is not None:
        user_vec = encode_text(request.free_text_preferences).reshape(1, -1)
        candidate_indices = candidates.index.tolist()
        candidate_vecs = all_embeddings[candidate_indices]
        sim_scores = cosine_similarity(user_vec, candidate_vecs).flatten()
        # Normalise cosine similarity from [-1, 1] to [0, 1]
        sim_normalised = (sim_scores + 1.0) / 2.0
        candidates["_semantic"] = sim_normalised
        # Blend: 50% heuristic + 50% semantic
        candidates["_score"] = 0.5 * candidates["_score"] + 0.5 * candidates["_semantic"]

    top = candidates.nlargest(request.limit, "_score")

    # --- LLM re-ranking & explanation ---
    candidate_dicts: list[dict] = []
    for _, row in top.iterrows():
        candidate_dicts.append({
            "id": str(row["id"]),
            "name": row["name"],
            "price_bucket": row["price_bucket"],
            "avg_rating": row["avg_rating"] if pd.notna(row["avg_rating"]) else None,
            "cuisines": row["cuisines_list"],
        })

    preferences = {
        "location": request.location,
        "price_range": request.price_range,
        "min_rating": request.min_rating,
        "cuisines": request.cuisines,
        "free_text_preferences": request.free_text_preferences,
    }

    llm_results = rank_and_explain(preferences, candidate_dicts)

    # --- Assemble response ---
    # Build a lookup from the top candidates for reordering
    row_by_id: dict[str, pd.Series] = {}
    for _, row in top.iterrows():
        row_by_id[str(row["id"])] = row

    # If LLM returned results, use its ordering; otherwise keep heuristic order
    if llm_results:
        ordered_ids = [rid for rid in llm_results if rid in row_by_id]
        # Append any candidates the LLM missed
        for rid in row_by_id:
            if rid not in llm_results:
                ordered_ids.append(rid)
    else:
        ordered_ids = list(row_by_id.keys())

    items: list[RecommendationItem] = []
    for rid in ordered_ids:
        row = row_by_id[rid]
        restaurant = RestaurantOut(
            id=str(row["id"]),
            name=row["name"],
            address=row["address"],
            city=row["city"],
            locality=row["locality"],
            price_bucket=row["price_bucket"],
            avg_cost_for_two=row["avg_cost_for_two"] if pd.notna(row["avg_cost_for_two"]) else None,
            avg_rating=row["avg_rating"] if pd.notna(row["avg_rating"]) else None,
            cuisines=row["cuisines_list"],
        )
        items.append(RecommendationItem(
            restaurant=restaurant,
            score=round(float(row["_score"]), 4),
            reason=llm_results.get(rid),
        ))

    response = RecommendationResponse(
        recommendations=items,
        total_candidates=total_candidates,
        variant=variant,
    )

    cache_set(request_dict, response)

    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    record_event("search", {
        "location": request.location,
        "cuisines": request.cuisines,
        "price_range": request.price_range,
        "min_rating": request.min_rating,
        "free_text": request.free_text_preferences,
        "total_candidates": total_candidates,
        "results_returned": len(items),
        "response_time_ms": elapsed_ms,
        "cache_hit": False,
    })

    return response
