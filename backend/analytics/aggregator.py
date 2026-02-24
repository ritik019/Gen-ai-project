from __future__ import annotations

from collections import Counter
from typing import Any

from .feedback import get_feedback


def compute_analytics(events: list[dict[str, Any]]) -> dict[str, Any]:
    searches = [e for e in events if e["type"] == "search"]
    total = len(searches)

    # Average response time
    times = [s["response_time_ms"] for s in searches if "response_time_ms" in s]
    avg_time = round(sum(times) / len(times), 1) if times else 0.0

    # Top locations
    loc_counter: Counter[str] = Counter()
    for s in searches:
        loc_counter[s.get("location", "unknown")] += 1
    top_locations = [{"name": n, "count": c} for n, c in loc_counter.most_common(10)]

    # Top cuisines
    cuisine_counter: Counter[str] = Counter()
    for s in searches:
        for c in s.get("cuisines", []) or []:
            cuisine_counter[c] += 1
    top_cuisines = [{"name": n, "count": c} for n, c in cuisine_counter.most_common(10)]

    # Price range usage
    price_counter: Counter[str] = Counter()
    for s in searches:
        for p in s.get("price_range", []) or []:
            price_counter[p] += 1
    price_usage = dict(price_counter)

    # Filter usage rates
    filter_counts = {"price_range": 0, "cuisine": 0, "rating": 0, "free_text": 0}
    for s in searches:
        if s.get("price_range"):
            filter_counts["price_range"] += 1
        if s.get("cuisines"):
            filter_counts["cuisine"] += 1
        if s.get("min_rating", 0) > 0:
            filter_counts["rating"] += 1
        if s.get("free_text"):
            filter_counts["free_text"] += 1
    filter_usage = {
        k: round(v / total * 100, 1) if total else 0.0
        for k, v in filter_counts.items()
    }

    # Cache stats
    cache_hits = sum(1 for s in searches if s.get("cache_hit"))
    cache_misses = total - cache_hits

    # Feedback summary
    feedback = get_feedback()
    positive = sum(1 for f in feedback if f["is_positive"])
    negative = len(feedback) - positive

    return {
        "total_searches": total,
        "avg_response_time_ms": avg_time,
        "top_locations": top_locations,
        "top_cuisines": top_cuisines,
        "price_range_usage": price_usage,
        "filter_usage": filter_usage,
        "cache_stats": {
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": round(cache_hits / total * 100, 1) if total else 0.0,
        },
        "feedback_summary": {
            "total": len(feedback),
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": round(positive / len(feedback) * 100, 1) if feedback else 0.0,
        },
    }
