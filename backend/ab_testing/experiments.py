"""
A/B Testing Framework
=====================

This module manages the **scoring-weights experiment** that powers the
recommendation engine's ranking algorithm.

What the experiment tests
-------------------------
The recommendation pipeline scores each restaurant using three factors:
**rating**, **cuisine match**, and **price alignment**.  The relative
importance (weight) of each factor is controlled by the A/B variant:

* **Variant A – "Rating-heavy" (control)**
  ``0.6 × rating  +  0.3 × cuisine  +  0.1 × price``
  Prioritises highly-rated restaurants.

* **Variant B – "Price-balanced" (treatment)**
  ``0.4 × rating  +  0.3 × cuisine  +  0.3 × price``
  Gives more importance to price alignment with user preferences.

How variant assignment works
----------------------------
On the **first search** a user makes, a variant is randomly assigned
(50/50 split) and stored in their session cookie.  Every subsequent
search by that user uses the **same variant** so the experience stays
consistent and measurement is clean.

How feedback determines a winner
--------------------------------
Users can give **thumbs-up / thumbs-down** on each recommendation.
Each piece of feedback is tagged with the variant the user was assigned.
We compute a **satisfaction rate** per variant:

    satisfaction = positive_feedback / total_feedback × 100

If one variant's satisfaction rate exceeds the other by **>= 5
percentage points**, it is declared the **winner**.
"""

from __future__ import annotations

import contextvars
import random
import time
from typing import Any

# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

EXPERIMENTS: dict[str, dict] = {
    "scoring_weights": {
        "name": "Scoring Weight Optimization",
        "description": "Test if higher price alignment weight improves satisfaction",
        "variants": {
            "A": {
                "label": "Rating-heavy (control)",
                "weights": {"rating": 0.6, "cuisine": 0.3, "price": 0.1},
            },
            "B": {
                "label": "Price-balanced (treatment)",
                "weights": {"rating": 0.4, "cuisine": 0.3, "price": 0.3},
            },
        },
        "active": True,
    },
}

# ---------------------------------------------------------------------------
# Session-variant context variable
# ---------------------------------------------------------------------------
# The endpoint handler calls ``set_session_variant()`` before invoking the
# recommendation pipeline.  ``assign_variant()`` reads this value so that
# retrieval.py (which we must NOT modify) transparently reuses the
# session-persisted variant instead of picking a new random one.

_session_variant_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_session_variant_ctx", default=None,
)


def set_session_variant(variant: str | None) -> None:
    """Set the session variant in the current request context."""
    _session_variant_ctx.set(variant)


# ---------------------------------------------------------------------------
# Variant assignment
# ---------------------------------------------------------------------------

_assignments: list[dict[str, Any]] = []


def assign_variant(experiment_id: str = "scoring_weights") -> str:
    """Return the A/B variant for the current request.

    1. If the session already has a variant (set via ``set_session_variant``),
       return it immediately — this keeps the user's experience consistent.
    2. Otherwise pick a variant at random (50/50) and log the assignment.
    """
    experiment = EXPERIMENTS.get(experiment_id)
    if not experiment or not experiment["active"]:
        return "A"

    # Reuse session variant when available
    session_variant = _session_variant_ctx.get()
    if session_variant in ("A", "B"):
        return session_variant

    # New random assignment
    variant = random.choice(["A", "B"])
    _assignments.append({
        "experiment_id": experiment_id,
        "variant": variant,
        "timestamp": time.time(),
    })
    return variant


def get_variant_weights(
    variant: str, experiment_id: str = "scoring_weights",
) -> dict[str, float]:
    """Return the scoring weights dict for the given variant."""
    experiment = EXPERIMENTS.get(experiment_id, {})
    variants = experiment.get("variants", {})
    default = {"rating": 0.6, "cuisine": 0.3, "price": 0.1}
    return variants.get(variant, variants.get("A", {})).get("weights", default)


def get_assignments() -> list[dict[str, Any]]:
    return _assignments


def clear_assignments() -> None:
    _assignments.clear()


# ---------------------------------------------------------------------------
# Per-variant stats (searches + feedback)
# ---------------------------------------------------------------------------

_variant_stats: dict[str, dict[str, int]] = {
    "A": {"searches": 0, "feedback_positive": 0, "feedback_negative": 0},
    "B": {"searches": 0, "feedback_positive": 0, "feedback_negative": 0},
}


def record_variant_search(variant: str) -> None:
    """Increment the search counter for *variant*."""
    if variant in _variant_stats:
        _variant_stats[variant]["searches"] += 1


def record_variant_feedback(variant: str, is_positive: bool) -> None:
    """Increment the positive or negative feedback counter for *variant*."""
    if variant in _variant_stats:
        key = "feedback_positive" if is_positive else "feedback_negative"
        _variant_stats[variant][key] += 1


def get_variant_stats() -> dict[str, Any]:
    """Return per-variant stats with satisfaction rates and winner.

    The winner is the variant whose satisfaction rate exceeds the other
    by at least 5 percentage points (and both must have at least one
    piece of feedback).
    """
    result: dict[str, Any] = {}
    for v in ("A", "B"):
        s = _variant_stats[v]
        total_fb = s["feedback_positive"] + s["feedback_negative"]
        rate = round(s["feedback_positive"] / total_fb * 100, 1) if total_fb > 0 else 0.0
        result[v] = {
            "searches": s["searches"],
            "feedback_positive": s["feedback_positive"],
            "feedback_negative": s["feedback_negative"],
            "total_feedback": total_fb,
            "satisfaction_rate": rate,
        }

    rate_a = result["A"]["satisfaction_rate"]
    rate_b = result["B"]["satisfaction_rate"]
    has_data = result["A"]["total_feedback"] > 0 and result["B"]["total_feedback"] > 0
    if has_data and abs(rate_a - rate_b) >= 5.0:
        result["winner"] = "A" if rate_a > rate_b else "B"
    else:
        result["winner"] = None

    return result


def clear_variant_stats() -> None:
    for v in _variant_stats.values():
        v["searches"] = 0
        v["feedback_positive"] = 0
        v["feedback_negative"] = 0
