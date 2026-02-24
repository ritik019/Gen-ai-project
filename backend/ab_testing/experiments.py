from __future__ import annotations

import random
import time
from typing import Any

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

_assignments: list[dict[str, Any]] = []


def assign_variant(experiment_id: str = "scoring_weights") -> str:
    experiment = EXPERIMENTS.get(experiment_id)
    if not experiment or not experiment["active"]:
        return "A"
    variant = random.choice(["A", "B"])
    _assignments.append({
        "experiment_id": experiment_id,
        "variant": variant,
        "timestamp": time.time(),
    })
    return variant


def get_variant_weights(
    variant: str, experiment_id: str = "scoring_weights"
) -> dict[str, float]:
    experiment = EXPERIMENTS.get(experiment_id, {})
    variants = experiment.get("variants", {})
    default = {"rating": 0.6, "cuisine": 0.3, "price": 0.1}
    return variants.get(variant, variants.get("A", {})).get("weights", default)


def get_assignments() -> list[dict[str, Any]]:
    return _assignments


def clear_assignments() -> None:
    _assignments.clear()
