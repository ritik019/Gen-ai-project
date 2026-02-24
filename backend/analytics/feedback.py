from __future__ import annotations

import time
from typing import Any

_feedback: list[dict[str, Any]] = []


def record_feedback(
    restaurant_id: str,
    query_location: str,
    is_positive: bool,
    variant: str | None = None,
) -> None:
    _feedback.append({
        "restaurant_id": restaurant_id,
        "query_location": query_location,
        "is_positive": is_positive,
        "variant": variant,
        "timestamp": time.time(),
    })


def get_feedback() -> list[dict[str, Any]]:
    return _feedback


def clear_feedback() -> None:
    _feedback.clear()
