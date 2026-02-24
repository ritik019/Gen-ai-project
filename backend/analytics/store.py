from __future__ import annotations

import time
from typing import Any

_events: list[dict[str, Any]] = []


def record_event(event_type: str, data: dict[str, Any]) -> None:
    _events.append({
        "type": event_type,
        "timestamp": time.time(),
        **data,
    })


def get_events() -> list[dict[str, Any]]:
    return _events


def clear_events() -> None:
    _events.clear()
