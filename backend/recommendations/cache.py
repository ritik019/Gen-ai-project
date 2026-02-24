from __future__ import annotations

import hashlib
import json
import time
from typing import Any

_cache: dict[str, dict[str, Any]] = {}
_hits: int = 0
_misses: int = 0
_DEFAULT_TTL = 300  # 5 minutes


def _make_key(request_dict: dict) -> str:
    normalized = json.dumps(request_dict, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def cache_get(request_dict: dict) -> Any | None:
    global _hits, _misses
    key = _make_key(request_dict)
    entry = _cache.get(key)
    if entry and time.time() - entry["created_at"] < _DEFAULT_TTL:
        _hits += 1
        return entry["value"]
    if entry:
        del _cache[key]
    _misses += 1
    return None


def cache_set(request_dict: dict, value: Any) -> None:
    key = _make_key(request_dict)
    _cache[key] = {"value": value, "created_at": time.time()}


def get_cache_stats() -> dict:
    total = _hits + _misses
    return {
        "size": len(_cache),
        "hits": _hits,
        "misses": _misses,
        "hit_rate": round(_hits / total * 100, 1) if total > 0 else 0.0,
    }


def clear_cache() -> None:
    global _hits, _misses
    _cache.clear()
    _hits = 0
    _misses = 0
