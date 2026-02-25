from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app import app
from backend.recommendations.cache import clear_cache, get_cache_stats

client = TestClient(app)


def _login_user(c):
    c.post("/auth/login", json={"username": "user", "password": "user123"})


def _login_admin(c):
    c.post("/auth/login", json={"username": "admin", "password": "admin123"})


@patch("backend.recommendations.retrieval.assign_variant", return_value="A")
def test_cache_miss_then_hit(mock_variant):
    clear_cache()
    _login_user(client)
    resp1 = client.post("/recommendations", json={"location": "BTM", "limit": 3})
    assert resp1.status_code == 200
    stats = get_cache_stats()
    assert stats["misses"] >= 1

    # Second identical call — cache hit (same variant forced)
    resp2 = client.post("/recommendations", json={"location": "BTM", "limit": 3})
    assert resp2.status_code == 200
    stats = get_cache_stats()
    assert stats["hits"] >= 1


@patch("backend.recommendations.retrieval.assign_variant", return_value="A")
def test_cache_different_queries_miss(mock_variant):
    clear_cache()
    _login_user(client)
    client.post("/recommendations", json={"location": "BTM"})
    client.post("/recommendations", json={"location": "Koramangala"})
    stats = get_cache_stats()
    assert stats["misses"] >= 2
    assert stats["hits"] == 0


@patch("backend.recommendations.retrieval.assign_variant", return_value="A")
def test_cache_stats_endpoint(mock_variant):
    clear_cache()
    _login_user(client)
    client.post("/recommendations", json={"location": "BTM", "limit": 3})
    client.post("/recommendations", json={"location": "BTM", "limit": 3})
    _login_admin(client)
    resp = client.get("/cache/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"] >= 1
    assert "hit_rate" in body
