from __future__ import annotations

from fastapi.testclient import TestClient

from backend.analytics.store import clear_events
from backend.app import app

client = TestClient(app)


def test_analytics_returns_empty_initially():
    clear_events()
    resp = client.get("/analytics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_searches"] == 0
    assert body["avg_response_time_ms"] == 0.0


def test_analytics_tracks_search():
    clear_events()
    client.post("/recommendations", json={"location": "BTM"})
    resp = client.get("/analytics")
    body = resp.json()
    assert body["total_searches"] == 1
    assert body["avg_response_time_ms"] > 0
    assert any(loc["name"] == "BTM" for loc in body["top_locations"])


def test_analytics_tracks_multiple_searches():
    clear_events()
    client.post("/recommendations", json={"location": "BTM"})
    client.post("/recommendations", json={"location": "Koramangala"})
    client.post("/recommendations", json={"location": "BTM"})
    resp = client.get("/analytics")
    body = resp.json()
    assert body["total_searches"] == 3


def test_analytics_filter_usage():
    clear_events()
    client.post("/recommendations", json={
        "location": "BTM",
        "cuisines": ["Chinese"],
        "price_range": ["$"],
    })
    resp = client.get("/analytics")
    body = resp.json()
    assert body["filter_usage"]["cuisine"] > 0
    assert body["filter_usage"]["price_range"] > 0
