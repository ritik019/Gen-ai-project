from __future__ import annotations

from fastapi.testclient import TestClient

from backend.analytics.feedback import clear_feedback
from backend.app import app

client = TestClient(app)


def test_feedback_records_positive():
    clear_feedback()
    resp = client.post("/feedback", json={
        "restaurant_id": "42",
        "query_location": "BTM",
        "is_positive": True,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"
    assert resp.json()["total_feedback"] == 1


def test_feedback_records_negative():
    clear_feedback()
    resp = client.post("/feedback", json={
        "restaurant_id": "42",
        "query_location": "BTM",
        "is_positive": False,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"


def test_feedback_validation_rejects_empty_id():
    resp = client.post("/feedback", json={
        "restaurant_id": "",
        "query_location": "BTM",
        "is_positive": True,
    })
    assert resp.status_code == 422


def test_feedback_stats():
    clear_feedback()
    client.post("/feedback", json={"restaurant_id": "1", "query_location": "BTM", "is_positive": True})
    client.post("/feedback", json={"restaurant_id": "2", "query_location": "BTM", "is_positive": True})
    client.post("/feedback", json={"restaurant_id": "3", "query_location": "BTM", "is_positive": False})
    resp = client.get("/feedback/stats")
    body = resp.json()
    assert body["total"] == 3
    assert body["positive"] == 2
    assert body["negative"] == 1
    assert body["satisfaction_rate"] == 66.7
