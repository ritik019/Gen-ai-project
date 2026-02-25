from __future__ import annotations

from fastapi.testclient import TestClient

from backend.ab_testing.experiments import (
    assign_variant,
    clear_assignments,
    get_variant_weights,
)
from backend.analytics.feedback import clear_feedback
from backend.app import app

client = TestClient(app)


def _login_user(c):
    c.post("/auth/login", json={"username": "user", "password": "user123"})


def _login_admin(c):
    c.post("/auth/login", json={"username": "admin", "password": "admin123"})


def test_assign_variant_returns_a_or_b():
    variant = assign_variant()
    assert variant in ("A", "B")


def test_variant_weights_differ():
    weights_a = get_variant_weights("A")
    weights_b = get_variant_weights("B")
    assert weights_a != weights_b
    assert weights_a["rating"] == 0.6
    assert weights_b["rating"] == 0.4


def test_recommendation_response_includes_variant():
    _login_user(client)
    resp = client.post("/recommendations", json={"location": "BTM", "limit": 3})
    body = resp.json()
    assert "variant" in body
    assert body["variant"] in ("A", "B")


def test_ab_test_results_endpoint():
    clear_assignments()
    clear_feedback()
    _login_admin(client)
    resp = client.get("/ab-test/results")
    assert resp.status_code == 200
    body = resp.json()
    assert "experiment" in body
    assert "results" in body
    assert "A" in body["results"]
    assert "B" in body["results"]


def test_feedback_with_variant():
    clear_feedback()
    _login_user(client)
    resp = client.post("/feedback", json={
        "restaurant_id": "42",
        "query_location": "BTM",
        "is_positive": True,
        "variant": "A",
    })
    assert resp.status_code == 200
