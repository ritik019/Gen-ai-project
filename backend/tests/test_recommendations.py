from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_recommendations_returns_results():
    resp = client.post("/recommendations", json={"location": "BTM"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_candidates"] > 0
    assert len(body["recommendations"]) > 0


def test_recommendations_respects_limit():
    resp = client.post("/recommendations", json={"location": "BTM", "limit": 3})
    body = resp.json()
    assert len(body["recommendations"]) <= 3


def test_recommendations_filters_by_price():
    resp = client.post(
        "/recommendations",
        json={"location": "Koramangala", "price_range": ["$"]},
    )
    body = resp.json()
    for item in body["recommendations"]:
        assert item["restaurant"]["price_bucket"] == "$"


def test_recommendations_filters_by_min_rating():
    resp = client.post(
        "/recommendations",
        json={"location": "Indiranagar", "min_rating": 4.5},
    )
    body = resp.json()
    for item in body["recommendations"]:
        assert item["restaurant"]["avg_rating"] >= 4.5


def test_recommendations_filters_by_cuisine():
    resp = client.post(
        "/recommendations",
        json={"location": "BTM", "cuisines": ["Chinese"]},
    )
    body = resp.json()
    for item in body["recommendations"]:
        cuisines_lower = [c.lower() for c in item["restaurant"]["cuisines"]]
        assert "chinese" in cuisines_lower


def test_recommendations_empty_for_unknown_location():
    resp = client.post("/recommendations", json={"location": "Nonexistent12345"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_candidates"] == 0
    assert body["recommendations"] == []


def test_recommendations_validation_rejects_bad_rating():
    resp = client.post(
        "/recommendations",
        json={"location": "BTM", "min_rating": 6.0},
    )
    assert resp.status_code == 422


def test_recommendations_validation_rejects_bad_limit():
    resp = client.post(
        "/recommendations",
        json={"location": "BTM", "limit": 0},
    )
    assert resp.status_code == 422


def test_recommendations_score_ordering():
    resp = client.post(
        "/recommendations",
        json={"location": "BTM", "limit": 10},
    )
    body = resp.json()
    scores = [item["score"] for item in body["recommendations"]]
    assert scores == sorted(scores, reverse=True)
