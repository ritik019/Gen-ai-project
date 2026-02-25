import numpy as np

from backend.embeddings.encoder import encode_text


def test_encode_text_returns_correct_shape():
    vec = encode_text("Italian cafe with rooftop seating")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)


def test_semantic_similarity_ranks_italian_higher():
    """Italian-related query should be closer to Italian restaurant text than North Indian."""
    query = encode_text("romantic Italian cafe date night")
    italian = encode_text("San Churro Cafe Cafe Mexican Italian Koramangala")
    north_indian = encode_text("Jalsa North Indian Mughlai Chinese Banashankari")

    sim_italian = float(np.dot(query, italian) / (np.linalg.norm(query) * np.linalg.norm(italian)))
    sim_north_indian = float(np.dot(query, north_indian) / (np.linalg.norm(query) * np.linalg.norm(north_indian)))

    assert sim_italian > sim_north_indian


def test_no_free_text_preserves_heuristic_order():
    """Without free_text_preferences, the API should still work (backward compat)."""
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    client.post("/auth/login", json={"username": "user", "password": "user123"})
    resp = client.post("/recommendations", json={"location": "BTM", "limit": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["recommendations"]) <= 3
    scores = [item["score"] for item in body["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_free_text_influences_results():
    """With free_text_preferences, results should differ from without."""
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    client.post("/auth/login", json={"username": "user", "password": "user123"})

    resp_without = client.post(
        "/recommendations",
        json={"location": "Koramangala", "limit": 5},
    )
    resp_with = client.post(
        "/recommendations",
        json={"location": "Koramangala", "free_text_preferences": "Italian pizza pasta", "limit": 5},
    )

    ids_without = [r["restaurant"]["id"] for r in resp_without.json()["recommendations"]]
    ids_with = [r["restaurant"]["id"] for r in resp_with.json()["recommendations"]]

    assert ids_without != ids_with
