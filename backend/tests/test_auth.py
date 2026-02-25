from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def _login_user(c):
    c.post("/auth/login", json={"username": "user", "password": "user123"})


def _login_admin(c):
    c.post("/auth/login", json={"username": "admin", "password": "admin123"})


# ── Login / Logout ───────────────────────────────────────────────────────


def test_login_success_user():
    resp = client.post("/auth/login", json={"username": "user", "password": "user123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["user"]["username"] == "user"
    assert body["user"]["role"] == "user"


def test_login_success_admin():
    resp = client.post("/auth/login", json={"username": "admin", "password": "admin123"})
    assert resp.status_code == 200
    assert resp.json()["user"]["role"] == "admin"


def test_login_wrong_password():
    resp = client.post("/auth/login", json={"username": "user", "password": "wrong"})
    assert resp.status_code == 401


def test_login_unknown_user():
    resp = client.post("/auth/login", json={"username": "nobody", "password": "x"})
    assert resp.status_code == 401


def test_auth_me_when_logged_in():
    _login_user(client)
    resp = client.get("/auth/me")
    assert resp.status_code == 200
    assert resp.json()["username"] == "user"


def test_auth_me_not_logged_in():
    c = TestClient(app)  # fresh client, no session
    resp = c.get("/auth/me")
    assert resp.status_code == 401


def test_logout():
    _login_user(client)
    resp = client.post("/auth/logout")
    assert resp.status_code == 200
    assert resp.json()["status"] == "logged_out"
    # Session should be cleared
    resp = client.get("/auth/me")
    assert resp.status_code == 401


# ── Route protection ─────────────────────────────────────────────────────


def test_recommendations_requires_login():
    c = TestClient(app)
    resp = c.post("/recommendations", json={"location": "BTM"})
    assert resp.status_code == 401


def test_feedback_requires_login():
    c = TestClient(app)
    resp = c.post("/feedback", json={
        "restaurant_id": "1", "query_location": "BTM", "is_positive": True,
    })
    assert resp.status_code == 401


def test_analytics_requires_admin():
    _login_user(client)
    resp = client.get("/analytics")
    assert resp.status_code == 403


def test_analytics_allowed_for_admin():
    _login_admin(client)
    resp = client.get("/analytics")
    assert resp.status_code == 200


def test_ab_test_results_requires_admin():
    c = TestClient(app)
    _login_user(c)
    resp = c.get("/ab-test/results")
    assert resp.status_code == 403


def test_feedback_stats_requires_admin():
    c = TestClient(app)
    _login_user(c)
    resp = c.get("/feedback/stats")
    assert resp.status_code == 403


def test_cache_stats_requires_admin():
    c = TestClient(app)
    _login_user(c)
    resp = c.get("/cache/stats")
    assert resp.status_code == 403


# ── Public endpoints stay public ─────────────────────────────────────────


def test_health_is_public():
    c = TestClient(app)
    assert c.get("/health").status_code == 200


def test_metadata_is_public():
    c = TestClient(app)
    assert c.get("/metadata").status_code == 200
