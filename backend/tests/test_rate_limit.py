def test_login_is_rate_limited_after_repeated_attempts(client):
    for _ in range(10):
        resp = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "wrong"})
        assert resp.status_code == 401

    limited = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "wrong"})
    assert limited.status_code == 429


def test_signup_is_rate_limited_after_repeated_attempts(client):
    for i in range(5):
        client.post("/api/auth/signup", json={"email": f"ratelimit{i}@example.com", "password": "password123"})

    limited = client.post("/api/auth/signup", json={"email": "onemore@example.com", "password": "password123"})
    assert limited.status_code == 429


def test_forgot_password_is_rate_limited_after_repeated_attempts(client):
    for _ in range(5):
        client.post("/api/auth/forgot-password", json={"email": "nobody@example.com"})

    limited = client.post("/api/auth/forgot-password", json={"email": "nobody@example.com"})
    assert limited.status_code == 429


def test_rate_limit_is_scoped_per_route(client):
    """Exhausting the login limit must not also block signup - the limiter
    is keyed per (path, client), not globally per client."""
    for _ in range(10):
        client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "wrong"})
    assert client.post(
        "/api/auth/login", json={"email": "nobody@example.com", "password": "wrong"}
    ).status_code == 429

    signup_resp = client.post("/api/auth/signup", json={"email": "stillworks@example.com", "password": "password123"})
    assert signup_resp.status_code == 201
