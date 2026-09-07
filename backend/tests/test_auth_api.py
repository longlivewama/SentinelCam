from app.models.password_reset_token import PasswordResetToken
from app.models.user import ROLE_VIEWER, User


def test_signup_creates_viewer_and_returns_token(client, db):
    resp = client.post(
        "/api/auth/signup",
        json={"email": "newuser@example.com", "password": "password123", "full_name": "New User"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["user"]["role"] == ROLE_VIEWER
    assert body["user"]["is_admin"] is False
    assert "access_token" in body

    user = db.query(User).filter(User.email == "newuser@example.com").first()
    assert user is not None
    assert user.role == ROLE_VIEWER


def test_signup_rejects_duplicate_email(client, viewer_user):
    resp = client.post(
        "/api/auth/signup",
        json={"email": viewer_user.email, "password": "password123"},
    )
    assert resp.status_code == 400


def test_signup_rejects_short_password(client):
    resp = client.post("/api/auth/signup", json={"email": "short@example.com", "password": "abc"})
    assert resp.status_code == 422


def test_login_success_and_failure(client, viewer_user):
    ok = client.post("/api/auth/login", json={"email": viewer_user.email, "password": "password123"})
    assert ok.status_code == 200
    assert ok.json()["user"]["email"] == viewer_user.email

    bad = client.post("/api/auth/login", json={"email": viewer_user.email, "password": "wrong"})
    assert bad.status_code == 401


def test_me_requires_auth(client):
    assert client.get("/api/auth/me").status_code == 401


def test_me_returns_current_user(client, viewer_headers, viewer_user):
    resp = client.get("/api/auth/me", headers=viewer_headers)
    assert resp.status_code == 200
    assert resp.json()["email"] == viewer_user.email


def test_forgot_password_returns_generic_message_for_unknown_email(client):
    resp = client.post("/api/auth/forgot-password", json={"email": "nobody@example.com"})
    assert resp.status_code == 200
    assert "reset link" in resp.json()["message"].lower()


def test_forgot_password_creates_reset_token_for_known_email(client, viewer_user, db):
    resp = client.post("/api/auth/forgot-password", json={"email": viewer_user.email})
    assert resp.status_code == 200

    token_row = db.query(PasswordResetToken).filter(PasswordResetToken.user_id == viewer_user.id).first()
    assert token_row is not None
    assert token_row.used_at is None


def test_reset_password_rejects_invalid_token(client):
    resp = client.post("/api/auth/reset-password", json={"token": "bogus", "new_password": "newpassword123"})
    assert resp.status_code == 400


def test_reset_password_full_flow(client, viewer_user, db, monkeypatch):
    captured = {}

    def _fake_notify(email, reset_url):
        captured["email"] = email
        captured["reset_url"] = reset_url
        return True

    import app.api.routes.auth as auth_routes

    monkeypatch.setattr(auth_routes.notification_service, "notify_password_reset", _fake_notify)

    resp = client.post("/api/auth/forgot-password", json={"email": viewer_user.email})
    assert resp.status_code == 200
    assert captured["email"] == viewer_user.email
    raw_token = captured["reset_url"].split("token=")[1]

    reset_resp = client.post(
        "/api/auth/reset-password", json={"token": raw_token, "new_password": "brandnewpassword"}
    )
    assert reset_resp.status_code == 200

    # Old password no longer works, new one does.
    assert client.post(
        "/api/auth/login", json={"email": viewer_user.email, "password": "password123"}
    ).status_code == 401
    assert client.post(
        "/api/auth/login", json={"email": viewer_user.email, "password": "brandnewpassword"}
    ).status_code == 200

    # Token is single-use.
    reused = client.post(
        "/api/auth/reset-password", json={"token": raw_token, "new_password": "anotherpassword"}
    )
    assert reused.status_code == 400


def test_update_settings_requires_current_password(client, viewer_headers):
    resp = client.put(
        "/api/auth/settings",
        json={"current_password": "wrong", "new_email": "changed@example.com"},
        headers=viewer_headers,
    )
    assert resp.status_code == 401


def test_update_settings_changes_email(client, viewer_headers, viewer_user):
    resp = client.put(
        "/api/auth/settings",
        json={"current_password": "password123", "new_email": "changed@example.com"},
        headers=viewer_headers,
    )
    assert resp.status_code == 200
    assert resp.json()["email"] == "changed@example.com"
