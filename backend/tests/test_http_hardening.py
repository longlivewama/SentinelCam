"""
Baseline response hardening: security headers on every response, generic
error bodies, and the bcrypt-driven password length bound.
"""
import pytest

from app.core.http_hardening import SECURITY_HEADERS
from app.schemas.user import MAX_PASSWORD_BYTES


@pytest.mark.parametrize("header,value", sorted(SECURITY_HEADERS.items()))
def test_security_headers_are_present_on_api_responses(client, viewer_headers, header, value):
    response = client.get("/api/alerts", headers=viewer_headers)
    assert response.headers.get(header) == value


def test_security_headers_are_present_on_unauthenticated_responses(client):
    """An error response is still a response the browser processes."""
    response = client.get("/api/alerts")
    assert response.status_code == 401
    assert response.headers.get("X-Content-Type-Options") == "nosniff"


def test_health_endpoint_does_not_leak_configuration(client):
    body = client.get("/health").json()
    assert set(body) == {"status", "environment"}


# --- password bounds ------------------------------------------------------

def test_signup_rejects_a_password_past_bcrypts_72_byte_limit(client):
    """Longer input would be silently truncated by bcrypt, so two
    different passwords sharing a 72-byte prefix would share a hash."""
    response = client.post("/api/auth/signup", json={
        "email": "long-password@example.com",
        "password": "a" * (MAX_PASSWORD_BYTES + 1),
        "full_name": "Long Password",
    })
    assert response.status_code == 422


def test_signup_counts_bytes_not_characters(client):
    """A 40-character emoji password is 160 bytes - over the limit even
    though len() says otherwise."""
    response = client.post("/api/auth/signup", json={
        "email": "emoji-password@example.com",
        "password": "\U0001f510" * 40,
        "full_name": "Emoji Password",
    })
    assert response.status_code == 422


def test_a_password_at_the_limit_is_still_accepted(client):
    response = client.post("/api/auth/signup", json={
        "email": "boundary@example.com",
        "password": "a" * MAX_PASSWORD_BYTES,
        "full_name": "Boundary",
    })
    assert response.status_code == 201


def test_short_passwords_are_still_rejected(client):
    response = client.post("/api/auth/signup", json={
        "email": "short@example.com",
        "password": "short",
    })
    assert response.status_code == 422
