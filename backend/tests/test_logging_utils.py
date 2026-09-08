"""
Security-event logging must be useful to an operator without turning the
log into a list of user email addresses.
"""
import logging

import pytest

from app.core.logging_utils import redact_email


@pytest.mark.parametrize("address,expected", [
    ("alice@example.com", "a***@example.com"),
    ("b@x.io", "b***@x.io"),
    ("@nolocal.com", "***@nolocal.com"),
])
def test_redaction_keeps_the_domain_and_first_character(address, expected):
    assert redact_email(address) == expected


def test_redaction_handles_missing_or_malformed_input():
    assert redact_email(None) == "<none>"
    assert redact_email("") == "<none>"
    # Must not echo back whatever was submitted.
    assert redact_email("not-an-address") == "<malformed>"


def test_failed_login_is_logged_without_the_full_address(client, viewer_user, caplog):
    with caplog.at_level(logging.WARNING, logger="app.api.routes.auth"):
        response = client.post("/api/auth/login", json={
            "email": viewer_user.email,
            "password": "definitely-the-wrong-password",
        })

    assert response.status_code == 401
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "Failed login" in logged
    assert viewer_user.email not in logged
    assert "v***@example.com" in logged


def test_no_password_ever_reaches_the_log(client, viewer_user, caplog):
    secret = "SuperSecretPassword!123"
    with caplog.at_level(logging.DEBUG):
        client.post("/api/auth/login", json={"email": viewer_user.email, "password": secret})
        client.post("/api/auth/signup", json={
            "email": "log-probe@example.com", "password": secret, "full_name": "Probe",
        })

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert secret not in logged


def test_forgot_password_for_an_unknown_address_does_not_log_it(client, caplog):
    with caplog.at_level(logging.INFO, logger="app.api.routes.auth"):
        response = client.post("/api/auth/forgot-password", json={"email": "ghost@nowhere.example"})

    assert response.status_code == 200
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "ghost@nowhere.example" not in logged
    assert "g***@nowhere.example" in logged


def test_rate_limit_trips_are_logged(client, caplog):
    """A burst of 429s is the signal that someone is being brute-forced;
    it has to be visible to whoever runs this."""
    with caplog.at_level(logging.WARNING, logger="app.core.rate_limit"):
        for _ in range(40):
            response = client.post("/api/auth/login", json={
                "email": "nobody@example.com", "password": "wrong-password-here",
            })
            if response.status_code == 429:
                break

    assert response.status_code == 429
    assert any("Rate limit hit" in r.getMessage() for r in caplog.records)
