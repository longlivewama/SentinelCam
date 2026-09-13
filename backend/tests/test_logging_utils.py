"""
Security-event logging must be useful to an operator without turning the
log into a list of user email addresses.
"""
import logging

import pytest

from app.core.logging_utils import (
    RedactTokensFilter,
    install_log_redaction,
    redact_email,
    redact_tokens,
)


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


# --- credential redaction in log records ----------------------------------
#
# The media token is short-lived and scoped to one clip, which is the real
# fix for credentials in URLs. These tests pin the second line of defence:
# even that weakened credential should not be written into a log file,
# because log files outlive the tokens in them and the WebSocket handshake
# URL still carries a session token.

@pytest.mark.parametrize("line,leaked", [
    ('GET /api/recordings/12/video?token=eyJhbGciOiJIUzI1NiJ9.abc.def HTTP/1.1',
     "eyJhbGciOiJIUzI1NiJ9.abc.def"),
    ('/api/ws/events?token=eyJhbGciOi.session.sig', "eyJhbGciOi.session.sig"),
    ('/api/recordings?page=2&token=abc123&page_size=20', "abc123"),
    ('Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.xyz.sig', "eyJhbGciOiJIUzI1NiJ9.xyz.sig"),
])
def test_redact_tokens_removes_the_credential_but_keeps_the_shape(line, leaked):
    redacted = redact_tokens(line)

    assert leaked not in redacted
    assert "[REDACTED]" in redacted


def test_redaction_does_not_swallow_the_rest_of_the_query_string():
    """An operator debugging a paginated request still needs to see which
    page was asked for."""
    redacted = redact_tokens("/api/recordings?page=2&token=secretvalue&page_size=20")

    assert "page=2" in redacted
    assert "page_size=20" in redacted
    assert "secretvalue" not in redacted


def test_redaction_leaves_ordinary_lines_alone():
    line = "Video analysis complete for upload 7: 2 fall event(s) in 41.3s of wall clock"

    assert redact_tokens(line) == line


def test_the_filter_redacts_the_arguments_uvicorn_actually_logs(caplog):
    """Uvicorn logs '%s - "%s %s HTTP/%s" %d' with the path-and-query as an
    ARGUMENT, so a filter that only inspected the format string would find
    nothing to redact. This reproduces that exact call shape."""
    logger = logging.getLogger("test.access.shape")
    logger.addFilter(RedactTokensFilter())

    with caplog.at_level(logging.INFO, logger="test.access.shape"):
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:54321", "GET",
            "/api/recordings/12/video?token=eyJhbGciOiJIUzI1NiJ9.leaked.sig",
            "1.1", 206,
        )

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "eyJhbGciOiJIUzI1NiJ9.leaked.sig" not in logged
    assert "[REDACTED]" in logged
    assert "/api/recordings/12/video" in logged, "the path itself is still useful"


def test_the_filter_redacts_the_websocket_handshake_line(caplog):
    logger = logging.getLogger("test.ws.shape")
    logger.addFilter(RedactTokensFilter())

    with caplog.at_level(logging.INFO, logger="test.ws.shape"):
        logger.info(
            '%s - "WebSocket %s" [accepted]',
            ("127.0.0.1", 54321), "/api/ws/events?token=eyJsession.token.sig",
        )

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "eyJsession.token.sig" not in logged
    assert "[REDACTED]" in logged


def test_the_filter_never_drops_a_record():
    """It exists to rewrite records, not to suppress them - a filter that
    returned False would silently delete the access log."""
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "no tokens here", None, None)

    assert RedactTokensFilter().filter(record) is True


def test_the_filter_survives_unformattable_records():
    """A logging filter that raises turns every log call in the process
    into an exception, so it has to tolerate whatever it is handed."""
    record = logging.LogRecord("x", logging.INFO, __file__, 1, object(), (object(),), None)

    assert RedactTokensFilter().filter(record) is True


def test_install_log_redaction_is_idempotent():
    """Called from app import; a test or a reload must not stack filters."""
    install_log_redaction()
    install_log_redaction()

    access = logging.getLogger("uvicorn.access")
    installed = [f for f in access.filters if isinstance(f, RedactTokensFilter)]
    assert len(installed) == 1


def test_a_real_media_request_does_not_log_its_token(client, viewer_headers, db, tmp_path, caplog):
    """End to end: mint a token, use it, and confirm NOTHING anywhere in
    the captured log contains it. This is the assertion the original QA
    capture failed.

    Redaction is not limited to loggers this application owns: the record
    factory catches every record in the process, which is what covers the
    HTTP client library used by TestClient - it logs full request URLs,
    token and all, through a logger nothing here names."""
    from datetime import datetime, timezone
    from app.models.recording import Recording

    install_log_redaction()

    path = tmp_path / "redaction.webm"
    path.write_bytes(b"\x1a\x45\xdf\xa3" + b"payload" * 64)
    row = Recording(
        camera_id=None, video_upload_id=None, filename="redaction.webm", file_path=str(path),
        duration_seconds=1.0, trigger_action="fall", file_size_bytes=path.stat().st_size,
        event_timestamp=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    token = client.post(f"/api/recordings/{row.id}/media-token", headers=viewer_headers).json()["token"]

    with caplog.at_level(logging.DEBUG):
        # Logged through the same logger Uvicorn uses, with the same call
        # shape - TestClient does not run Uvicorn's access log itself.
        access = logging.getLogger("uvicorn.access")
        response = client.get(f"/api/recordings/{row.id}/video?token={token}")
        access.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:1234", "GET", f"/api/recordings/{row.id}/video?token={token}", "1.1",
            response.status_code,
        )

    assert response.status_code == 200
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert token not in logged, "the media token reached the log verbatim"
    assert "[REDACTED]" in logged
