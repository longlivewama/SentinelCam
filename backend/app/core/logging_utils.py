"""
Helpers for logging security-relevant events without writing personal data
or credentials into the log.

Failed logins, disabled-account attempts and rate-limit trips all need to
be visible to whoever operates this - they are how you notice a
credential-stuffing run - but the natural way to write them dumps user
email addresses into a log file that is less protected than the database
they came from, and often shipped off-host to an aggregator.

`redact_email` keeps what an operator actually needs (enough to correlate
repeated attempts against the same account, and the domain, which is what
distinguishes a targeted attack from a spray) while dropping the rest of
the local part.

`install_log_redaction` covers the other half: tokens. Uvicorn's access
log writes the full request line, query string included, so every request
to a media URL used to record its credential verbatim:

    127.0.0.1:54xxx - "GET /api/recordings/12/video?token=eyJhbGciOi... HTTP/1.1" 206

Media tokens are now short-lived and scoped to one clip (see
core/security.py), which is the real fix - but a credential that is merely
weak still should not be written down, log files outlive the tokens in
them, and the WebSocket handshake URL carries a session token that this
same filter catches. Redaction is the belt to that fix's braces.

The same filter also strips the inline userinfo of a stream URL, because
an IP camera's credentials live there (`rtsp://admin:hunter2@host/...`)
and a log file is the last place they should end up. See
core/camera_url.py.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from app.core.camera_url import redact_credentials


def redact_email(email: Optional[str]) -> str:
    """`alice@example.com` -> `a***@example.com`. Stable for a given
    address, so repeated attempts correlate, but not the address itself.
    Never raises - this is called from logging paths."""
    if not email:
        return "<none>"
    local, sep, domain = str(email).partition("@")
    if not sep:
        # Not an address at all; don't echo whatever was submitted.
        return "<malformed>"
    visible = local[:1] if local else ""
    return f"{visible}***@{domain}"


def client_ip(request) -> str:
    """The peer address, for correlating auth failures. Behind a proxy
    this is the proxy - see core/rate_limit.py's note."""
    try:
        return request.client.host if request.client else "unknown"
    except Exception:
        return "unknown"


# --- credential redaction in log records ---------------------------------

REDACTED = "[REDACTED]"

# A query-string credential runs to the next `&`, whitespace, or quote -
# in an access log the request line is `GET /path?token=... HTTP/1.1`, so
# the space before HTTP is what terminates it. `access_token` is matched
# too because that is the name the login response uses, and a future
# caller could plausibly put it in a URL.
_QUERY_TOKEN_RE = re.compile(r"(?i)((?:access_token|token)=)[^&\s\"'<>]+")

# Belt and braces for anywhere a header value reaches a log line. Uvicorn
# does not log headers, but application code, a traceback, or a future
# middleware could.
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*")


def redact_tokens(text: str) -> str:
    """`?token=eyJhbGci...` -> `?token=[REDACTED]`. Leaves the parameter
    name in place: an operator debugging a 401 needs to see that a token
    was presented, just not what it was."""
    return _BEARER_RE.sub(rf"\1{REDACTED}", _QUERY_TOKEN_RE.sub(rf"\1{REDACTED}", text))


def redact_sensitive(text: str) -> str:
    """Every credential class this process knows how to recognise in a log
    line: bearer/query tokens, and the inline userinfo of a stream URL.

    The camera case is the same argument as the token case, one layer
    down. An IP camera's URL carries its password
    (`rtsp://admin:hunter2@host/...`), the capture loop already redacts it
    at the one site that logs it deliberately, and this is the backstop for
    every site that does not know it is holding one - an exception message
    from FFmpeg, a repr in a traceback, a future log line."""
    return redact_credentials(redact_tokens(text))


def _redact_value(value: Any) -> Any:
    return redact_sensitive(value) if isinstance(value, str) else value


def _redact_record(record: logging.LogRecord) -> None:
    """Rewrites a record in place. Touches `args` as well as `msg` because
    that is where the interesting string actually lives: Uvicorn logs
    `'%s - "%s %s HTTP/%s" %d'` with the path-and-query-string as an
    ARGUMENT, so anything that only inspected `msg` would find nothing but
    the format string. The WebSocket handshake line
    (`'%s - "WebSocket %s" [accepted]'`) has the same shape.

    Swallows its own errors: anything on this path runs inside every log
    call in the process, and raising here would turn logging itself into
    a source of exceptions."""
    try:
        if isinstance(record.msg, str):
            record.msg = redact_sensitive(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(_redact_value(a) for a in record.args)
        elif isinstance(record.args, dict):
            record.args = {k: _redact_value(v) for k, v in record.args.items()}

        # Not every credential arrives as a string. An argument can be an
        # object that only reveals one when formatted - the HTTP client
        # library logs an httpx.URL, not a str, so the pass above leaves
        # it untouched and the token appears at render time. Re-check the
        # rendered message, and where something survived, replace the
        # record with its redacted rendering.
        #
        # Deliberately last and conditional: collapsing msg+args into one
        # string costs structured-logging consumers their fields, so it
        # only happens for the records that would otherwise leak.
        if record.args:
            rendered = record.getMessage()
            redacted = redact_sensitive(rendered)
            if redacted != rendered:
                record.msg = redacted
                record.args = None
    except Exception:  # pragma: no cover - defensive; must never break logging
        pass


class RedactTokensFilter(logging.Filter):
    """Filter form of `_redact_record`, for attaching to a specific logger
    or handler.

    Returns True unconditionally - it exists to modify records, never to
    drop them. A filter that returned False would silently delete the
    access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        _redact_record(record)
        return True


# Uvicorn's own loggers do not propagate to root, so they are named
# explicitly rather than relying on a root handler to catch them.
_REDACTED_LOGGERS = ("uvicorn", "uvicorn.access", "uvicorn.error")

# Set once the record factory below has been installed, so repeated calls
# don't nest the wrapper N deep.
_FACTORY_MARKER = "_sentinelcam_redacts_tokens"


def _install_record_factory() -> None:
    """Redacts at record CREATION, which is the only hook that covers
    every logger and every handler unconditionally.

    Filters are attached to particular loggers and handlers, so they miss
    two cases that matter: a library that logs full URLs through a logger
    nobody thought to name (the HTTP client used by the test suite does
    exactly this), and a handler added after startup by some later logging
    configuration. A record factory has neither gap - nothing reaches a
    handler without passing through it."""
    factory = logging.getLogRecordFactory()
    if getattr(factory, _FACTORY_MARKER, False):
        return

    def redacting_factory(*args, **kwargs):
        record = factory(*args, **kwargs)
        _redact_record(record)
        return record

    setattr(redacting_factory, _FACTORY_MARKER, True)
    logging.setLogRecordFactory(redacting_factory)


def install_log_redaction() -> RedactTokensFilter:
    """Installs credential redaction across the process's logging.

    Belt and braces, deliberately:

      * a log-record factory, which every record passes through no matter
        which logger created it or which handler emits it;
      * explicit filters on Uvicorn's loggers, which do not propagate to
        root, so that redaction survives even if something later replaces
        the record factory.

    Idempotent - safe to call from app import and from a test."""
    redactor = RedactTokensFilter()

    _install_record_factory()

    for name in _REDACTED_LOGGERS:
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactTokensFilter) for f in logger.filters):
            logger.addFilter(redactor)

    return redactor
