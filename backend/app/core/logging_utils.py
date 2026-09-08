"""
Helpers for logging security-relevant events without writing personal data
into the log.

Failed logins, disabled-account attempts and rate-limit trips all need to
be visible to whoever operates this - they are how you notice a
credential-stuffing run - but the natural way to write them dumps user
email addresses into a log file that is less protected than the database
they came from, and often shipped off-host to an aggregator.

`redact_email` keeps what an operator actually needs (enough to correlate
repeated attempts against the same account, and the domain, which is what
distinguishes a targeted attack from a spray) while dropping the rest of
the local part.
"""
from __future__ import annotations

from typing import Optional


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
