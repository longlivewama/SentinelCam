"""
Password hashing (bcrypt via passlib), JWT issuance/verification
(python-jose, HS256), media-token issuance, and password-reset token
generation.

Two kinds of bearer credential are minted here, and the `typ` claim is
what keeps them apart:

  * the SESSION token (`typ` absent, or "access") - full API privilege,
    lifetime measured in hours, and only ever accepted from an
    `Authorization: Bearer` header.
  * the MEDIA token (`typ` == "media") - readable only by the one media
    resource named in its `res` claim, lifetime measured in minutes, and
    the only thing the media endpoints will accept in a `?token=` query
    string.

Neither is usable where the other belongs: `decode_media_token` refuses a
session token, and `core/deps.py` refuses a media token presented as a
bearer. That is the whole point - a credential that has to travel in a
URL (and therefore into access logs, proxies and browser history) must
not also be the credential that can delete an account.
"""
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


# Value of the `typ` claim on a media token. A session token carries
# TOKEN_TYPE_ACCESS - or, for one issued before this claim existed, no
# `typ` at all, which is why "absent" is read as "session" and only an
# explicit "media" is treated as scoped.
TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_MEDIA = "media"


def create_access_token(user_id: int, email: str) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(hours=settings.JWT_EXPIRE_HOURS)
    payload = {
        "sub": str(user_id),
        "user_id": user_id,
        "email": email,
        "typ": TOKEN_TYPE_ACCESS,
        "iat": now,
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict[str, Any]]:
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        return None


def is_media_token(payload: dict[str, Any]) -> bool:
    return payload.get("typ") == TOKEN_TYPE_MEDIA


def media_resource_id(kind: str, resource_id: Any) -> str:
    """The canonical `res` claim for one media resource, e.g.
    `recording:12`. Minting and checking both go through this so the two
    can never drift into disagreeing about the format - a mismatch there
    would fail open only if it were built by hand in two places."""
    return f"{kind}:{resource_id}"


def create_media_token(user_id: int, resource: str) -> tuple[str, int]:
    """A short-lived credential that reads exactly one media resource.

    `resource` comes from `media_resource_id`. Returns
    `(token, expires_in_seconds)` so the caller can tell the client when
    to come back for a fresh one rather than hard-coding the lifetime on
    both sides.

    Carries no email and no role: everything an endpoint needs it looks
    up from `user_id` at request time, so a token recovered from a log
    discloses nothing about the account beyond its numeric id."""
    now = datetime.now(timezone.utc)
    expires_in = max(int(settings.MEDIA_TOKEN_EXPIRE_SECONDS), 1)
    payload = {
        "sub": str(user_id),
        "user_id": user_id,
        "typ": TOKEN_TYPE_MEDIA,
        "res": resource,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
    }
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, expires_in


def decode_media_token(token: str, resource: str) -> Optional[dict[str, Any]]:
    """Verifies `token` is a media token, unexpired, and issued for
    `resource` specifically. Returns None on any failure - a forged
    signature, an expired token, a session token presented in a media
    endpoint's query string, or a valid media token for some OTHER clip.

    The `res` comparison is what stops one leaked media URL from reading
    the rest of the library."""
    payload = decode_access_token(token)
    if payload is None:
        return None
    if not is_media_token(payload):
        return None
    if payload.get("res") != resource:
        return None
    return payload


def generate_password_reset_token() -> tuple[str, str, datetime]:
    """Returns (raw_token, token_hash, expires_at). The raw token is what
    gets emailed to the user and is never persisted; only its SHA-256 hash
    is stored, so a database leak alone can't be used to reset accounts."""
    raw_token = secrets.token_urlsafe(32)
    token_hash = hash_reset_token(raw_token)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)
    return raw_token, token_hash, expires_at


def hash_reset_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
