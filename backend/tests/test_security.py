from datetime import datetime, timedelta, timezone

from app.core.security import (
    create_access_token,
    decode_access_token,
    generate_password_reset_token,
    hash_password,
    hash_reset_token,
    verify_password,
)


def test_password_hash_roundtrip():
    hashed = hash_password("correct-horse-battery-staple")
    assert hashed != "correct-horse-battery-staple"
    assert verify_password("correct-horse-battery-staple", hashed)
    assert not verify_password("wrong-password", hashed)


def test_access_token_roundtrip():
    token = create_access_token(user_id=42, email="a@b.com")
    payload = decode_access_token(token)
    assert payload is not None
    assert payload["user_id"] == 42
    assert payload["email"] == "a@b.com"


def test_decode_rejects_garbage_token():
    assert decode_access_token("not-a-real-jwt") is None


def test_password_reset_token_is_hashed_not_stored_raw():
    raw_token, token_hash, expires_at = generate_password_reset_token()
    assert raw_token != token_hash
    assert hash_reset_token(raw_token) == token_hash
    # Expiry should be in the future but bounded (uses
    # PASSWORD_RESET_TOKEN_EXPIRE_MINUTES, default 30 min).
    assert expires_at > datetime.now(timezone.utc)
    assert expires_at < datetime.now(timezone.utc) + timedelta(hours=1)
