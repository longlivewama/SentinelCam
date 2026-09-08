import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.config import settings
from app.core.deps import get_current_user
from app.core.logging_utils import client_ip, redact_email
from app.core.rate_limit import rate_limit
from app.core.security import (
    create_access_token,
    generate_password_reset_token,
    hash_password,
    hash_reset_token,
    verify_password,
)
from app.database import get_db
from app.models.password_reset_token import PasswordResetToken
from app.models.user import ROLE_VIEWER, User
from app.schemas.auth import ForgotPasswordRequest, LoginRequest, ResetPasswordRequest, SignupRequest, TokenResponse
from app.schemas.user import SettingsUpdate, UserOut
from app.services.notifications import notification_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Generic response for forgot-password regardless of whether the email is
# registered, to avoid leaking which addresses have accounts.
FORGOT_PASSWORD_GENERIC_MESSAGE = "If an account with that email exists, a password reset link has been sent."


@router.post(
    "/signup",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit(settings.AUTH_SIGNUP_MAX_REQUESTS, settings.AUTH_SIGNUP_WINDOW_SECONDS))],
)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already in use")

    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        role=ROLE_VIEWER,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token, token_type="bearer", user=UserOut.model_validate(user))


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(rate_limit(settings.AUTH_LOGIN_MAX_REQUESTS, settings.AUTH_LOGIN_WINDOW_SECONDS))],
)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        # Logged because a burst of these is how credential stuffing looks
        # from the outside; the address is redacted (see logging_utils) so
        # the log stays correlatable without becoming a list of user
        # emails. The response stays deliberately identical either way so
        # it doesn't reveal which accounts exist.
        logger.warning(
            "Failed login for %s from %s", redact_email(payload.email), client_ip(request),
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    if not user.is_active:
        logger.warning(
            "Login attempt on disabled account %s from %s",
            redact_email(user.email), client_ip(request),
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is disabled")

    token = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token, token_type="bearer", user=UserOut.model_validate(user))


@router.post(
    "/forgot-password",
    dependencies=[
        Depends(rate_limit(settings.AUTH_FORGOT_PASSWORD_MAX_REQUESTS, settings.AUTH_FORGOT_PASSWORD_WINDOW_SECONDS))
    ],
)
def forgot_password(payload: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if user is not None and user.is_active:
        raw_token, token_hash, expires_at = generate_password_reset_token()
        reset_row = PasswordResetToken(user_id=user.id, token_hash=token_hash, expires_at=expires_at)
        db.add(reset_row)
        db.commit()

        reset_url = f"{_frontend_reset_url()}?token={raw_token}"
        notification_service.notify_password_reset(user.email, reset_url)
    else:
        logger.info(
            "Password reset requested for unknown/inactive account %s", redact_email(payload.email),
        )

    return {"message": FORGOT_PASSWORD_GENERIC_MESSAGE}


@router.post(
    "/reset-password",
    dependencies=[
        Depends(rate_limit(settings.AUTH_RESET_PASSWORD_MAX_REQUESTS, settings.AUTH_RESET_PASSWORD_WINDOW_SECONDS))
    ],
)
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    token_hash = hash_reset_token(payload.token)
    reset_row = db.query(PasswordResetToken).filter(PasswordResetToken.token_hash == token_hash).first()

    invalid_exception = HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="This password reset link is invalid or has expired.",
    )

    if reset_row is None or reset_row.used_at is not None:
        raise invalid_exception

    expires_at = reset_row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise invalid_exception

    user = db.query(User).filter(User.id == reset_row.user_id).first()
    if user is None or not user.is_active:
        raise invalid_exception

    now = datetime.now(timezone.utc)
    user.password_hash = hash_password(payload.new_password)
    reset_row.used_at = now

    # Burn every other outstanding reset token for this account, not just
    # the one presented. Requesting a reset several times (or an attacker
    # requesting one for a victim) leaves multiple live tokens; consuming
    # one must not leave the others usable against the new password.
    _invalidate_outstanding_reset_tokens(db, user.id, now)

    db.commit()

    return {"message": "Password has been reset successfully. You can now sign in with your new password."}


def _invalidate_outstanding_reset_tokens(db: Session, user_id: int, now: datetime) -> None:
    db.query(PasswordResetToken).filter(
        PasswordResetToken.user_id == user_id,
        PasswordResetToken.used_at.is_(None),
    ).update({"used_at": now}, synchronize_session=False)


def _frontend_reset_url() -> str:
    return f"{settings.FRONTEND_URL.rstrip('/')}/reset-password"


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user


@router.put("/settings", response_model=UserOut)
def update_settings(
    payload: SettingsUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(payload.current_password, current_user.password_hash):
        # A wrong current-password on an authenticated session is worth
        # seeing: it is what a hijacked-token account takeover looks like.
        logger.warning(
            "Settings change rejected - wrong current password for user %s from %s",
            current_user.id, client_ip(request),
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Current password is incorrect")

    if payload.new_email is not None and payload.new_email != current_user.email:
        existing = db.query(User).filter(User.email == payload.new_email, User.id != current_user.id).first()
        if existing is not None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already in use")
        current_user.email = payload.new_email

    if payload.new_password:
        current_user.password_hash = hash_password(payload.new_password)
        # A user who just changed their password deliberately should not
        # leave a live "forgot password" link in their mailbox.
        _invalidate_outstanding_reset_tokens(db, current_user.id, datetime.now(timezone.utc))

    db.commit()
    db.refresh(current_user)
    return current_user
