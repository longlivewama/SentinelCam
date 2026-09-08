from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, field_validator

from app.models.user import VALID_ROLES


MIN_PASSWORD_LENGTH = 8
# bcrypt hashes at most the first 72 BYTES of a password and ignores the
# rest. Accepting longer input would mean two different passwords sharing
# a hash whenever they agree on that prefix - and passlib's bcrypt backend
# raises on over-length input in some versions, which would surface to the
# user as a 500 rather than a validation error. Rejecting at the boundary
# makes the limit explicit instead of silent.
MAX_PASSWORD_BYTES = 72


def _validate_password_strength(password: str) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes long "
            "(non-ASCII characters count as more than one byte)"
        )
    return password


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: Optional[str] = None
    role: str
    is_admin: bool
    is_active: bool
    created_at: datetime


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: str = "viewer"

    @field_validator("password")
    @classmethod
    def _check_password(cls, v: str) -> str:
        return _validate_password_strength(v)

    @field_validator("role")
    @classmethod
    def _check_role(cls, v: str) -> str:
        if v not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}")
        return v


class UserUpdateRole(BaseModel):
    role: str

    @field_validator("role")
    @classmethod
    def _check_role(cls, v: str) -> str:
        if v not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}")
        return v


class SettingsUpdate(BaseModel):
    current_password: str
    new_email: Optional[EmailStr] = None
    new_password: Optional[str] = None

    @field_validator("new_password")
    @classmethod
    def _check_password(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return _validate_password_strength(v)
