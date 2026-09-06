from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: Optional[str] = None
    is_admin: bool
    is_active: bool
    created_at: datetime


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    is_admin: bool = False


class SettingsUpdate(BaseModel):
    current_password: str
    new_email: Optional[EmailStr] = None
    new_password: Optional[str] = None
