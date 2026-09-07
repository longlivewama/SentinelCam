"""
Shared FastAPI dependencies: current-user resolution and role-based access
control (RBAC) gating.

Critical detail: browsers cannot attach a custom `Authorization` header to
`<img src>`, `<video src>`, or plain `<a href>` downloads. So
`get_current_user` accepts the JWT from EITHER the `Authorization: Bearer
<token>` header OR a `?token=<jwt>` query parameter, checking the header
first and falling back to the query param. JSON API calls from axios will
carry the header; the stream/video/download endpoints are hit via raw URLs
(e.g. <img>/<video> src) and rely on the query param instead.

Roles: "admin" (full control incl. user management), "operator" (manage
cameras, acknowledge alerts, upload/analyze video), "viewer" (read-only
plus video upload/analysis, which is a customer-facing analysis tool
rather than an infrastructure-control action).
"""
from typing import Optional

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.database import get_db
from app.models.user import ROLE_ADMIN, ROLE_OPERATOR, User

# auto_error=False so we can fall back to the query-param token instead of
# FastAPI immediately raising 403 when no Authorization header is present.
_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    token: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> User:
    jwt_token = None
    if credentials is not None and credentials.credentials:
        jwt_token = credentials.credentials
    elif token:
        jwt_token = token

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not jwt_token:
        raise credentials_exception

    payload = decode_access_token(jwt_token)
    if payload is None:
        raise credentials_exception

    user_id = payload.get("user_id")
    if user_id is None:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
        )

    return user


def require_roles(*allowed_roles: str):
    """Dependency factory: require the current user's role to be one of
    `allowed_roles`. Usage: `Depends(require_roles(ROLE_ADMIN))`."""

    def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action",
            )
        return current_user

    return _check


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != ROLE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


def require_operator(current_user: User = Depends(get_current_user)) -> User:
    """Admin or operator - i.e. anyone allowed to manage cameras/alerts."""
    if current_user.role not in (ROLE_ADMIN, ROLE_OPERATOR):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator privileges required",
        )
    return current_user
