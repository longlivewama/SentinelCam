"""
Shared FastAPI dependencies: current-user resolution and role-based access
control (RBAC) gating.

Critical detail: browsers cannot attach a custom `Authorization` header to
`<img src>`, `<video src>`, or plain `<a href>` downloads, so those
endpoints have to take the JWT some other way. That is what
`get_current_user_allowing_query_token` is for - it checks the header
first and falls back to `?token=<jwt>`.

`get_current_user` itself deliberately does NOT accept the query
parameter. A token in a URL is a token in browser history, in `Referer`
headers, and in every access log and proxy along the way, so the places
that leak it should be the few that cannot avoid it - the four media and
stream endpoints - rather than the entire API. Everything else is reached
by axios, which sends the header, so requiring it costs nothing and keeps
a full-privilege credential out of URLs that never needed to carry one.

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


def _authenticate(jwt_token: Optional[str], db: Session) -> User:
    """Resolves a JWT to its active user, or raises 401.

    Shared by both dependencies below so that where a token is allowed to
    come from is the only difference between them - the validation of it
    never diverges."""
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


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """The default for every JSON endpoint: `Authorization: Bearer` only.

    A `?token=` query parameter is NOT accepted here, so a credential
    that leaks into a URL cannot be replayed against the rest of the
    API."""
    return _authenticate(credentials.credentials if credentials else None, db)


def get_current_user_allowing_query_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    token: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> User:
    """For the handful of endpoints a browser reaches by raw URL - the
    recording clip and download, the uploaded source video, and the MJPEG
    camera stream - where no header can be attached. Prefers the header
    when one is present."""
    header_token = credentials.credentials if credentials else None
    return _authenticate(header_token or token, db)


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
