"""
Shared FastAPI dependencies: current-user resolution and role-based access
control (RBAC) gating.

Critical detail: browsers cannot attach a custom `Authorization` header to
`<img src>`, `<video src>`, or plain `<a href>` downloads, so those
endpoints have to take a credential some other way - in the URL, which is
the one place a credential is guaranteed to be written down. A URL reaches
the access log of every server and proxy on the path, the browser's own
history, and any `Referer` sent onward from the page.

Narrowing WHERE that credential is accepted was the first half of the fix
and is still true: `get_current_user` refuses `?token=` outright, so a URL
credential cannot be replayed against the JSON API.

`MediaAccess` is the second half - narrowing WHAT the credential is. The
media endpoints no longer accept a session JWT in the query string at all.
They accept only a media token (see core/security.py): minted per
resource, expiring in minutes rather than hours, carrying no role and no
email, and refused by every endpoint except the one resource named in its
`res` claim. So the worst a recovered media URL can now do is re-read the
clip it was already a URL for, for a few minutes.

The `Authorization: Bearer` header still works everywhere it did before,
including on these endpoints - API clients and the test suite reach them
that way, and a header is not written into logs or history.

Roles: "admin" (full control incl. user management), "operator" (manage
cameras, acknowledge alerts, upload/analyze video), "viewer" (read-only
plus video upload/analysis, which is a customer-facing analysis tool
rather than an infrastructure-control action).
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import (
    create_media_token,
    decode_access_token,
    decode_media_token,
    is_media_token,
    media_resource_id,
)
from app.database import get_db
from app.models.user import ROLE_ADMIN, ROLE_OPERATOR, User
from app.schemas.media import MediaTokenOut

# Resource kinds used in a media token's `res` claim. Named here rather
# than spelled inline at each route so a token minted for one kind can
# never be checked against a typo of another.
MEDIA_KIND_RECORDING = "recording"
MEDIA_KIND_UPLOAD = "video-upload"
MEDIA_KIND_CAMERA = "camera"

# auto_error=False so we can fall back to the query-param token instead of
# FastAPI immediately raising 403 when no Authorization header is present.
_bearer_scheme = HTTPBearer(auto_error=False)


def _credentials_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _user_from_payload(payload: dict[str, Any], db: Session) -> User:
    """Resolves an already-verified token payload to its active user, or
    raises 401. Shared by session and media authentication so the account
    checks - exists, still active - can never apply to one and not the
    other."""
    user_id = payload.get("user_id")
    if user_id is None:
        raise _credentials_exception()

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise _credentials_exception()

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is disabled",
        )

    return user


def _authenticate(jwt_token: Optional[str], db: Session) -> User:
    """Resolves a SESSION JWT to its active user, or raises 401.

    Rejects a media token outright. A media token is deliberately weak -
    it travels in URLs and is handed out to anything that can read one
    clip - so honouring it here would hand back exactly the full-privilege
    access the split exists to prevent."""
    if not jwt_token:
        raise _credentials_exception()

    payload = decode_access_token(jwt_token)
    if payload is None:
        raise _credentials_exception()

    if is_media_token(payload):
        raise _credentials_exception()

    return _user_from_payload(payload, db)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """The default for every JSON endpoint: `Authorization: Bearer` only.

    A `?token=` query parameter is NOT accepted here, so a credential
    that leaks into a URL cannot be replayed against the rest of the
    API."""
    return _authenticate(credentials.credentials if credentials else None, db)


class MediaAccess:
    """Authentication for the media endpoints a browser can only reach by
    URL: the recording clip and download, the uploaded source video, and
    the MJPEG camera stream.

    Accepts either

      * an `Authorization: Bearer` session token, exactly as before - this
        is how API clients and the test suite reach these routes, and a
        header is never written into an access log or browser history; or
      * a `?token=` MEDIA token, which must be unexpired AND carry a `res`
        claim naming this very resource.

    A session JWT in the query string is refused. That is the change: it
    used to be accepted here, which meant every `<video src>` in the app
    published a full-privilege, day-long credential into the URL - and
    from there into Uvicorn's access log.

    The resource id is read from the request's own path parameters rather
    than taken as an argument, so the value checked against the token is
    necessarily the same one the route is about to serve. Passing it
    separately would leave room for a route to authorize clip A and then
    stream clip B.
    """

    def __init__(self, kind: str, path_param: str):
        self.kind = kind
        self.path_param = path_param

    def __call__(
        self,
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
        token: Optional[str] = Query(default=None),
        db: Session = Depends(get_db),
    ) -> User:
        header_token = credentials.credentials if credentials else None
        if header_token:
            return _authenticate(header_token, db)

        if not token:
            raise _credentials_exception()

        resource = media_resource_id(self.kind, request.path_params.get(self.path_param))
        payload = decode_media_token(token, resource)
        if payload is None:
            raise _credentials_exception()

        return _user_from_payload(payload, db)


def issue_media_token(user: User, kind: str, resource_id: Any) -> MediaTokenOut:
    """Mints a media token for one resource, on behalf of `user`.

    The counterpart to `MediaAccess`, kept beside it on purpose: the two
    have to agree about the `res` format, and the way to guarantee that is
    for both to go through `media_resource_id` in one file.

    This performs NO authorization of its own. Every caller is a route
    that has already run the same ownership check its media endpoint runs,
    which is the only place that check can be made correctly - "may this
    user see this clip" is a question about recordings, uploads and
    cameras, not about tokens."""
    token, expires_in = create_media_token(user.id, media_resource_id(kind, resource_id))
    return MediaTokenOut(
        token=token,
        expires_in=expires_in,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        resource=media_resource_id(kind, resource_id),
    )


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
