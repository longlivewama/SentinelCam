"""
The response shape for a media-token mint.

`expires_in` is returned alongside the token so the client never has to
hard-code the lifetime. The backend can shorten MEDIA_TOKEN_EXPIRE_SECONDS
without a frontend release, and a player that wants to refresh before
expiry has the number it needs rather than a guess.
"""
from datetime import datetime

from pydantic import BaseModel


class MediaTokenOut(BaseModel):
    # The token itself. Scoped to one resource and valid for `expires_in`
    # seconds - see core/security.py's create_media_token.
    token: str
    expires_in: int
    expires_at: datetime
    # Echoed back so a client holding several tokens can tell them apart
    # (and so a mistake in scoping is visible in the response rather than
    # only in a later 401).
    resource: str
