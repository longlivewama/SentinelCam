"""
Response-level hardening applied to every route: baseline security
headers, and a catch-all handler that keeps internal detail out of error
responses while making sure it still reaches the logs.

Both are cross-cutting concerns that belong in one place rather than
being repeated per route - and both were missing entirely, which meant

  * every response, including the video/download endpoints that serve
    user-supplied bytes, went out with no `X-Content-Type-Options`,
    `X-Frame-Options` or `Referrer-Policy`, and
  * any unhandled exception produced Starlette's default 500, whose body
    text varies with the deployment's debug configuration.
"""
from __future__ import annotations

import logging
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Deliberately conservative and API-appropriate:
#   nosniff        - the streaming endpoints return bytes a user uploaded;
#                    browsers must not content-sniff them into something
#                    executable.
#   DENY           - this API is never meant to be framed. The SPA is
#                    served by its own nginx, not from here.
#   Referrer-Policy- URLs here carry a ?token= JWT on the stream/video
#                    endpoints; no referrer must leak that cross-origin.
#   Permissions    - the API needs none of these device capabilities.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Logs the full traceback server-side and returns a generic body with
    a correlation ID, so an operator can find the exact stack trace for a
    user's report without the response itself carrying internal paths,
    SQL, or configuration."""
    error_id = uuid.uuid4().hex[:12]
    logger.exception(
        "Unhandled exception [%s] on %s %s", error_id, request.method, request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again.",
            "error_id": error_id,
        },
    )
