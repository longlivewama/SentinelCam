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

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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


class SecurityHeadersMiddleware:
    """Adds SECURITY_HEADERS to every HTTP response that does not already
    set them.

    Pure ASGI, deliberately not BaseHTTPMiddleware: that base class puts
    its own task and `receive` wrapper between the route and the server,
    and a streaming response behind it never learned its client had gone.
    The live MJPEG endpoint polls `request.is_disconnected()` to end
    abandoned views of a camera that has no frame to send; through
    BaseHTTPMiddleware that never reported the disconnect, so each such
    view ran forever and uvicorn's graceful shutdown waited on it
    indefinitely. This only edits the response-start message's headers."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for header, value in SECURITY_HEADERS.items():
                    headers.setdefault(header, value)
            await send(message)

        await self.app(scope, receive, send_with_headers)


class RequestBodyLimitMiddleware:
    """Refuses a request body larger than `max_body_bytes` while it is
    still arriving, rather than after it has all been stored.

    The upload route's own size check cannot do this on its own: FastAPI
    parses a multipart form - spooling every byte of the file to a
    temporary file - before the route function runs, so a body of any size
    was written to the server's temp directory in full and only then
    rejected with a 413 (measured: all 600MB of a 600MB body). Any
    self-registered account could fill that disk.

    A declared Content-Length over the limit is refused before a byte of
    the body is read; a chunked body is counted as it streams and cut off
    at the limit. Pure ASGI rather than BaseHTTPMiddleware, because the
    latter would buffer nothing usefully here - the point is to sit in
    front of `receive`."""

    def __init__(self, app: ASGIApp, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    def _too_large(self) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Request body exceeds the {self.max_body_bytes // (1024 * 1024)}MB limit",
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            error = self._too_large()
            # Connection: close - the unread body is still on the wire, so
            # this connection cannot be reused for another request.
            response = JSONResponse(
                {"detail": error.detail}, status_code=error.status_code, headers={"Connection": "close"},
            )
            await response(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    # Re-raised untouched by FastAPI's body parsing and
                    # rendered as a 413 by the normal exception handling.
                    raise self._too_large()
            return message

        await self.app(scope, limited_receive, send)


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
