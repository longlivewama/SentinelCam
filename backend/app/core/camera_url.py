"""
Camera source handling: what a camera `url` is allowed to be, and how it
is written down.

A camera's `url` is the one configuration value in this application that
is BOTH a secret and an instruction to open a network connection, and it
needed treating as both:

CREDENTIAL. An IP camera is conventionally addressed with its credentials
inline - `rtsp://admin:hunter2@192.168.1.50:554/Streaming/Channels/101` -
because that is the only way the RTSP URL form carries them. That string
is the camera's password. It must not reach a lower-privileged API caller
(`redact_credentials`, used by the cameras routes for non-operators) and
it must not reach a log file (same helper, used at the capture loop's
failure log and, belt-and-braces, by the process-wide log redaction in
logging_utils.py).

INSTRUCTION. `cv2.VideoCapture` hands the string to FFmpeg, which
supports far more protocols than a camera needs - `file:`, `concat:`,
`gopher:` and friends - and treats a bare path as a local file. So an
operator-set URL was, in effect, "open this arbitrary thing on the
server's behalf": local file disclosure and a request forgery primitive
against internal services. `validate_source` narrows it to the schemes a
camera actually speaks, plus the bare device index a USB webcam uses.

Deliberately NOT here: any block on private/loopback address ranges. IP
cameras live on private LANs - that is the normal deployment, not an
attack - so refusing RFC1918 destinations would break the product's main
use case while an attacker who can already set camera configuration has
better options. Narrowing the PROTOCOL is the part that removes
capability without removing the feature.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

# Schemes a camera or video stream plausibly speaks. Everything else -
# `file:`, `concat:`, `subfile:`, `cache:`, `data:`, `gopher:`, `ftp:`,
# `javascript:` - is refused: none of them address a camera, and each is
# a way to point the server's decoder at something that is not one.
ALLOWED_SCHEMES = ("rtsp", "rtsps", "rtmp", "rtmps", "http", "https", "udp", "rtp")

# A USB webcam is addressed by its device index, given as a string ("0").
_DEVICE_INDEX_RE = re.compile(r"^\d{1,3}$")

# `scheme://userinfo@host...`, where userinfo is everything before the
# last `@` of the authority. Matched with a regex rather than by
# round-tripping through urlsplit/urlunsplit so this cannot raise, cannot
# normalise the rest of the URL into something subtly different, and works
# on the malformed values a log line may well contain.
_URL_CREDENTIALS_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)([^/\s@]+@)")

REDACTED_USERINFO = "***:***@"


def redact_credentials(value: Any) -> Any:
    """`rtsp://admin:hunter2@host/path` -> `rtsp://***:***@host/path`.

    Only the userinfo is removed. The scheme, host, port and path stay
    readable, because those are what makes a log line or an API response
    useful to whoever is looking at it - and they are not the secret.

    Non-string input (a USB device index arrives as an int) is returned
    unchanged, and this never raises: it is called from logging paths."""
    if not isinstance(value, str):
        return value
    try:
        return _URL_CREDENTIALS_RE.sub(rf"\1{REDACTED_USERINFO}", value)
    except Exception:  # pragma: no cover - defensive; must never break logging
        return value


def has_credentials(value: str) -> bool:
    """Whether `value` carries inline userinfo. Used by tests and by
    callers that want to say "this camera has stored credentials" without
    quoting them."""
    return isinstance(value, str) and bool(_URL_CREDENTIALS_RE.search(value))


class InvalidCameraSource(ValueError):
    """The camera `url` is not something this application will open."""


def validate_source(url: str) -> str:
    """Returns `url` unchanged if it is an acceptable camera source, else
    raises InvalidCameraSource.

    Acceptable is either

      * a bare device index for a USB camera ("0", "1", ...) - what
        `stream_manager._resolve_source` turns into the int
        `cv2.VideoCapture` wants; or
      * a URL whose scheme is in ALLOWED_SCHEMES and which names a host.

    A bare path ("/etc/passwd", "storage/x.mp4") is refused along with
    every unlisted scheme: without a scheme FFmpeg reads it as a local
    file, which is how "configure a camera" became "read a server file"."""
    if not isinstance(url, str):
        raise InvalidCameraSource("Camera URL must be a string")

    candidate = url.strip()
    if not candidate:
        raise InvalidCameraSource("Camera URL must not be empty")

    if _DEVICE_INDEX_RE.match(candidate):
        return candidate

    parts = urlsplit(candidate)
    scheme = parts.scheme.lower()
    if not scheme:
        raise InvalidCameraSource(
            "Camera URL must be a stream URL (e.g. rtsp://host/path) or a USB device index (e.g. 0)"
        )
    if scheme not in ALLOWED_SCHEMES:
        raise InvalidCameraSource(
            f"Unsupported camera URL scheme '{scheme}'. Allowed: {', '.join(ALLOWED_SCHEMES)}, "
            "or a USB device index (e.g. 0)"
        )
    if not parts.hostname:
        raise InvalidCameraSource(f"Camera URL '{scheme}://...' must include a host")

    return candidate
