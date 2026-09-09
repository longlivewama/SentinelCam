"""
HTTP Range request parsing for the two byte-serving endpoints (recording
clips and uploaded source videos).

Extracted and hardened: both endpoints previously did
`int(range_header.replace("bytes=", "").partition("-")[0])` inline, which

  * raised ValueError -> HTTP 500 on any non-numeric Range header (a
    single `Range: bytes=abc-` from a scanner, or a legitimate but
    unsupported multi-range header like `bytes=0-99,200-299`),
  * mis-read a suffix range (`bytes=-500`, "the last 500 bytes") as
    "bytes 0-500", serving the wrong part of the file, and
  * answered a start beyond EOF with a 206 and a nonsensical
    Content-Range instead of 416.

RFC 9110 says a malformed Range header must be *ignored* (serve the whole
representation), and an unsatisfiable one answered with 416 - which is
what this does.
"""
from __future__ import annotations

import os
from typing import NamedTuple, Optional

from fastapi import Request, Response, status
from fastapi.responses import StreamingResponse

CHUNK_SIZE = 1024 * 1024  # 1 MB

# Clips are written in whatever container the deployment's FFmpeg can
# actually encode (see recording_engine.open_writer), so the served
# Content-Type has to follow the file rather than being assumed. Serving
# a WebM as video/mp4 makes browsers refuse to decode a file they
# otherwise play perfectly.
_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
}
DEFAULT_MEDIA_TYPE = "video/mp4"


def media_type_for(file_path: str) -> str:
    """Content-Type for a stored clip, from its container extension."""
    return _MEDIA_TYPES.get(os.path.splitext(file_path)[1].lower(), DEFAULT_MEDIA_TYPE)


class ByteRange(NamedTuple):
    start: int
    end: int  # inclusive

    @property
    def length(self) -> int:
        return self.end - self.start + 1


class RangeNotSatisfiable(Exception):
    """The header was well-formed but asks for bytes outside the file."""


def parse_range_header(range_header: Optional[str], file_size: int) -> Optional[ByteRange]:
    """Returns the requested range, or None to serve the whole file
    (no header, or a malformed/unsupported one - per RFC 9110 a malformed
    Range is ignored rather than rejected). Raises RangeNotSatisfiable for
    a well-formed range that lies outside the file, which the caller
    should answer with 416."""
    if not range_header or file_size <= 0:
        return None

    value = range_header.strip().lower()
    if not value.startswith("bytes="):
        return None
    value = value[len("bytes="):].strip()

    # Multi-range ("bytes=0-99,200-299") is legal but unsupported here;
    # ignoring it and sending the whole file is a valid response.
    if "," in value:
        return None

    start_str, sep, end_str = value.partition("-")
    if not sep:
        return None

    try:
        if not start_str:
            # Suffix range: "bytes=-N" means the LAST n bytes.
            if not end_str:
                return None
            suffix_length = int(end_str)
            if suffix_length <= 0:
                raise RangeNotSatisfiable(range_header)
            start = max(0, file_size - suffix_length)
            end = file_size - 1
        else:
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1
    except ValueError:
        return None

    if start < 0 or end < 0:
        return None
    if start >= file_size or start > end:
        raise RangeNotSatisfiable(range_header)

    return ByteRange(start=start, end=min(end, file_size - 1))


def serve_file_range(
    file_path: str,
    request: Request,
    media_type: Optional[str] = None,
) -> Response:
    """Serves `file_path`, honouring a Range header when present. Used by
    both the recording-clip and uploaded-source-video endpoints, which had
    identical (and identically buggy) copies of this logic.

    Callers must have already resolved and authorized `file_path` - this
    function does no access control and no path resolution.

    `media_type` defaults to whatever the file's container implies, so a
    WebM clip is not announced as video/mp4."""
    if media_type is None:
        media_type = media_type_for(file_path)

    file_size = os.path.getsize(file_path)

    try:
        byte_range = parse_range_header(request.headers.get("range"), file_size)
    except RangeNotSatisfiable:
        # RFC 9110 416: tell the client the size it should have asked within.
        return Response(
            status_code=416,  # Range Not Satisfiable (the starlette constant's name changed between versions)
            headers={"Content-Range": f"bytes */{file_size}", "Accept-Ranges": "bytes"},
        )

    if byte_range is None:
        start, end = 0, max(file_size - 1, 0)
        status_code = status.HTTP_200_OK
        headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size)}
    else:
        start, end = byte_range.start, byte_range.end
        status_code = status.HTTP_206_PARTIAL_CONTENT
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(byte_range.length),
        }

    content_length = max(end - start + 1, 0)

    def iterfile():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = f.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(iterfile(), status_code=status_code, headers=headers, media_type=media_type)
