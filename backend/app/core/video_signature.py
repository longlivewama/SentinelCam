"""
Container-format sniffing for uploaded videos.

The upload route already rejects anything whose *filename extension* is
not in ALLOWED_VIDEO_EXTENSIONS, but an extension is a claim made by the
client, not a fact about the bytes. Without a content check, a ZIP, an
HTML page, a shell script or a polyglot payload renamed to `.mp4` is
accepted, written into storage, and later handed to the video decoder -
and, on the way back out, served from `/api/video-uploads/{id}/video`
with `Content-Type: video/mp4`.

So this module verifies the leading bytes against the signatures of the
container formats the app actually accepts. It is a *format* check, not a
malware scanner: a genuinely malformed-but-correctly-signed file still
has to be handled gracefully by the decoder (video_analysis.py fails the
upload with an error message rather than crashing the worker). What this
rules out is the whole class of "not remotely a video" content.

Deliberately implemented with fixed byte comparisons rather than a
libmagic binding: the accepted set is five well-documented container
formats, and this avoids a native dependency in the container image for
what amounts to twenty bytes of comparison.
"""
from __future__ import annotations

# ISO Base Media File Format (.mp4, .mov, .m4v): bytes 4..8 are the box
# type 'ftyp'. The first four bytes are that box's length, so they carry
# no signature and are skipped.
_ISOBMFF_BRAND_OFFSET = 4
_ISOBMFF_BRAND = b"ftyp"

# Matroska / WebM (.mkv, .webm) are both EBML documents.
_EBML_MAGIC = b"\x1a\x45\xdf\xa3"

# RIFF containers (.avi): "RIFF" <4-byte size> "AVI ".
_RIFF_MAGIC = b"RIFF"
_RIFF_AVI_FORM = b"AVI "

# QuickTime files occasionally start with one of these top-level atoms
# instead of 'ftyp' (older .mov writers).
_QUICKTIME_ATOMS = (b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot")

MIN_HEADER_BYTES = 12


def looks_like_supported_video(header: bytes) -> bool:
    """True if `header` (the first bytes of the file - at least
    MIN_HEADER_BYTES) begins with the signature of a container format the
    app accepts. Conservative: an unrecognised signature returns False."""
    if not header or len(header) < MIN_HEADER_BYTES:
        return False

    if header[_ISOBMFF_BRAND_OFFSET:_ISOBMFF_BRAND_OFFSET + 4] == _ISOBMFF_BRAND:
        return True

    if header.startswith(_EBML_MAGIC):
        return True

    if header.startswith(_RIFF_MAGIC) and header[8:12] == _RIFF_AVI_FORM:
        return True

    if header[4:8] in _QUICKTIME_ATOMS:
        return True

    return False
