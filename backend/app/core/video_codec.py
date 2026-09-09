"""
Which video codec is actually inside a stored clip.

The container extension says nothing about decodability: `.mp4` is a
perfectly valid wrapper around MPEG-4 Part 2 ("mp4v"), which no current
browser can decode. That is not hypothetical - it is the bug that made
every generated fall clip render as an empty black player while the
H.264 source video beside it played fine (see
recording_engine.open_writer's codec ladder).

So anything that needs to answer "will this file play?" has to look at
the sample entry inside the file rather than at its name. This module
reads that fourcc out of an ISO-BMFF sample-description box.

Seek-based rather than read-the-whole-file: OpenCV writes `moov` at the
end of the file (it is only complete on release), so finding it means
walking the box tree past `mdat` - and `mdat` is the entire video. This
parses headers and skips payloads, so probing a 500 MB clip costs a
handful of small reads.

Deliberately not shelling out to ffprobe: it is not installed in the
runtime image, and adding it for the sake of reading four bytes would be
a native dependency for what fits in this file.
"""
from __future__ import annotations

import os
from typing import BinaryIO, Iterator, NamedTuple, Optional

# Video sample entries a current browser can decode. VP8/VP9 in WebM are
# not listed: WebM carries its codec in EBML, not in an ISO-BMFF sample
# entry, and is handled by extension below.
BROWSER_PLAYABLE_FOURCCS = frozenset({"avc1", "avc3", "hvc1", "hev1", "av01"})

# Valid MP4 video, undecodable in Chrome/Firefox/Edge. `mp4v` is what the
# old encoder fallback produced; the rest are the same MPEG-4 Part 2
# lineage under other writers' tags, worth naming so a clip written by
# some other tool is diagnosed rather than reported as "unknown".
UNPLAYABLE_VIDEO_FOURCCS = frozenset({"mp4v", "divx", "DIVX", "xvid", "XVID", "DX50", "s263", "h263"})

# Audio sample entries, skipped when looking for the video track's codec.
_AUDIO_FOURCCS = frozenset({"mp4a", "ac-3", "ec-3", "alac", "Opus", "opus", ".mp3", "sowt", "twos"})

# Boxes whose payload is more boxes, on the path from the file root down
# to a sample description.
_CONTAINER_BOXES = frozenset({"moov", "trak", "mdia", "minf", "stbl"})

_ISOBMFF_EXTENSIONS = frozenset({".mp4", ".m4v", ".mov"})
# Browsers play these containers when the streams inside them are ones
# they support; everything this app writes into WebM is VP8 or VP9.
_ASSUMED_PLAYABLE_EXTENSIONS = frozenset({".webm"})


class ClipCodec(NamedTuple):
    """What a probe could determine about one stored clip."""

    fourcc: Optional[str]
    # None where it could not be determined - an unreadable file, or a
    # container this module does not parse. Callers must not treat that
    # as "unplayable" and re-encode it blindly.
    playable: Optional[bool]
    detail: str

    @property
    def needs_transcode(self) -> bool:
        return self.playable is False


def _iter_boxes(fh: BinaryIO, end: int) -> Iterator[tuple]:
    """Yields `(type, body_offset, box_end)` for each box between the
    handle's current position and `end`, leaving the handle positioned
    after each box."""
    while True:
        start = fh.tell()
        if start + 8 > end:
            return
        header = fh.read(8)
        if len(header) < 8:
            return

        size = int.from_bytes(header[0:4], "big")
        box_type = header[4:8].decode("latin-1", "replace")
        body = start + 8

        if size == 1:
            # 64-bit size: the real length follows the type.
            large = fh.read(8)
            if len(large) < 8:
                return
            size = int.from_bytes(large, "big")
            body = start + 16
        elif size == 0:
            # "extends to end of file", legal for the last box.
            size = end - start

        if size < 8 or start + size > end:
            return

        yield box_type, body, start + size
        fh.seek(start + size)


def _sample_entry_fourccs(fh: BinaryIO, start: int, end: int) -> list:
    """Every sample-entry fourcc under the box range, depth-first."""
    found = []
    fh.seek(start)
    for box_type, body, box_end in _iter_boxes(fh, end):
        if box_type == "stsd":
            fh.seek(body + 4)  # version + flags
            count_bytes = fh.read(4)
            if len(count_bytes) < 4:
                continue
            entry = body + 8
            for _ in range(int.from_bytes(count_bytes, "big")):
                fh.seek(entry)
                entry_header = fh.read(8)
                if len(entry_header) < 8:
                    break
                entry_size = int.from_bytes(entry_header[0:4], "big")
                found.append(entry_header[4:8].decode("latin-1", "replace"))
                if entry_size < 8:
                    break
                entry += entry_size
        elif box_type in _CONTAINER_BOXES:
            found.extend(_sample_entry_fourccs(fh, body, box_end))
            fh.seek(box_end)

    return found


def mp4_sample_fourccs(path: str) -> list:
    """Sample-entry fourccs declared in an ISO-BMFF file, video and audio
    alike, in the order they appear. Empty if the file has none or could
    not be read."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            return _sample_entry_fourccs(fh, 0, size)
    except (OSError, ValueError):
        return []


def inspect_clip(path: str) -> ClipCodec:
    """Reports whether `path` holds video a browser can decode."""
    extension = os.path.splitext(path)[1].lower()

    if extension in _ASSUMED_PLAYABLE_EXTENSIONS:
        return ClipCodec(None, True, f"{extension} container, assumed browser-playable")

    if extension not in _ISOBMFF_EXTENSIONS:
        return ClipCodec(None, None, f"unrecognised container {extension or '(no extension)'}")

    fourccs = mp4_sample_fourccs(path)
    if not fourccs:
        return ClipCodec(None, None, "no sample description found; file may be truncated or not an MP4")

    video = [code for code in fourccs if code not in _AUDIO_FOURCCS]
    if not video:
        return ClipCodec(None, None, f"no video sample entry (found {', '.join(fourccs)})")

    fourcc = video[0]
    if fourcc in BROWSER_PLAYABLE_FOURCCS:
        return ClipCodec(fourcc, True, f"{fourcc} - browser-playable")
    if fourcc in UNPLAYABLE_VIDEO_FOURCCS:
        return ClipCodec(fourcc, False, f"{fourcc} - no current browser can decode this")

    return ClipCodec(fourcc, None, f"{fourcc} - unrecognised codec, left alone")
