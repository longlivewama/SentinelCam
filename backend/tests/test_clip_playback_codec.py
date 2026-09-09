"""
Regression tests for browser-playable event clips.

THE BUG
-------
`open_writer` tried `avc1` (H.264) and fell straight back to `mp4v`. In
the shipped deployment `avc1` never opens: the PyPI
`opencv-python-headless` wheel bundles an FFmpeg built without libx264
(it is GPL, so the wheels omit it), and a container has no hardware
encoder either - exactly what the production log showed:

    h264_v4l2m2m: Could not find a valid device
    ... fallback to use tag 'mp4v'

So every fall clip was written as MPEG-4 Part 2. That container is a
perfectly valid `.mp4` and the range endpoint served it correctly with
206s - but **Chrome, Firefox and Edge cannot decode mp4v**, so the
`<video>` element rendered an empty player. The source video (H.264 from
a phone) played fine, which is what made this look like a results-loading
bug rather than a codec one.

Confirmed on the real artifacts of upload 24 by parsing the MP4 `stsd`
box:

    source.mp4  -> stsd fourcc 'avc1' + 'mp4a'   (plays)
    clip.mp4    -> stsd fourcc 'mp4v'            (does not play)

THE FIX
-------
`open_writer` now walks a ladder of codecs ordered by browser
compatibility - `avc1` (mp4), then VP9 and VP8 (webm, both present in the
stock wheel), and only then `mp4v` as a last resort - and returns the
path it actually wrote, since the container extension has to match the
codec. Serving follows the file's extension rather than assuming mp4.

WHAT THESE TESTS PIN
--------------------
The codec ladder's contract, not one specific codec: which rung opens
depends on the FFmpeg build, and asserting "it must be VP9" would fail on
a machine that has libx264. What must hold everywhere is that the
extension matches the codec, that the returned path is the file actually
written, that the row points at it, and that it is served with the right
Content-Type.
"""
import struct
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.ranges import DEFAULT_MEDIA_TYPE, media_type_for
from app.models.recording import Recording
from app.services.recording_engine import _CODEC_LADDER, open_writer

# Codecs a current browser can actually decode. `mp4v` is deliberately
# absent - that is the whole point of the bug.
BROWSER_PLAYABLE_CODECS = {"avc1", "vp09", "VP80"}

EXPECTED_EXTENSION = {"avc1": ".mp4", "vp09": ".webm", "VP80": ".webm", "mp4v": ".mp4"}


def _frames(n=15, w=64, h=48):
    return [np.full((h, w, 3), (i * 17) % 255, dtype=np.uint8) for i in range(n)]


def _mp4_stsd_fourccs(path: Path):
    """The codec fourccs declared in an MP4's sample-description box.
    Used instead of ffprobe, which is not installed in the image."""
    data = path.read_bytes()

    def walk(start, end):
        off, found = start, []
        while off + 8 <= end:
            size = struct.unpack(">I", data[off:off + 4])[0]
            typ = data[off + 4:off + 8].decode("latin-1", "replace")
            if size == 0:
                size = end - off
            if size < 8:
                break
            body = off + 8
            if typ == "stsd":
                count = struct.unpack(">I", data[body + 4:body + 8])[0]
                entry = body + 8
                for _ in range(count):
                    entry_size = struct.unpack(">I", data[entry:entry + 4])[0]
                    found.append(data[entry + 4:entry + 8].decode("latin-1", "replace"))
                    entry += entry_size
            if typ in ("moov", "trak", "mdia", "minf", "stbl"):
                found += walk(body, off + size)
            off += size
        return found

    return walk(0, len(data))


# --- the ladder itself -----------------------------------------------------

def test_the_ladder_prefers_browser_playable_codecs_over_mp4v():
    """mp4v must be the LAST rung. Anything else ahead of it is a codec a
    browser can play; if mp4v ever moves up, clips silently stop playing
    again."""
    codecs = [codec for codec, _ in _CODEC_LADDER]

    assert codecs[-1] == "mp4v", f"mp4v must be the last resort, ladder is {codecs}"
    assert set(codecs[:-1]) <= BROWSER_PLAYABLE_CODECS
    assert "avc1" == codecs[0], "H.264 is the ideal when the build supports it"


def test_every_ladder_entry_pairs_its_codec_with_the_right_container():
    """FFmpeg picks the muxer from the extension, so a VP9 stream in a
    .mp4 (or H.264 in a .webm) simply fails to open."""
    for codec, extension in _CODEC_LADDER:
        assert extension == EXPECTED_EXTENSION[codec], f"{codec} must be muxed into {EXPECTED_EXTENSION[codec]}"


# --- open_writer's contract ------------------------------------------------

def test_open_writer_returns_the_path_it_actually_wrote(tmp_path):
    """The regression that would break playback a second way: the caller
    asks for `.mp4`, the writer may produce `.webm`, and a Recording row
    storing the requested name would 404 on a file sitting right there
    under another suffix."""
    requested = tmp_path / "clip.mp4"
    codec, writer, written = open_writer(str(requested), 64, 48, 10)

    assert writer is not None, "no codec in the ladder opened in this environment"
    for frame in _frames():
        writer.write(frame)
    writer.release()

    assert written is not None
    assert Path(written).exists(), "returned path must be the file that was written"
    assert Path(written).stat().st_size > 0
    assert Path(written).parent == requested.parent
    assert Path(written).stem == requested.stem


def test_the_written_container_matches_the_codec_that_opened(tmp_path):
    codec, writer, written = open_writer(str(tmp_path / "clip.mp4"), 64, 48, 10)
    assert writer is not None
    for frame in _frames():
        writer.write(frame)
    writer.release()

    assert Path(written).suffix == EXPECTED_EXTENSION[codec]


def test_the_clip_this_environment_produces_is_decodable(tmp_path):
    """A clip that opens but decodes to nothing would pass every check
    above and still be useless."""
    codec, writer, written = open_writer(str(tmp_path / "clip.mp4"), 64, 48, 10)
    assert writer is not None
    frames = _frames(n=15)
    for frame in frames:
        writer.write(frame)
    writer.release()

    cap = cv2.VideoCapture(written)
    decoded = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()

    assert decoded > 0, f"{codec} clip at {written} decoded zero frames"


def test_an_mp4_result_is_never_silently_mp4v_when_something_better_opened(tmp_path):
    """If the ladder settled on an `.mp4`, the stream inside it must be
    H.264 - the only browser-playable MP4 rung. An mp4v result is
    tolerated (it is the documented last resort) but must be the codec
    the writer actually reported, so the warning log is truthful."""
    codec, writer, written = open_writer(str(tmp_path / "clip.mp4"), 64, 48, 10)
    assert writer is not None
    for frame in _frames():
        writer.write(frame)
    writer.release()

    if Path(written).suffix != ".mp4":
        pytest.skip(f"this environment encoded {codec} into {Path(written).suffix}")

    fourccs = _mp4_stsd_fourccs(Path(written))
    assert fourccs, "no sample description found in the written MP4"
    if codec == "avc1":
        assert "avc1" in fourccs
    else:
        assert codec == "mp4v", f"unexpected mp4 codec {codec}"
        assert "mp4v" in fourccs


def test_a_failed_rung_leaves_no_stray_empty_file(tmp_path):
    """The avc1 attempt can create a zero-byte .mp4 before failing. If it
    were left behind, the directory would hold an empty .mp4 next to the
    real .webm - and anything globbing for clips would find it."""
    codec, writer, written = open_writer(str(tmp_path / "clip.mp4"), 64, 48, 10)
    assert writer is not None
    for frame in _frames():
        writer.write(frame)
    writer.release()

    leftovers = [p for p in tmp_path.iterdir() if p.stat().st_size == 0]
    assert leftovers == [], f"zero-byte file(s) left behind: {leftovers}"


# --- Content-Type follows the container ------------------------------------

def test_media_type_is_derived_from_the_container_extension():
    assert media_type_for("/clips/a.mp4") == "video/mp4"
    assert media_type_for("/clips/a.webm") == "video/webm"
    assert media_type_for("/clips/A.WEBM") == "video/webm", "extension match must be case-insensitive"
    assert media_type_for("/clips/a.mkv") == "video/x-matroska"


def test_an_unknown_extension_falls_back_rather_than_guessing():
    assert media_type_for("/clips/a.bin") == DEFAULT_MEDIA_TYPE
    assert media_type_for("/clips/noextension") == DEFAULT_MEDIA_TYPE


@pytest.fixture()
def webm_recording(db, tmp_path):
    """A recording stored as WebM - what this deployment now produces."""
    clip = tmp_path / "clip.webm"
    clip.write_bytes(b"\x1a\x45\xdf\xa3" + b"webm-ish payload" * 64)
    row = Recording(
        camera_id=None, video_upload_id=None, filename="clip.webm", file_path=str(clip),
        duration_seconds=4.0, trigger_action="fall", file_size_bytes=clip.stat().st_size,
        event_timestamp=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def test_a_webm_clip_is_streamed_as_video_webm(client, operator_headers, webm_recording):
    """Serving WebM bytes under `video/mp4` makes a browser refuse a file
    it can otherwise play - the second half of the playback bug."""
    resp = client.get(f"/api/recordings/{webm_recording.id}/video", headers=operator_headers)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("video/webm")
    assert resp.headers["accept-ranges"] == "bytes"


def test_a_ranged_webm_request_keeps_its_content_type(client, operator_headers, webm_recording):
    """Range requests are how a browser actually loads video, so the
    partial responses must carry the right type too."""
    resp = client.get(
        f"/api/recordings/{webm_recording.id}/video",
        headers={**operator_headers, "Range": "bytes=0-99"},
    )

    assert resp.status_code == 206
    assert resp.headers["content-type"].startswith("video/webm")
    assert resp.headers["content-range"].startswith("bytes 0-99/")
    assert resp.headers["content-length"] == "100"


def test_downloading_a_webm_clip_uses_its_real_media_type(client, operator_headers, webm_recording):
    resp = client.get(f"/api/recordings/{webm_recording.id}/download", headers=operator_headers)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("video/webm")
    assert "attachment" in resp.headers["content-disposition"]


def test_existing_mp4_recordings_are_still_served_as_mp4(client, operator_headers, db, tmp_path):
    """Clips written before this change (upload 24's, for instance) must
    keep working exactly as they did - the fix must not orphan them."""
    clip = tmp_path / "legacy.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"legacy payload" * 64)
    row = Recording(
        camera_id=None, video_upload_id=None, filename="legacy.mp4", file_path=str(clip),
        duration_seconds=4.0, trigger_action="fall", file_size_bytes=clip.stat().st_size,
        event_timestamp=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    resp = client.get(f"/api/recordings/{row.id}/video", headers=operator_headers)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("video/mp4")


def test_streaming_a_clip_still_requires_authentication(client, webm_recording):
    """The fix touches Content-Type only; access control must be
    untouched."""
    assert client.get(f"/api/recordings/{webm_recording.id}/video").status_code == 401
    assert client.get(f"/api/recordings/{webm_recording.id}/download").status_code == 401
