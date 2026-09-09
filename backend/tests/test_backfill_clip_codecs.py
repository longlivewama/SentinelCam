"""
Tests for scripts/backfill_clip_codecs.py.

Clips written before the codec ladder went to disk as MPEG-4 Part 2 -
valid `.mp4`, served with correct 206 responses, and undecodable in every
current browser. Fixing the encoder only helps new clips; these rows keep
pointing at a file that renders an empty player until something rewrites
them.

What matters most here is what happens when a re-encode goes wrong. A
migration that leaves a row pointing at a file that does not exist, or at
a half-written one, turns "this clip does not play" into "this clip is
gone" - so the failure paths get as much attention as the happy one.
"""
import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.video_codec import inspect_clip
from app.models.recording import Recording

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backfill_clip_codecs.py"


def _load():
    spec = importlib.util.spec_from_file_location("backfill_clip_codecs", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def backfill():
    return _load()


@pytest.fixture(autouse=True)
def _isolated_recordings_dir(backfill, tmp_path, monkeypatch):
    """The orphan scan walks RECORDINGS_DIR; point it somewhere empty so
    these tests never depend on what a previous run left in storage/."""
    monkeypatch.setattr(backfill.settings, "RECORDINGS_DIR", str(tmp_path / "recordings"))


def _write_mp4v(path: Path, frames=20, width=64, height=48, fps=10):
    """A clip in the codec this migration exists to remove."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "mp4v must be available to build the fixture"
    for i in range(frames):
        writer.write(np.full((height, width, 3), (i * 11) % 255, dtype=np.uint8))
    writer.release()
    return path


def _decodable_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


@pytest.fixture()
def unplayable_recording(db, tmp_path):
    clip = _write_mp4v(tmp_path / "clips" / "20260101T000000_fall.mp4")
    row = Recording(
        camera_id=None, video_upload_id=None, filename=clip.name, file_path=str(clip),
        duration_seconds=2.0, trigger_action="fall", file_size_bytes=clip.stat().st_size,
        event_timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# --- reporting mode --------------------------------------------------------

def test_a_bare_run_changes_nothing(backfill, db, unplayable_recording):
    """The default has to be inert: this rewrites files and rows."""
    before_path = unplayable_recording.file_path
    before_bytes = Path(before_path).read_bytes()

    assert backfill.main([]) == 0

    db.refresh(unplayable_recording)
    assert unplayable_recording.file_path == before_path
    assert Path(before_path).read_bytes() == before_bytes
    assert inspect_clip(before_path).needs_transcode, "still the original mp4v file"


# --- converting ------------------------------------------------------------

def test_apply_makes_the_clip_playable_and_repoints_the_row(backfill, db, unplayable_recording):
    original = Path(unplayable_recording.file_path)
    source_frames = _decodable_frames(original)

    assert backfill.main(["--apply"]) == 0

    db.refresh(unplayable_recording)
    new_path = Path(unplayable_recording.file_path)

    assert new_path.exists(), "the row must point at a file that exists"
    assert not inspect_clip(str(new_path)).needs_transcode, "the whole point: it must now be playable"
    assert unplayable_recording.filename == new_path.name, "filename must follow the file"
    assert unplayable_recording.file_size_bytes == new_path.stat().st_size
    assert _decodable_frames(new_path) == source_frames, "no frames may be lost"


def test_the_re_encode_preserves_the_picture_size(backfill, db, tmp_path):
    clip = _write_mp4v(tmp_path / "clips" / "wide.mp4", width=160, height=120)
    row = Recording(
        camera_id=None, video_upload_id=None, filename=clip.name, file_path=str(clip),
        duration_seconds=2.0, trigger_action="fall", file_size_bytes=clip.stat().st_size,
        event_timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    db.add(row)
    db.commit()

    backfill.main(["--apply"])
    db.refresh(row)

    capture = cv2.VideoCapture(row.file_path)
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape[:2] == (120, 160)


def test_the_superseded_original_is_kept_unless_asked(backfill, db, unplayable_recording):
    """Losing the only copy of a clip to a migration is worse than the bug
    it fixes, so the old file survives by default."""
    original = Path(unplayable_recording.file_path)

    backfill.main(["--apply"])

    assert original.exists(), "the original must still be there"


def test_delete_originals_leaves_only_the_file_the_row_names(backfill, db, unplayable_recording):
    """Asserting "the old .mp4 is gone" would be wrong where the ladder
    reaches avc1: the replacement is itself an .mp4 and takes over that
    exact path. What must hold in either environment is that one file is
    left and the row points at it."""
    clip_dir = Path(unplayable_recording.file_path).parent

    backfill.main(["--apply", "--delete-originals"])
    db.refresh(unplayable_recording)

    survivors = sorted(p.name for p in clip_dir.iterdir())
    assert survivors == [Path(unplayable_recording.file_path).name], survivors
    assert Path(unplayable_recording.file_path).exists()
    assert not inspect_clip(unplayable_recording.file_path).needs_transcode


def test_running_it_twice_is_a_no_op(backfill, db, unplayable_recording):
    """Migrations get re-run. The second pass must find nothing to do
    rather than re-encoding an already-converted clip."""
    backfill.main(["--apply"])
    db.refresh(unplayable_recording)
    after_first = (unplayable_recording.file_path, Path(unplayable_recording.file_path).read_bytes())

    backfill.main(["--apply"])
    db.refresh(unplayable_recording)

    assert unplayable_recording.file_path == after_first[0]
    assert Path(unplayable_recording.file_path).read_bytes() == after_first[1]


# --- what must NOT be touched ---------------------------------------------

def test_clips_that_already_play_are_left_alone(backfill, db, tmp_path):
    """A `.webm` clip written since the fix must not be re-encoded - that
    would burn a generation of quality for nothing."""
    from app.services.recording_engine import open_writer

    clip_dir = tmp_path / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    codec, writer, written = open_writer(str(clip_dir / "already_fine.mp4"), 64, 48, 10)
    assert writer is not None
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    if inspect_clip(written).needs_transcode:
        pytest.skip("this environment has no browser-playable encoder")

    row = Recording(
        camera_id=None, video_upload_id=None, filename=Path(written).name, file_path=written,
        duration_seconds=1.0, trigger_action="fall", file_size_bytes=Path(written).stat().st_size,
        event_timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    db.add(row)
    db.commit()
    before = Path(written).read_bytes()

    backfill.main(["--apply"])
    db.refresh(row)

    assert row.file_path == written
    assert Path(written).read_bytes() == before


def test_a_row_whose_file_is_missing_is_reported_not_crashed_on(backfill, db, tmp_path):
    row = Recording(
        camera_id=None, video_upload_id=None, filename="gone.mp4",
        file_path=str(tmp_path / "clips" / "gone.mp4"),
        duration_seconds=1.0, trigger_action="fall", file_size_bytes=0,
        event_timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    db.add(row)
    db.commit()

    assert backfill.main(["--apply"]) == 0

    db.refresh(row)
    assert row.file_path.endswith("gone.mp4"), "a missing file must not be rewritten away"


def test_an_unreadable_clip_leaves_the_row_pointing_at_the_original(backfill, db, tmp_path, caplog):
    """A truncated/garbage file must fail loudly and change nothing - the
    dangerous outcome is a row aimed at a file that was never written."""
    broken = tmp_path / "clips" / "broken.mp4"
    broken.parent.mkdir(parents=True, exist_ok=True)
    # Valid ftyp header so it is recognised as MP4, no decodable video.
    broken.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 512)

    row = Recording(
        camera_id=None, video_upload_id=None, filename="broken.mp4", file_path=str(broken),
        duration_seconds=1.0, trigger_action="fall", file_size_bytes=broken.stat().st_size,
        event_timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )
    db.add(row)
    db.commit()

    backfill.main(["--apply"])
    db.refresh(row)

    assert row.file_path == str(broken), "the row must still name the file that exists"
    assert Path(row.file_path).exists()
    leftovers = [p.name for p in broken.parent.iterdir() if backfill.TEMP_MARKER in p.name]
    assert leftovers == [], f"a half-written encode was left behind: {leftovers}"


# --- orphans ---------------------------------------------------------------

def test_unreferenced_files_are_reported_but_never_rewritten(backfill, db, tmp_path, caplog):
    """Files no row points at cannot be served, so re-encoding them would
    be pure disk churn - but an operator should still be told."""
    orphan = _write_mp4v(Path(backfill.settings.RECORDINGS_DIR) / "9" / "orphan.mp4")
    before = orphan.read_bytes()

    with caplog.at_level("INFO"):
        backfill.main(["--apply"])

    assert orphan.exists() and orphan.read_bytes() == before, "must not be touched"
    assert "orphan.mp4" in caplog.text, "but must be reported"


def test_the_temporary_name_cannot_resolve_back_to_the_clip_being_read(backfill):
    """`open_writer` derives its base name with `os.path.splitext`, so a
    marker that starts with a dot is stripped as if it were the
    extension - and the "temporary" path resolves back to the source
    clip, which the encode then truncates while still reading from it.

    That is not hypothetical: the first version of this script used
    ".__transcode__" and destroyed the file it was converting."""
    source = "/clips/20260101T000000_fall.mp4"
    temp_base = os.path.join(os.path.dirname(source), Path(source).stem + backfill.TEMP_MARKER)

    # Exactly what open_writer does with the path it is handed.
    resolved = os.path.splitext(temp_base)[0] + ".mp4"

    assert resolved != source, f"the temp path collapses onto the source clip: {resolved}"


# --- deleting unreferenced files ------------------------------------------

def _age(path: Path, seconds: int):
    """Backdates a file so the in-flight guard does not protect it."""
    stamp = os.stat(path).st_mtime - seconds
    os.utime(path, (stamp, stamp))


def test_orphans_are_only_deleted_when_asked(backfill, db, tmp_path):
    orphan = _write_mp4v(Path(backfill.settings.RECORDINGS_DIR) / "9" / "orphan.mp4")
    _age(orphan, 7200)

    backfill.main(["--apply"])

    assert orphan.exists(), "the default must not delete anything"


def test_delete_orphans_removes_unreferenced_unplayable_files(backfill, db, tmp_path):
    orphan = _write_mp4v(Path(backfill.settings.RECORDINGS_DIR) / "9" / "orphan.mp4")
    _age(orphan, 7200)

    backfill.main(["--apply", "--delete-orphans"])

    assert not orphan.exists()


def test_a_referenced_file_is_never_treated_as_an_orphan(backfill, db, unplayable_recording):
    """The whole safety of this rests on 'no row points at it', so a row
    that does point at a file must protect it even while it is being
    converted in the same run."""
    backfill.main(["--apply", "--delete-orphans"])

    db.refresh(unplayable_recording)
    assert Path(unplayable_recording.file_path).exists()


def test_a_recently_written_orphan_is_left_alone(backfill, db, tmp_path, caplog):
    """Clips are written before their row is inserted, so a clip being
    encoded right now is indistinguishable from an orphan. Deleting it
    would destroy an analysis in flight."""
    fresh = _write_mp4v(Path(backfill.settings.RECORDINGS_DIR) / "9" / "in_flight.mp4")

    with caplog.at_level("INFO"):
        backfill.main(["--apply", "--delete-orphans"])

    assert fresh.exists(), "a just-written clip must survive"
    assert "in flight" in caplog.text


def test_playable_orphans_survive_the_default_scope(backfill, db, tmp_path):
    """--delete-orphans reclaims dead weight from the codec bug. A
    playable unreferenced clip is a different judgement - it may be worth
    recovering - so widening to it has to be asked for separately."""
    from app.services.recording_engine import open_writer

    clip_dir = Path(backfill.settings.RECORDINGS_DIR) / "9"
    clip_dir.mkdir(parents=True, exist_ok=True)
    codec, writer, written = open_writer(str(clip_dir / "playable_orphan.mp4"), 64, 48, 10)
    assert writer is not None
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    if inspect_clip(written).needs_transcode:
        pytest.skip("this environment has no browser-playable encoder")
    _age(Path(written), 7200)

    backfill.main(["--apply", "--delete-orphans"])

    assert Path(written).exists()


def test_delete_orphans_requires_apply(backfill):
    """Reporting mode must never delete."""
    with pytest.raises(SystemExit):
        backfill.main(["--delete-orphans"])


def _playable_orphan(backfill, name="playable_orphan.mp4"):
    """An unreferenced clip that decodes fine, or a skip if this build
    cannot produce one."""
    from app.services.recording_engine import open_writer

    clip_dir = Path(backfill.settings.RECORDINGS_DIR) / "9"
    clip_dir.mkdir(parents=True, exist_ok=True)
    codec, writer, written = open_writer(str(clip_dir / name), 64, 48, 10)
    assert writer is not None
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    if inspect_clip(written).needs_transcode:
        pytest.skip("this environment has no browser-playable encoder")
    _age(Path(written), 7200)
    return Path(written)


def test_include_playable_orphans_removes_them_too(backfill, db):
    playable = _playable_orphan(backfill)

    backfill.main(["--apply", "--delete-orphans", "--include-playable-orphans"])

    assert not playable.exists()


def test_include_playable_orphans_reports_without_deleting(backfill, db, caplog):
    """Widening the scan must be answerable without acting on the answer.

    --include-playable-orphans used to require --delete-orphans, which in
    turn requires --apply, so the only way to find out which playable
    clips were unreferenced was to run the command that deleted them. An
    operator auditing disk usage had no safe question to ask, and this
    script's whole contract is that the safe direction is the one you get
    by accident."""
    playable = _playable_orphan(backfill)

    with caplog.at_level("INFO"):
        assert backfill.main(["--include-playable-orphans"]) == 0

    assert playable.exists(), "a report-only run must not delete anything"
    assert playable.name in caplog.text, "but it must say the file is there"
    assert "--delete-orphans" in caplog.text, "and how to act on it"


def test_playable_orphans_stay_out_of_the_default_report(backfill, db, caplog):
    """Widening is opt-in in reporting too: a bare run still speaks only
    about clips that cannot be played."""
    playable = _playable_orphan(backfill)

    with caplog.at_level("INFO"):
        backfill.main([])

    assert playable.exists()
    assert playable.name not in caplog.text


def test_widening_the_scope_still_spares_a_referenced_clip(backfill, db, unplayable_recording):
    """The broadest deletion this script can do must still never touch a
    file a row points at."""
    _playable_orphan(backfill)

    backfill.main(["--apply", "--delete-orphans", "--include-playable-orphans"])

    db.refresh(unplayable_recording)
    assert Path(unplayable_recording.file_path).exists()


def test_files_that_are_not_recognisable_video_are_never_deleted(backfill, db):
    """"Unreferenced" is a claim about the database, not licence to
    delete whatever else lives in the storage tree."""
    root = Path(backfill.settings.RECORDINGS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    keep = root / ".gitkeep"
    keep.write_text("")
    notes = root / "operator-notes.txt"
    notes.write_text("do not delete me")
    for path in (keep, notes):
        _age(path, 7200)

    backfill.main(["--apply", "--delete-orphans", "--include-playable-orphans"])

    assert keep.exists() and notes.exists()


def test_the_wider_scope_never_deletes_on_its_own(backfill, db):
    """Widening the scan is a reporting decision, deleting is a separate
    one. --include-playable-orphans may be combined with --apply (which
    re-encodes rows) without that becoming licence to remove unreferenced
    files: only --delete-orphans does that."""
    playable = _playable_orphan(backfill)

    assert backfill.main(["--apply", "--include-playable-orphans"]) == 0

    assert playable.exists()


def test_the_in_flight_window_can_be_narrowed_deliberately(backfill, db):
    """The default spares anything recent, which is right when the
    operator cannot know what is running - but wrong when they can. It
    has to be lowerable without editing the script."""
    fresh = _write_mp4v(Path(backfill.settings.RECORDINGS_DIR) / "9" / "recent_orphan.mp4")

    backfill.main(["--apply", "--delete-orphans"])
    assert fresh.exists(), "the default must still spare it"

    backfill.main(["--apply", "--delete-orphans", "--orphan-min-age-minutes", "0"])
    assert not fresh.exists()


def test_a_negative_age_window_is_rejected(backfill):
    with pytest.raises(SystemExit):
        backfill.main(["--apply", "--delete-orphans", "--orphan-min-age-minutes", "-1"])
