#!/usr/bin/env python
"""
Re-encodes stored clips that no browser can play.

Clips written before the codec-ladder fix went to disk as MPEG-4 Part 2
("mp4v"): valid `.mp4` files, served correctly with 206 responses, and
undecodable in Chrome, Firefox and Edge. `open_writer` now picks a
browser-playable codec for new clips, but that is forward-only - rows
already pointing at an mp4v file keep rendering an empty black player.
This walks those rows and rewrites the files in place.

There is no ffmpeg in the runtime image, so the re-encode goes through
OpenCV: decode each frame from the old clip, write it out through the
same `open_writer` ladder new recordings use. Frames are streamed one at
a time rather than collected - a 30-second 576x1024 clip is over 300 MB
of raw frames, which is not something a maintenance script should hold in
memory.

Re-encoding is lossy and these clips are silent (cv2.VideoWriter never
wrote an audio track), so nothing but a generation of video quality is
lost. There is no lossless alternative: mp4v is the codec, not just the
container, so remuxing cannot help.

Unlike cleanup_old_recordings.py, this reports by default and changes
nothing without --apply. It rewrites files *and* database rows, and it is
a one-shot migration rather than a routine job, so the safe direction is
the one you get by accident.

Usage:
    python scripts/backfill_clip_codecs.py                       # report only
    python scripts/backfill_clip_codecs.py --apply
    python scripts/backfill_clip_codecs.py --apply --delete-originals

In Docker, run it as the uid the app runs as, or the rewritten files land
root-owned and the server cannot manage them afterwards:

    docker compose exec --user 10001:10001 backend \
        python scripts/backfill_clip_codecs.py --apply
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.video_codec import inspect_clip  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.models.recording import Recording  # noqa: E402
from app.services.recording_engine import open_writer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Infix for the half-written file, so a crash mid-encode never leaves
# something that looks like a real clip next to the real ones.
#
# Deliberately NOT starting with a dot: `open_writer` derives its base
# name with `os.path.splitext`, which would read a leading-dot marker as
# the extension, strip it, and hand back the source file's own path - so
# the encode would open the clip it is reading from and truncate it.
TEMP_MARKER = "__transcode__"
# What a replaced original is renamed to when the new file would occupy
# the same path (an environment whose ladder reaches avc1). Keeps the
# source intact until the row has been updated.
BACKUP_SUFFIX = ".pre-transcode"

DEFAULT_FPS = 10.0

# An unreferenced file is not necessarily an abandoned one: clips are
# written to disk before their row is inserted (see
# video_analysis._write_upload_clip), so a clip being encoded right now
# looks exactly like an orphan. Anything touched recently is left alone
# rather than deleted out from under an analysis in flight.
ORPHAN_MIN_AGE_SECONDS = 3600
ORPHAN_MIN_AGE_MINUTES = ORPHAN_MIN_AGE_SECONDS // 60


class TranscodeError(RuntimeError):
    pass


def _source_fps(capture, recording) -> float:
    """FPS to re-encode at. The container's own value is authoritative
    where it is sane; some mp4v files report 0, in which case the row's
    duration is a better guess than a hardcoded default."""
    reported = capture.get(cv2.CAP_PROP_FPS)
    if reported and reported > 0 and reported < 1000:
        return float(reported)

    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = recording.duration_seconds or 0
    if frames and frames > 0 and duration > 0:
        return float(frames) / float(duration)

    logger.warning("    %s reports no usable frame rate; assuming %.0f fps", recording.filename, DEFAULT_FPS)
    return DEFAULT_FPS


def _transcode(source: Path, recording) -> Path:
    """Decodes `source` and writes a browser-playable copy beside it.

    Returns the path actually written. Raises TranscodeError - having
    cleaned up after itself - if the clip cannot be read, if no playable
    codec is available, or if the result does not decode."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise TranscodeError("could not open the existing clip for decoding")

    try:
        ok, frame = capture.read()
        if not ok:
            raise TranscodeError("the existing clip decoded zero frames; it may be truncated")

        height, width = frame.shape[:2]
        fps = _source_fps(capture, recording)

        codec, writer, written = open_writer(str(source.parent / (source.stem + TEMP_MARKER)), width, height, round(fps))
        if writer is None:
            raise TranscodeError("no codec in the ladder could be opened in this environment")

        written_path = Path(written)
        if codec == "mp4v":
            # Re-encoding mp4v into mp4v would burn a generation of
            # quality to produce a file just as unplayable.
            writer.release()
            written_path.unlink(missing_ok=True)
            raise TranscodeError(
                "this build offers no browser-playable encoder (only mp4v); "
                "install an FFmpeg with libx264 or VP8/VP9 support and re-run"
            )

        try:
            count = 0
            while ok:
                writer.write(frame)
                count += 1
                ok, frame = capture.read()
        finally:
            writer.release()
    finally:
        capture.release()

    verified = _verify(written_path, width, height)
    if verified == 0:
        written_path.unlink(missing_ok=True)
        raise TranscodeError("the re-encoded clip decoded zero frames; discarded it")

    logger.info("    re-encoded %d frames as %s (%d decoded back)", count, codec, verified)
    return written_path


def _verify(path: Path, width: int, height: int) -> int:
    """Frames that can be decoded back out of `path`, 0 if its dimensions
    do not match the source. A file that opens but decodes to nothing
    would otherwise pass every other check and still be useless."""
    capture = cv2.VideoCapture(str(path))
    try:
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                return 0
            decoded += 1
        return decoded
    finally:
        capture.release()


def _install(written: Path, source: Path):
    """Moves the finished encode into place.

    Returns `(final_path, superseded_path)`. Where the new file would
    occupy the source's own path - an environment whose ladder reaches
    avc1, so `.mp4` in and `.mp4` out - the original is renamed aside
    first rather than overwritten, so nothing is destroyed before the
    database has been updated."""
    final = source.with_suffix(written.suffix)

    if final == source:
        backup = source.with_suffix(source.suffix + BACKUP_SUFFIX)
        os.replace(source, backup)
        os.replace(written, final)
        return final, backup

    os.replace(written, final)
    return final, source


def _orphaned_clips(known_paths: set, include_playable: bool = False) -> list:
    """Clip files under the recordings directory that no row points at.

    Never re-encoded: without a row they cannot be served, so converting
    them would be disk churn - but an operator should know they are
    there.

    Only files this module recognises as video are eligible. A file whose
    codec could not be determined is left out entirely rather than
    guessed at: "unreferenced" is a claim about the database, and it must
    not become licence to delete something that is not even a clip."""
    root = Path(settings.RECORDINGS_DIR)
    if not root.is_dir():
        return []

    orphans = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or str(path) in known_paths:
            continue
        playable = inspect_clip(str(path)).playable
        if playable is False or (include_playable and playable is True):
            orphans.append(path)
    return orphans


def _delete_orphans(orphans: list, min_age_seconds: int = ORPHAN_MIN_AGE_SECONDS) -> tuple:
    """Removes unreferenced clips, skipping any written within
    `min_age_seconds`. Returns `(deleted_count, freed_bytes, skipped)`."""
    now = time.time()
    deleted = freed = 0
    skipped = []

    for path in orphans:
        try:
            stat = path.stat()
            if now - stat.st_mtime < min_age_seconds:
                skipped.append(path)
                continue
            path.unlink()
        except OSError as exc:
            logger.error("    could not remove %s: %s", path, exc)
            continue
        deleted += 1
        freed += stat.st_size

    return deleted, freed, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description="Re-encode stored clips that no browser can play.")
    parser.add_argument("--apply", action="store_true", help="Actually re-encode; without this, only reports")
    parser.add_argument(
        "--delete-orphans",
        action="store_true",
        help=(
            "Also remove unplayable clip files that no recording row points at. "
            "They cannot be served, so this only reclaims disk - but it is irreversible"
        ),
    )
    parser.add_argument(
        "--orphan-min-age-minutes",
        type=int,
        default=ORPHAN_MIN_AGE_MINUTES,
        help=(
            "How recently written an unreferenced file may be and still be spared "
            f"(default {ORPHAN_MIN_AGE_MINUTES}). Lower it only once you have confirmed no "
            "analysis is running: a clip reaches disk before its row exists"
        ),
    )
    parser.add_argument(
        "--include-playable-orphans",
        action="store_true",
        help=(
            "Widen --delete-orphans to unreferenced clips that still play. They are equally "
            "unreachable, but unlike mp4v dead weight they may be worth recovering first"
        ),
    )
    parser.add_argument(
        "--delete-originals",
        action="store_true",
        help="Remove each superseded file once its row points at the new one (implies --apply)",
    )
    args = parser.parse_args(argv)

    if args.delete_originals and not args.apply:
        parser.error("--delete-originals only makes sense with --apply")
    if args.delete_orphans and not args.apply:
        parser.error("--delete-orphans only makes sense with --apply")
    if args.include_playable_orphans and not args.delete_orphans:
        parser.error("--include-playable-orphans only makes sense with --delete-orphans")
    if args.orphan_min_age_minutes < 0:
        parser.error("--orphan-min-age-minutes cannot be negative")

    converted = failed = skipped = 0
    reclaimable = 0

    with SessionLocal() as db:
        recordings = db.query(Recording).order_by(Recording.id).all()
        known_paths = {rec.file_path for rec in recordings}

        for recording in recordings:
            source = Path(recording.file_path)

            if not source.exists():
                logger.warning("recording %s: file is missing (%s)", recording.id, source)
                skipped += 1
                continue

            codec = inspect_clip(str(source))
            if not codec.needs_transcode:
                skipped += 1
                continue

            logger.info("recording %s: %s -> %s", recording.id, recording.filename, codec.detail)

            if not args.apply:
                converted += 1
                continue

            try:
                written = _transcode(source, recording)
            except TranscodeError as exc:
                logger.error("    FAILED: %s", exc)
                failed += 1
                continue

            final, superseded = _install(written, source)

            recording.filename = final.name
            recording.file_path = str(final)
            recording.file_size_bytes = final.stat().st_size
            db.commit()

            if args.delete_originals:
                superseded.unlink(missing_ok=True)
            else:
                reclaimable += superseded.stat().st_size

            logger.info("    now serving %s", final.name)
            converted += 1

        orphans = _orphaned_clips(known_paths, include_playable=args.include_playable_orphans)

    if args.apply:
        logger.info("\nRe-encoded %d clip(s), %d failed, %d already playable or unreadable.", converted, failed, skipped)
        if reclaimable:
            logger.info(
                "Kept %s of superseded originals; re-run with --delete-originals to remove them.",
                _human(reclaimable),
            )
    else:
        logger.info("\n%d clip(s) would be re-encoded, %d already playable or unreadable.", converted, skipped)
        if converted:
            logger.info("Nothing has been changed. Re-run with --apply to do it.")

    if orphans:
        total = sum(path.stat().st_size for path in orphans)

        if args.delete_orphans:
            deleted, freed, skipped = _delete_orphans(orphans, args.orphan_min_age_minutes * 60)
            logger.info("\nRemoved %d unreferenced file(s), freeing %s.", deleted, _human(freed))
            for path in skipped:
                logger.info(
                    "    kept %s - written less than %d minutes ago, so it may belong to an "
                    "analysis still in flight",
                    path,
                    args.orphan_min_age_minutes,
                )
        else:
            logger.info(
                "\nAlso found %d file(s) (%s) that no recording row points at, "
                "so they cannot be served and were left alone:",
                len(orphans),
                _human(total),
            )
            for path in orphans:
                logger.info("    %s", path)
            logger.info("Pass --delete-orphans to remove them.")

    return 1 if failed else 0


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024


if __name__ == "__main__":
    sys.exit(main())
