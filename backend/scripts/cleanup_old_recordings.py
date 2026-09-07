#!/usr/bin/env python
"""
Deletes recordings (DB row + file on disk) older than a retention window.
Intended to be run periodically (e.g. a daily cron job / scheduled task) -
this project doesn't run a background scheduler itself, so wiring this up
to actually run on a schedule is a deployment-time decision, not something
the app does automatically.

Usage:
    python scripts/cleanup_old_recordings.py --days 90
    python scripts/cleanup_old_recordings.py --days 90 --dry-run
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal  # noqa: E402
from app.models.event import Event  # noqa: E402
from app.models.recording import Recording  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, required=True, help="Delete recordings older than this many days")
    parser.add_argument("--dry-run", action="store_true", help="List what would be deleted without deleting")
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be at least 1")

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    with SessionLocal() as db:
        old_recordings = db.query(Recording).filter(Recording.event_timestamp < cutoff).all()

        if not old_recordings:
            logger.info("No recordings older than %d days found.", args.days)
            return

        logger.info("Found %d recording(s) older than %d days (cutoff: %s).", len(old_recordings), args.days, cutoff.isoformat())

        for recording in old_recordings:
            if args.dry_run:
                logger.info("[dry-run] Would delete recording %d: %s", recording.id, recording.file_path)
                continue

            file_path = recording.file_path

            # The alert history (Event rows) is kept even after its video
            # clip is pruned - only the recording_id link is cleared, so
            # deleting the Recording doesn't hit the FK constraint.
            db.query(Event).filter(Event.recording_id == recording.id).update({"recording_id": None})
            db.delete(recording)
            db.commit()

            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    logger.exception("Failed to remove file %s for deleted recording %d", file_path, recording.id)

            logger.info("Deleted recording %d: %s", recording.id, file_path)

    if args.dry_run:
        logger.info("Dry run complete - no changes made.")
    else:
        logger.info("Cleanup complete.")


if __name__ == "__main__":
    main()
