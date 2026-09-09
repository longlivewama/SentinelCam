#!/usr/bin/env python
"""
Resets the database to a known state and seeds fixtures for the Playwright
e2e suite (e2e/): wipes all tables, then creates one admin user, one
camera, and one recording+event pair so the Recordings/Alerts pages have
something deterministic to assert against without depending on real
detection timing.

Never run this against a database with real data you care about - it
truncates everything.

There is no separate e2e database to fall back on: this runs inside the
backend container (see e2e/global-setup.js), where DATABASE_URL addresses
the compose stack's own `sentinelcam` database. Resetting that is the
intent when the e2e suite calls it, and data loss when anyone else does -
and the two invocations are otherwise identical. So the wipe has to be
asked for explicitly, with `--wipe-database`; without it this script
refuses and reports what it would have destroyed.
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from sqlalchemy import func, inspect, select  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402
from app.models.camera import Camera  # noqa: E402
from app.models.event import Event  # noqa: E402
from app.models.password_reset_token import PasswordResetToken  # noqa: E402
from app.models.recording import Recording  # noqa: E402
from app.models.user import ROLE_ADMIN, User  # noqa: E402
from app.models.video_upload import VideoUpload  # noqa: E402
from app.services.recording_engine import open_writer  # noqa: E402

E2E_ADMIN_EMAIL = os.environ.get("E2E_ADMIN_EMAIL", "e2e-admin@example.com")
E2E_ADMIN_PASSWORD = os.environ.get("E2E_ADMIN_PASSWORD", "e2e-admin-password123")

_TABLES_IN_DELETE_ORDER = [
    PasswordResetToken.__table__,
    Event.__table__,
    Recording.__table__,
    VideoUpload.__table__,
    Camera.__table__,
    User.__table__,
]


CONFIRM_FLAG = "--wipe-database"


def _database_description() -> str:
    """Which database is about to be reset, named without its credentials
    - enough for an operator to recognise it, nothing worth redacting."""
    url = engine.url
    return f"{url.database!r} on {url.host or 'localhost'}:{url.port or 5432}"


def _existing_row_counts() -> dict:
    """Rows currently in the tables this script truncates.

    Read-only, and tolerant of tables that do not exist yet: on a fresh
    database `create_all` has not run, which is the ordinary first-run
    case rather than an error."""
    present = set(inspect(engine).get_table_names())
    counts = {}
    with engine.connect() as conn:
        for table in _TABLES_IN_DELETE_ORDER:
            if table.name in present:
                counts[table.name] = conn.execute(select(func.count()).select_from(table)).scalar_one()
    return counts


def _assert_wipe_was_requested(argv) -> None:
    """Refuses to truncate anything unless the caller said so on the
    command line.

    A name-based rule (the one guarding tests/conftest.py) cannot work
    here: the database this legitimately targets is the application's own
    `sentinelcam`, so any check that let the e2e suite through would let
    an accidental run through too. What separates the two is intent, and
    intent has to be stated rather than inferred."""
    if settings.ENVIRONMENT == "production":
        # No flag overrides this. A fixture seeder has no business in
        # production at all, so treat the combination as a mistake.
        print(
            f"Refusing to seed e2e fixtures: ENVIRONMENT is 'production' and this script "
            f"truncates every table in database {_database_description()}.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if CONFIRM_FLAG in argv:
        return

    occupied = {name: count for name, count in _existing_row_counts().items() if count}
    if occupied:
        inventory = "\n".join(f"    {count:>6}  {name}" for name, count in occupied.items())
        holdings = f"It currently holds:\n\n{inventory}\n"
    else:
        holdings = "It is currently empty, but that is checked at this moment only.\n"

    print(
        f"Refusing to reset database {_database_description()} without {CONFIRM_FLAG}.\n"
        f"\n"
        f"This script DELETEs every row from users, cameras, recordings, events,\n"
        f"video_uploads and password_reset_tokens before seeding its fixtures.\n"
        f"\n"
        f"{holdings}"
        f"\n"
        f"The e2e suite passes {CONFIRM_FLAG} for you (e2e/global-setup.js); running\n"
        f"`npm test` in e2e/ is the supported path. Pass the flag by hand only if you\n"
        f"mean to destroy the data above.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _write_tiny_video(path: Path) -> Path:
    """Writes the seeded clip through the same codec ladder real
    recordings use, and returns the path actually written.

    This used to hardcode `mp4v`, so the one recording the e2e suite
    asserts a video player on was itself a clip no browser could decode -
    the exact bug the ladder exists to prevent, baked into the fixture
    data. The container follows the codec, so the caller has to use the
    returned path rather than the one it asked for."""
    path.parent.mkdir(parents=True, exist_ok=True)

    codec, writer, written = open_writer(str(path), 320, 240, 10)
    if writer is None:
        raise RuntimeError(f"could not open a VideoWriter for {path} with any codec")

    for i in range(30):
        frame = np.full((240, 320, 3), fill_value=(i * 5) % 200 + 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    return Path(written)


def main(argv=None):
    _assert_wipe_was_requested(sys.argv[1:] if argv is None else argv)

    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        for table in _TABLES_IN_DELETE_ORDER:
            conn.execute(table.delete())

    with SessionLocal() as db:
        admin = User(
            email=E2E_ADMIN_EMAIL,
            password_hash=hash_password(E2E_ADMIN_PASSWORD),
            full_name="E2E Admin",
            role=ROLE_ADMIN,
            is_active=True,
        )
        db.add(admin)

        camera = Camera(
            name="E2E Test Camera",
            url="0",
            camera_type="usb",
            location="Test Lab",
            status="active",
            is_active=True,
            ai_detection_enabled=False,
        )
        db.add(camera)
        db.commit()
        db.refresh(camera)

        clip_path = _write_tiny_video(Path(settings.RECORDINGS_DIR) / str(camera.id) / "e2e_seed_fall.mp4")

        event_timestamp = datetime.now(timezone.utc)
        recording = Recording(
            camera_id=camera.id,
            filename=clip_path.name,
            file_path=str(clip_path),
            duration_seconds=3.0,
            trigger_action="fall",
            file_size_bytes=clip_path.stat().st_size,
            event_timestamp=event_timestamp,
        )
        db.add(recording)
        db.commit()
        db.refresh(recording)

        event = Event(
            camera_id=camera.id,
            recording_id=recording.id,
            event_type="fall",
            confidence_score=0.82,
            timestamp=event_timestamp,
            triggered_recording=True,
            acknowledged=False,
        )
        db.add(event)
        db.commit()

        print(f"Seeded e2e fixtures: admin={E2E_ADMIN_EMAIL}, camera_id={camera.id}, recording_id={recording.id}, event_id={event.id}")


if __name__ == "__main__":
    main()
