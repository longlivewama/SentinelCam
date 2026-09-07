#!/usr/bin/env python
"""
Resets the database to a known state and seeds fixtures for the Playwright
e2e suite (e2e/): wipes all tables, then creates one admin user, one
camera, and one recording+event pair so the Recordings/Alerts pages have
something deterministic to assert against without depending on real
detection timing.

Never run this against a database with real data you care about - it
truncates everything. Intended to be run only against the dedicated
`sentinelcam_e2e` database via `docker compose exec backend python
scripts/seed_e2e_fixtures.py` (see e2e/README.md).
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402
from app.models.camera import Camera  # noqa: E402
from app.models.event import Event  # noqa: E402
from app.models.password_reset_token import PasswordResetToken  # noqa: E402
from app.models.recording import Recording  # noqa: E402
from app.models.user import ROLE_ADMIN, User  # noqa: E402
from app.models.video_upload import VideoUpload  # noqa: E402

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


def _write_tiny_video(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10, (320, 240))
    for i in range(30):
        frame = np.full((240, 320, 3), fill_value=(i * 5) % 200 + 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def main():
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

        clip_path = Path(settings.RECORDINGS_DIR) / str(camera.id) / "e2e_seed_fall.mp4"
        _write_tiny_video(clip_path)

        event_timestamp = datetime.now(timezone.utc)
        recording = Recording(
            camera_id=camera.id,
            filename="e2e_seed_fall.mp4",
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
