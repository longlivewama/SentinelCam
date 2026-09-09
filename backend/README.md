# SentinelCam Backend

FastAPI backend for the SentinelCam AI surveillance platform: JWT-authenticated, role-based REST
API, MJPEG live camera streaming, video upload & offline analysis, event-triggered recording,
realtime WebSocket alerts, and YOLOv8-based fall / violence / crowd / abandoned-object detection.
See the root `README.md` for full architecture notes; this file covers backend-specific setup.

## Setup

1. **Create and activate a virtual environment**

   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Create PostgreSQL databases**

   ```bash
   createdb sentinelcam
   createdb sentinelcam_test   # only needed to run the test suite
   # or, from psql:
   # CREATE DATABASE sentinelcam;
   # CREATE USER sentinelcam WITH PASSWORD 'sentinelcam';
   # GRANT ALL PRIVILEGES ON DATABASE sentinelcam TO sentinelcam;
   ```

   If `sentinelcam` is a dedicated (non-superuser) role, also grant it `CREATEDB`:
   `ALTER ROLE sentinelcam CREATEDB;` — one test
   (`tests/test_alembic_migrations.py`) creates and drops a throwaway database to verify
   `alembic upgrade head` actually works against a genuinely fresh database (this caught a real
   bug: the schema previously only worked as an in-place upgrade of an already-existing
   pre-Alembic database, and failed on a brand new one - see that test's docstring). Docker
   Compose's Postgres user already has this by default.

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and fill in at least `DATABASE_URL` and `JWT_SECRET_KEY`. SMTP settings are only
   required for real alert/password-reset emails (`EmailNotifier` fails safe and just logs a
   warning otherwise). See `.env.example` for the full, commented list.

5. **Apply the database schema**

   ```bash
   alembic upgrade head
   ```

   `Base.metadata.create_all()` also still runs automatically on app startup as a dev-convenience
   safety net (it only ever adds missing tables, never alters existing ones) — but for any
   database that already has real data, `alembic upgrade head` is the actual migration path going
   forward. See `alembic/versions/` for the migration history.

6. **Create the first admin user**

   ```bash
   python seed_admin.py
   ```

   Reads `ADMIN_EMAIL` / `ADMIN_PASSWORD` from the environment if set, otherwise prompts
   interactively. Safe to re-run.

7. **Run the API**

   ```bash
   uvicorn app.main:app --reload
   ```

   Served under `http://localhost:8000/api/...`; interactive docs at `http://localhost:8000/docs`.
   CORS origins are configured via `CORS_ORIGINS` (defaults to the Vite dev server).

## Running tests

```bash
createdb sentinelcam_test   # one-time
pytest                       # runs against a real Postgres database, not a mock
ruff check .                 # lint
```

See `tests/conftest.py` for fixtures (role-scoped users/auth headers, table cleanup between
tests, rate-limiter reset between tests).

The suite is destructive — it drops every table at session start and empties them between
tests — so `conftest.py` refuses to run unless the target database is named like a scratch
one (`*_test`, or `test_*`). `DATABASE_URL` takes precedence over the default above when it
is already set, which is how CI points the suite at its own Postgres; it is also how running
`pytest` inside the backend container would otherwise aim those drops at live application
data. Unset `DATABASE_URL` (or point it at `sentinelcam_test`) rather than working around the
refusal; `SENTINELCAM_ALLOW_NON_TEST_DATABASE=1` overrides it if you genuinely mean to wipe
the database you named.

## Backfilling clips written before the codec fix

Clips recorded before `open_writer` gained its codec ladder were written as
MPEG-4 Part 2 (`mp4v`) — valid `.mp4` files that no current browser can decode,
so they render an empty player. The encoder fix is forward-only; existing rows
need rewriting:

```bash
python scripts/backfill_clip_codecs.py            # report what would change
python scripts/backfill_clip_codecs.py --apply
```

It re-encodes through the same ladder new recordings use, verifies the result
decodes before repointing the row, and keeps each superseded original until you
pass `--delete-originals`. Unplayable files that no recording row points at are
reported — they cannot be served, so they are pure dead weight; `--delete-orphans`
removes them, skipping anything written in the last hour, since a clip is on disk
before its row exists and an orphan is indistinguishable from an encode in flight. In Docker, run it as
the uid the app runs as (`--user 10001:10001`) so the rewritten files stay
manageable by the server.

## Roles (RBAC)

`User.role` is one of `admin` / `operator` / `viewer`, enforced via the `require_admin` /
`require_operator` FastAPI dependencies in `app/core/deps.py` on every mutating endpoint — not
just hidden in the frontend. See the root README's architecture section for the exact permission
matrix.

## API surface

- `auth` — signup, login, forgot/reset-password, `/me`, account settings
- `cameras` — CRUD (operator+), MJPEG `/stream`
- `recordings` — list/filter, ranged video streaming, download, delete (operator+)
- `video-uploads` — upload, list/get, ranged source-video streaming, delete
- `alerts` — list/filter, acknowledge (operator+)
- `analytics` — `/summary` aggregate stats for the dashboard
- `admin` — user management (admin only)
- `system` — model/detection/notification status (operator+)
- `ws` — `/ws/events` realtime WebSocket (alerts, camera status, upload progress)

## Notes

- **Recordings/snapshots/uploads** are written to `storage/{recordings,snapshots,uploads}/` under
  `backend/` by default (configurable via `RECORDINGS_DIR` / `SNAPSHOTS_DIR` / `UPLOADS_DIR`).
- **Video codec**: recordings are written with the `avc1` (H.264) fourcc first, falling back to
  `mp4v` if the local OpenCV/ffmpeg build can't open an H.264 encoder; check server logs for which
  codec was used for a given recording.
- **YOLO model weights** (`yolov8n-pose.pt`, `yolov8n.pt`) are downloaded automatically by
  `ultralytics` on first use if not already present locally, and are gitignored.
- **Fall detection** (`app/services/detection/fall_detection.py`) requires a sustained on-ground
  posture (≥1.2s continuous, not a single frame) before firing, specifically to avoid flagging
  quick bends/sit-downs as falls. An optional trained classifier
  (`app/services/detection/fall_classifier.py`, disabled by default via empty
  `FALL_CLASSIFIER_MODEL_PATH`) can corroborate/adjust confidence but never bypasses that gate —
  see `ml/README.md` for how it was trained and its honestly-reported limitations.
- **Violence detection** is an MVP heuristic (pose-keypoint velocity + directional variance), not a
  trained model — see the docstring in `app/services/detection/violence_detection.py` and the
  `VIOLENCE_MODEL_PATH` config hook.
- **Crowd detection** counts YOLO "person" detections rather than using a density-map model.
- **Abandoned object tracking** uses a small hand-rolled centroid tracker rather than
  DeepSORT/ByteTrack.
- **Notifications** go through `app/services/notifications/` (a `Notifier` interface + fan-out
  service) — `email` is implemented; `sms`/`whatsapp` are documented stubs, not real integrations.
- **Rate limiting** on `/auth/login`, `/auth/signup`, `/auth/forgot-password`,
  `/auth/reset-password` is a simple in-memory per-process sliding window
  (`app/core/rate_limit.py`) — fine for a single backend process; needs a shared store (Redis) if
  ever run as multiple worker processes.
