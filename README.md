# SentinelCam

A full-stack AI surveillance platform: live multi-camera monitoring, offline video upload &
analysis, automatic AI detection of falls (plus violence/abandoned-object/crowd heuristics),
role-based access control, realtime alerts over WebSocket, and a dashboard/analytics UI — each
detection triggering a recorded clip and an email alert.

- `backend/` — FastAPI + PostgreSQL + OpenCV + YOLOv8 (see `backend/README.md`)
- `frontend/` — React + Vite + Tailwind + Zustand (see `frontend/README.md`)
- `ml/` — standalone ML training pipeline for the corroborating fall classifier (see `ml/README.md`) — **not** imported by the running app; it only produces a model artifact the backend can optionally load

## Architecture

```
┌──────────────┐      REST + WebSocket       ┌───────────────────┐
│   Frontend   │ ◀─────────────────────────▶ │      Backend       │
│ React/Vite   │                             │  FastAPI (Python)  │
└──────────────┘                             └─────────┬──────────┘
                                                         │
                       ┌─────────────────────────────────┼─────────────────────────────────┐
                       │                                 │                                 │
                 ┌─────▼─────┐                    ┌──────▼──────┐                   ┌──────▼──────┐
                 │ PostgreSQL │                    │  Detection   │                   │ Notification │
                 │  (SQLAlchemy│                   │   engine     │                   │   service    │
                 │  + Alembic) │                   │ (YOLOv8-Pose │                   │ (email today,│
                 └────────────┘                    │ + object det)│                   │ SMS/WhatsApp │
                                                     └──────┬──────┘                   │ stubs ready) │
                                                            │                          └─────────────┘
                                              ┌─────────────┴─────────────┐
                                        ┌──────▼──────┐            ┌──────▼───────┐
                                        │Live cameras │            │Uploaded video │
                                        │(stream_mgr, │            │  (video_      │
                                        │ per-camera  │            │  analysis.py) │
                                        │ thread)     │            │               │
                                        └─────────────┘            └───────────────┘
```

Both the live-camera path and the video-upload path share the exact same pose model,
`FallDetector`, and `recording_engine`/clip-writing code — only the frame source and timeline
differ. See "Where this deviates from a textbook implementation" below for the detector design
rationale.

**Auth.** JWT (24h expiry), bcrypt-hashed passwords, forgot/reset-password via one-time
SHA-256-hashed tokens emailed to the user (raw token never persisted). Every protected backend
route accepts the token either as an `Authorization: Bearer` header (used by all JSON API calls
from the frontend's Axios client) or as a `?token=` query parameter — needed because browsers
cannot attach custom headers to `<img src>`, `<video src>`, `<a href>` downloads, or the native
WebSocket API, so the MJPEG stream, video playback, download endpoints, and the realtime
WebSocket are linked to directly with the token in the URL.

**Authorization.** Three roles: `admin` (full access incl. user management), `operator` (manage
cameras, acknowledge alerts, upload/analyze video), `viewer` (read-only, plus upload/analysis —
that's a customer-facing analysis action, not infrastructure control). Enforced on the backend via
FastAPI dependencies (`require_admin`, `require_operator`) on every mutating endpoint — the
frontend's route guards and hidden buttons are a UX convenience, not the security boundary.

**Streaming.** Each camera gets one lazily-started background thread that owns the
`cv2.VideoCapture` connection, decodes frames, and holds the latest JPEG in a shared buffer. The
`/stream` endpoint serves that shared buffer as `multipart/x-mixed-replace` to as many simultaneous
viewers as connect, without opening a second capture connection per viewer.

**Recording.** The same capture thread keeps a rolling buffer of the last ~3 seconds of raw
frames. When a detector fires an event, the recording engine snapshots that buffer, appends ~3
more seconds of live frames, and writes the combined clip to disk (`avc1`/H.264 first, falling
back to `mp4v`), records it in the database, and sends an email alert with a snapshot attached.

**Video upload & analysis.** An uploaded video is decoded frame-by-frame (single pass), run
through the same pose model + `FallDetector` used for live cameras (with the sustained-duration
gate driven by the video's own timeline instead of wall-clock time), and any fall event gets its
own pre/post-event clip written out exactly like a live-camera recording — with `video_upload_id`
set instead of `camera_id`. Progress is reported over the realtime WebSocket as the file is
processed.

**Realtime.** A `RealtimeBroadcaster` (in-process pub/sub over one WebSocket endpoint,
`/api/ws/events`) lets background worker threads — the per-camera detection loop, the recording
engine, the video-analysis worker — push `alert.created`, `camera.status`, `upload.progress`, and
`upload.completed` events to every connected browser tab instantly, without polling. The frontend
uses this to update the dashboard, alerts badge, and upload progress bars live.

**Notifications.** `app/services/notifications/` defines a `Notifier` interface with one
implementation today (`EmailNotifier`, stdlib `smtplib`) fanned out by `NotificationService`, and
two architectural stubs (`SmsNotifier`, `WhatsAppNotifier`) that log a clear "not configured"
warning rather than silently no-op — wiring in a real provider later is a config change plus one
new class, not a call-site rewrite.

## Where this deviates from a "textbook" implementation, and why

Three of the four live-camera detectors assume production-grade models trained on specific
research datasets (RWF-2000 for violence, ShanghaiTech via CSRNet for crowd density, DeepSORT for
tracking). Training those from scratch needs the actual datasets plus substantial GPU time, so
each was replaced with a practical, honestly-scoped equivalent that fits the same interface and
can be swapped for a trained model later without touching the rest of the system:

| Detector | Spec called for | What's actually implemented | Upgrade path |
|---|---|---|---|
| Fall | YOLOv8-Pose, 3-signal heuristic | Pose heuristic (aspect ratio / keypoint alignment / hip velocity) gated by a **sustained on-ground-posture requirement** (≥1.2s continuous, not a single frame) to cut false positives from bending/sitting, **plus** an optional trained MLP classifier (`ml/`) as a corroborating confidence signal | Retrain `ml/` on real (non-synthetic) footage for a stronger corroborating signal; see `ml/README.md`'s honest evaluation caveats before trusting it further than that |
| Violence | CNN+LSTM trained on RWF-2000 | Heuristic reusing the same pose keypoints: erratic high-velocity wrist/elbow motion between people in close proximity | Swap in a trained temporal model via `VIOLENCE_MODEL_PATH` |
| Crowd counting | CSRNet density-map estimation | YOLOv8 person-class detection + count, smoothed over frames, thresholded per camera | Swap in CSRNet for very dense/occluded scenes |
| Abandoned object | YOLO + DeepSORT | YOLO object detection + a small hand-rolled centroid tracker | Swap in DeepSORT/ByteTrack for robust ID persistence through occlusion |

## Local setup

### Prerequisites

- Python 3.11+, Node 20+, a local PostgreSQL server.

### 1. Database

```bash
createdb sentinelcam
createdb sentinelcam_test   # only needed to run the backend test suite
```

### 2. Backend

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                     # fill in DATABASE_URL, JWT_SECRET_KEY, SMTP_* etc.
alembic upgrade head                     # apply the schema (see "Database" below)
python seed_admin.py                     # creates the first admin user
uvicorn app.main:app --reload            # http://localhost:8000  (docs at /docs)
```

### 3. Frontend (separate terminal)

```bash
cd frontend
npm install
cp .env.example .env                     # VITE_API_URL=http://localhost:8000
npm run dev                              # http://localhost:5173
```

Sign up (or log in with the admin account you created), then add a camera (an RTSP URL for an IP
camera, or a device index like `0` for a USB webcam) from the Cameras page, or go straight to
Upload to analyze a video file without any camera hardware.

## Environment variables

See `backend/.env.example` and `frontend/.env.example` for the full, commented list. Highlights:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `JWT_SECRET_KEY` | JWT signing secret — **the app refuses to start** if this is left at its insecure default while `ENVIRONMENT=production` |
| `CORS_ORIGINS` | Comma-separated list of allowed frontend origins |
| `FRONTEND_URL` | Used to build links in emails (e.g. the password-reset link) |
| `NOTIFICATION_CHANNELS` | Comma-separated enabled channels (`email` works out of the box) |
| `SMTP_*` / `ALERT_RECIPIENTS` | Email delivery config |
| `UPLOADS_DIR` / `MAX_UPLOAD_SIZE_MB` / `ALLOWED_VIDEO_EXTENSIONS` | Video upload config |
| `FALL_CLASSIFIER_MODEL_PATH` | Optional path to the `ml/`-trained classifier's exported ONNX file (empty = heuristic-only, the default) |
| `VITE_API_URL` (frontend) | Backend base URL; the realtime WebSocket URL is derived from this automatically |

Never hardcode credentials — everything above is read from the environment / `.env` (gitignored).

## Database

Schema changes are managed with Alembic (`backend/alembic/`), targeting whatever `DATABASE_URL`
resolves to. `alembic upgrade head` is the production migration path.

```bash
cd backend
alembic upgrade head                                    # apply all pending migrations
alembic revision --autogenerate -m "describe the change" # after changing a model
```

For local/test convenience, `Base.metadata.create_all()` also still runs once on app startup —
it only ever *adds missing tables*, it never alters existing ones, so it's a safety net for a
fresh dev database, not a substitute for running migrations against a database that already has
data (see the RBAC migration, `alembic/versions/dc14a9d0..._add_rbac_roles...py`, for an example of
a real data-preserving migration: it backfills `role='admin'` for existing `is_admin=true` users
before dropping that column).

## ML training & model export

The corroborating fall classifier is trained completely separately from the running backend —
see **`ml/README.md`** for the full pipeline (dataset sourcing + licenses, train/val/test split,
training run, held-out evaluation with an honest discussion of what the metrics do and don't
prove, and ONNX export). Quick version:

```bash
cd backend && source venv/bin/activate      # reuses the backend venv (torch/numpy/ultralytics already there)
pip install -r ../ml/requirements.txt
cd ../ml
python3 scripts/download_simuletic_labels.py && python3 scripts/download_coco_pose_labels.py
python3 scripts/prepare_dataset.py
python3 train.py
python3 evaluate.py           # writes reports/eval_report.md
python3 export.py             # writes exported/fall_classifier_v1.onnx + metadata
```

To use the exported model in the backend, add `onnxruntime` (already in `backend/requirements.txt`)
and set `FALL_CLASSIFIER_MODEL_PATH=../ml/exported/fall_classifier_v1.onnx` in `backend/.env`. It's
used only as a confidence adjustment on top of the heuristic's sustained-duration gate, never to
bypass it — read `ml/README.md`'s "Known limitations" before enabling this in anything resembling
production; the headline metrics are flagged there as likely inflated by domain-shortcut learning
on a small, narrow test set.

## Running tests

### Backend

```bash
cd backend
createdb sentinelcam_test   # one-time
source venv/bin/activate
pytest                       # 61 tests: unit (security, fall-detection logic, classifier),
                              # API (auth/RBAC, cameras, alerts, recordings, video uploads,
                              # analytics, system status, rate limiting), one real end-to-end
                              # video-analysis run through actual YOLO inference, WebSocket auth
ruff check .                 # lint
```

Tests run against a **real PostgreSQL** database (`sentinelcam_test`), not a mock — the app relies
on Postgres-specific behavior (boolean filters, timezone-aware timestamps) in enough places that a
sqlite substitute would give false confidence.

### Frontend

```bash
cd frontend
npm run lint
npm run test     # 39 tests: auth pages (login/signup/forgot/reset), route guards, upload
                  # workflow (drag-drop, progress, delete), dashboard, stores
npm run build
```

### Browser QA

Both the live-camera pipeline (real webcam, real MJPEG stream, enabling AI detection) and the full
upload → process → view-results workflow were exercised end-to-end in a real browser during
development (Cameras CRUD, live stream, Alerts acknowledge, Recordings playback, Analytics charts
with hover tooltips, System Status, Users/RBAC nav visibility per role, responsive mobile layout,
signup → auto-login, forgot/reset-password including invalid-token handling). Not automated as
part of CI — see "Remaining limitations" for what a proper Playwright/Cypress e2e suite would add.

## Running in production

1. Set `ENVIRONMENT=production` and a real random `JWT_SECRET_KEY` (the app refuses to boot
   otherwise).
2. Run `alembic upgrade head` against the production database as part of your deploy step, not
   `create_all`.
3. Run the backend behind a real ASGI server config (e.g. `uvicorn app.main:app --workers N`
   behind a reverse proxy) — note the in-memory rate limiter and realtime broadcaster are
   per-process; scaling to multiple worker processes needs a shared store (Redis) for both. See
   the docstrings in `app/core/rate_limit.py` and `app/services/realtime.py`.
4. Serve the frontend's `npm run build` output (`frontend/dist/`) from a static host/CDN, pointed
   at the backend via `VITE_API_URL` baked in at build time.
5. Configure real SMTP credentials for `NOTIFICATION_CHANNELS=email` to work, and set
   `CORS_ORIGINS` to your real frontend origin(s).

## Troubleshooting

- **Backend won't start: "JWT_SECRET_KEY is still set to its insecure default"** — you have
  `ENVIRONMENT=production` set with the placeholder secret. Generate one:
  `python -c "import secrets; print(secrets.token_hex(32))"`.
- **`alembic upgrade head` fails on a fresh database** — make sure `DATABASE_URL` in `.env` points
  at a database that exists (`createdb sentinelcam` first) and that the user has privileges to
  create tables.
- **Live camera stream never loads** — check the camera's `url` (numeric string for a USB device
  index, full `rtsp://...` for an IP camera) and check backend logs for `cv2.VideoCapture` open
  failures; the camera tile falls back to "Stream unavailable" without crashing the page.
- **Video recordings won't play in Safari** — check the backend log line for which the codec was
  used (`avc1` vs `mp4v` fallback); `mp4v` is far less broadly compatible.
- **Uploaded video stuck at "processing"** — check backend logs; `video_analysis.py` catches and
  records exceptions as `status=failed` with an `error_message`, so "stuck" (rather than "failed")
  usually means a very large/long file still genuinely processing on CPU.
- **Realtime "Reconnecting" never goes "Live"** — the WebSocket auth token comes from the same
  JWT as everything else; check it hasn't expired, and check the backend log for the
  `/api/ws/events` connection attempt.
- **`pytest` fails with a connection error** — `createdb sentinelcam_test` first; the suite expects
  a real reachable Postgres at the `DATABASE_URL` its `conftest.py` defaults to (override via env
  var if your local Postgres runs on a non-default port, as this project's own dev setup does).

## Remaining limitations (see also `ml/README.md` and the security notes below)

- No automated end-to-end browser test suite (Playwright/Cypress) — the flows were verified
  manually in a real browser during development but aren't wired into CI. Adding one is the
  highest-value next testing investment.
- The fall classifier's headline metrics are on a small, narrow, largely-synthetic dataset and are
  explicitly flagged as likely optimistic — treat it as an experimental corroborating signal, not
  a validated production model, until retrained on real footage.
- Violence/crowd/abandoned-object detectors remain heuristics, not trained models (see table
  above) — this was true of the original prototype and remains a known scope boundary.
- The in-memory rate limiter and realtime broadcaster assume a single backend process; a
  multi-worker/multi-instance production deployment needs a shared backing store (Redis) for both.
- Two residual, low-risk dependency vulnerabilities are documented rather than force-fixed: `ecdsa`
  (only reachable via ECDSA JWT algorithms, and this app only uses HS256) and `pyasn1` (pinned to
  an old version by `python-jose`'s own declared constraint; upgrading it breaks that package).
  `react-router-dom` and the Vite/Vitest dev-tooling chain also have known advisories fixed only in
  major-version bumps (v7 and v8 respectively) that weren't attempted this late without time to
  fully re-test routing/tooling behavior — see `npm audit` for details.

## Anything requiring real external credentials/services

- **SMTP** — alert emails and password-reset emails need real `SMTP_*` credentials; without them,
  the app logs a warning and continues (never crashes), so everything except actual email delivery
  works end-to-end without this.
- **A real RTSP/IP camera or USB webcam** — the live-camera pipeline was verified against a real
  USB webcam during development; an actual RTSP IP camera has not been tested.
- **SMS/WhatsApp** — architectural stubs only (`app/services/notifications/sms_notifier.py`); a
  real provider (e.g. Twilio) needs credentials and a small implementation to make these live.
- **A GPU** — everything here (live detection, video analysis, ML training/export) was built and
  verified CPU-only; a GPU would speed up inference/training but isn't required.
