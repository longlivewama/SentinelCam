<div align="center">

# SentinelCam

**AI-powered fall detection and video surveillance platform — live multi-camera monitoring,
offline video analysis, and realtime alerts, built on YOLO computer vision with a
production-oriented full-stack architecture.**

[![CI](https://github.com/longlivewama/SentinelCam/actions/workflows/ci.yml/badge.svg)](https://github.com/longlivewama/SentinelCam/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![PostgreSQL 16](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-0B0B0B)](https://docs.ultralytics.com/)
[![Tests](https://img.shields.io/badge/tests-142%20passing-success)](#testing)

</div>

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Local development (without Docker)](#local-development-without-docker)
- [Configuration](#configuration)
- [Testing](#testing)
- [Machine learning: models & datasets](#machine-learning-models--datasets)
- [Engineering notes & design decisions](#engineering-notes--design-decisions)
- [Security](#security)
- [Roadmap](#roadmap)
- [Known limitations](#known-limitations)
- [Production deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Falls are one of the leading causes of injury-related hospitalisation for older adults, and in
care homes, hospitals and industrial sites the cost of a fall is driven far less by the fall
itself than by **how long the person lies there before anyone notices**. Camera coverage in those
buildings is usually already in place — what is missing is something watching the feed.

SentinelCam is that layer. It ingests video from IP/USB cameras (or from an uploaded file),
runs pose-based computer vision over the frames, and when it detects a fall it does the three
things a human operator would: records a clip of what just happened, raises an alert in the
dashboard in realtime, and emails the on-call recipients.

**How it works, end to end:**

1. A camera stream (or an uploaded video) is decoded frame by frame.
2. Each processed frame goes through a **fine-tuned YOLOv8 fall detector** — a single-class model
   trained for this project on 11k images (see [the model card](ml/MODEL_CARD.md)) — which locates
   fallen people directly. A YOLOv8 pose model runs alongside it for person counting and for the
   fallback heuristic.
3. A single frame never raises an alert. A detection must **persist for the same tracked subject**
   before an event fires (0.6 s for the trained model, 1.2 s of sustained on-ground posture for the
   heuristic fallback), so bending down or sitting doesn't trigger it, and one bad frame can't
   either.
4. A firing event snapshots a rolling pre-event buffer, appends post-event frames, and writes a
   playable MP4 clip plus a still snapshot.
5. The event is persisted — with its **position in the footage**, its confidence, and which
   detector produced it — pushed to the browsers entitled to see it over a WebSocket, and emailed out.
6. Operators review, play back and acknowledge alerts from the web UI.

The same detection code path serves both live cameras and uploaded files — only the frame source
and the timeline differ — which is what keeps the "analyse this video" feature honest: it is the
production detector running, not a separate demo path.

> **Scope note.** This is an engineering project, **not a certified medical or safety device**.
> The trained detector scores mAP@50 0.877 on a held-out test split of stills; it has **not** been
> evaluated on video, and not against realistic floor-level hard negatives. Those limits are
> spelled out in [`ml/MODEL_CARD.md`](ml/MODEL_CARD.md) and must be read before any operational use.

---

## Features

### Detection & video

| Feature | Status |
|---|---|
| Live multi-camera monitoring (RTSP IP cameras and USB devices) | ✅ |
| MJPEG live stream, one capture connection shared across all viewers | ✅ |
| Pose-based fall detection with a sustained-posture gate | ✅ |
| Additional heuristic detectors: violence, crowd density, abandoned object | ✅ (heuristics — see [notes](#engineering-notes--design-decisions)) |
| Automatic clip recording with a ~3 s pre-event rolling buffer | ✅ |
| Snapshot capture attached to every event | ✅ |
| Offline video upload with drag-and-drop, client-side validation and progress tracking | ✅ |
| Server-side analysis of uploaded video through the same detector | ✅ |
| Per-event confidence scores | ✅ |
| Ranged video playback and clip download | ✅ |

### Application platform

| Feature | Status |
|---|---|
| JWT authentication — signup, login, logout, forgot/reset password | ✅ |
| Role-based access control: `admin` / `operator` / `viewer`, enforced server-side | ✅ |
| Protected frontend routes + role-aware navigation | ✅ |
| Realtime updates over WebSocket (alerts, camera status, upload progress) | ✅ |
| Email notifications (SMTP), with SMS/WhatsApp notifier interfaces stubbed | ✅ / 🧩 |
| Alert acknowledgement workflow | ✅ |
| Dashboard analytics — event counts, trends, per-camera breakdown | ✅ |
| Admin user management (create, disable, change role) | ✅ |
| System status page (database, detector, storage) | ✅ |
| Per-IP rate limiting on all sensitive auth endpoints | ✅ |
| Alembic database migrations, including a data-preserving RBAC migration | ✅ |
| Responsive UI with toast notifications and graceful error states | ✅ |
| One-command Docker Compose stack (Postgres + backend + frontend + mail catcher) | ✅ |
| CI pipeline: lint, backend, frontend, end-to-end and dependency-audit jobs | ✅ |

---

## Architecture

```mermaid
flowchart TB
    subgraph client["Browser"]
        UI["React + Vite SPA<br/>dashboard · alerts · upload · analytics"]
    end

    subgraph api["Backend — FastAPI"]
        REST["REST API<br/>auth · cameras · alerts · recordings · uploads"]
        WS["WebSocket /api/ws/events<br/>RealtimeBroadcaster"]
        RBAC["JWT auth + RBAC<br/>+ rate limiting"]
    end

    subgraph workers["Processing — background threads"]
        STREAM["Stream manager<br/>one capture thread per camera"]
        UPLOAD["Video analysis worker<br/>frame-by-frame single pass"]
        DET["Detection engine<br/>trained YOLO fall detector<br/>+ YOLOv8-Pose (heuristic fallback)"]
        REC["Recording engine<br/>pre/post-event clip writer"]
    end

    subgraph data["Persistence & delivery"]
        DB[("PostgreSQL<br/>SQLAlchemy + Alembic")]
        FS[["Clip / snapshot storage"]]
        MAIL["Notification service<br/>EmailNotifier over SMTP"]
    end

    CAM["IP / USB cameras"] --> STREAM
    UI -->|"REST + JWT"| REST
    UI <-->|"live events"| WS
    UI -->|"video file upload"| REST
    REST --> RBAC
    REST --> UPLOAD
    STREAM --> DET
    UPLOAD --> DET
    DET -->|"event fired"| REC
    REC --> FS
    REC --> DB
    REC --> MAIL
    REST --> DB
    DET --> WS
    UPLOAD --> WS
    REC --> WS
```

**Request/authentication path.** JWT (24 h expiry) with bcrypt-hashed passwords. Password reset
uses one-time SHA-256-hashed tokens emailed to the user — the raw token is never persisted. Every
protected route accepts the token either as an `Authorization: Bearer` header (all JSON API calls)
or as a `?token=` query parameter, because browsers cannot attach custom headers to `<img src>`,
`<video src>`, `<a href>` downloads or the native WebSocket API — so the MJPEG stream, video
playback, downloads and the realtime socket authenticate via the URL instead.

**Authorization.** Two layers. *Role* gates, enforced by FastAPI dependencies (`require_admin`,
`require_operator`) on every mutating endpoint: `admin` has full access including user management;
`operator` manages cameras, acknowledges alerts and analyses video; `viewer` is read-only plus
upload/analysis. And *row-level* scoping (`app/core/scoping.py`), which decides who may see a
given alert, clip or realtime event: anything derived from a user's uploaded video belongs to that
user, anything from a shared camera is visible to every authenticated user, and operators see
everything. The frontend's route guards and hidden buttons are a UX convenience, **not** the
security boundary.

**Streaming.** Each camera gets one lazily-started background thread owning the
`cv2.VideoCapture` connection. It decodes frames and keeps the latest JPEG in a shared buffer,
which `/stream` serves as `multipart/x-mixed-replace` to any number of simultaneous viewers —
without opening a second capture connection per viewer.

**Recording.** That same thread keeps a rolling buffer of the last ~3 seconds of raw frames. When
a detector fires, the recording engine snapshots the buffer, appends ~3 more seconds of live
frames, and writes the combined clip (`avc1`/H.264, falling back to `mp4v`), records it in the
database, and dispatches an email alert with the snapshot attached. Events store which detector
fired them and, for uploads, the fall's **position in the source footage** — so the results screen
can seek the video to the moment itself rather than showing the time the analysis ran.

**Realtime.** An in-process pub/sub broadcaster over a single WebSocket endpoint lets background
worker threads push `alert.created`, `camera.status`, `upload.progress` and `upload.completed`
events to connected tabs instantly — no polling. Delivery is **scoped per subscriber**, using the
same ownership rule as the REST API: camera events go to everyone, upload events only to the
uploader and to operators. The socket also closes itself when the token it was opened with
expires, since a long-lived connection cannot re-authorize per message the way an HTTP request
can.

---

## Tech stack

| Layer | Technologies |
|---|---|
| **Frontend** | React 18, Vite 5, Tailwind CSS 3, Zustand, React Router, Axios |
| **Backend** | Python 3.11, FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2.0, Alembic |
| **Database** | PostgreSQL 16 |
| **Computer vision** | Ultralytics YOLOv8 (pose + object detection), OpenCV |
| **ML runtime** | PyTorch (training), ONNX Runtime (inference) |
| **Realtime** | WebSocket (FastAPI/Starlette) |
| **Auth** | JWT (python-jose, HS256), bcrypt via passlib |
| **Testing** | pytest, Vitest + Testing Library, Playwright |
| **Tooling / CI** | Docker & Docker Compose, nginx, GitHub Actions, ruff, ESLint |

---

## Project structure

```
SentinelCam/
├── backend/                  FastAPI service
│   ├── app/
│   │   ├── api/routes/       auth, cameras, alerts, recordings, video_uploads,
│   │   │                     analytics, admin, realtime, system
│   │   ├── core/             security, RBAC dependencies, rate limiting
│   │   ├── models/           SQLAlchemy models (user, camera, event, recording, …)
│   │   ├── schemas/          Pydantic request/response schemas
│   │   ├── services/         detection engine, stream manager, recording engine,
│   │   │                     video analysis, realtime broadcaster, notifications
│   │   ├── config.py         environment-driven settings
│   │   └── main.py           application entrypoint
│   ├── alembic/              database migrations
│   ├── tests/                69 backend tests (pytest, real PostgreSQL)
│   └── Dockerfile
├── frontend/                 React + Vite single-page app
│   ├── src/
│   │   ├── pages/            dashboard, cameras, alerts, recordings, upload,
│   │   │                     analytics, admin users, auth pages
│   │   ├── components/       route guards, camera tiles, charts, modals, toasts
│   │   ├── store/            Zustand stores (auth, toasts, alert badge)
│   │   ├── lib/              realtime client, formatting, upload validation
│   │   └── api/              Axios client with auth interception
│   ├── nginx.conf            production static-serving config
│   └── Dockerfile
├── ml/                       standalone ML pipeline (not imported by the app)
│   ├── detector/             YOLO fall-detector: dataset download, analysis,
│   │                         preparation and training scripts
│   ├── scripts/              keypoint-feature dataset builders
│   ├── exported/             committed model artifact + metadata (ONNX)
│   ├── train.py evaluate.py export.py
│   └── README.md             full methodology, evaluation and honest caveats
├── e2e/                      Playwright suite driving the real Docker stack
│   ├── tests/                auth, cameras, alerts/recordings, upload
│   └── fixtures/             test media
├── .github/workflows/ci.yml  backend · frontend · e2e · dependency-audit jobs
├── docker-compose.yml        Postgres + backend + frontend + Mailpit
└── .env.example              documented configuration template
```

---

## Quick start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose

That's it — Postgres, the backend, the built frontend and a mail catcher all come up together.

### 1. Clone

```bash
git clone https://github.com/longlivewama/SentinelCam.git
cd SentinelCam
```

### 2. Configure (optional)

Every value has a working local default. To override any of them:

```bash
cp .env.example .env      # then edit as needed
```

Set a real `JWT_SECRET_KEY` for anything beyond a local trial:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### 3. Start the stack

```bash
docker compose up --build
```

### 4. Create the first admin user

In a second terminal, once the stack is healthy:

```bash
docker compose exec backend python seed_admin.py
```

### 5. Open the app

| Service | URL |
|---|---|
| Web UI | http://localhost:5173 |
| API + interactive docs | http://localhost:8000 · http://localhost:8000/docs |
| Mailpit (catches alert & password-reset email) | http://localhost:8025 |

Log in, then either add a camera (an `rtsp://…` URL, or a device index like `0` for a USB
webcam) from the **Cameras** page, or go straight to **Upload** and analyse a video file — no
camera hardware required.

> **Camera access from inside Docker** is host-OS dependent. Linux can pass a USB device through
> with `devices:` in `docker-compose.yml`; Docker Desktop on macOS/Windows generally cannot. RTSP
> IP cameras work over the network regardless. For local USB webcam testing, run the backend
> outside Docker (below).

Recordings, snapshots and uploads persist in named Docker volumes across restarts.
`docker compose down -v` wipes them along with the database.

---

## Local development (without Docker)

**Prerequisites:** Python 3.11+, Node 20.18+ (Node 22 recommended — jsdom 30 needs a recent
Node to run the frontend tests), and a local PostgreSQL server.

```bash
# 1. Databases
createdb sentinelcam
createdb sentinelcam_test          # only needed for the backend test suite

# 2. Backend
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # fill in DATABASE_URL, JWT_SECRET_KEY, SMTP_* …
alembic upgrade head               # apply the schema
python seed_admin.py               # create the first admin user
uvicorn app.main:app --reload      # http://localhost:8000  (docs at /docs)

# 3. Frontend (separate terminal)
cd frontend
npm install
cp .env.example .env               # VITE_API_URL=http://localhost:8000
npm run dev                        # http://localhost:5173
```

### Database migrations

Schema changes are managed with Alembic; `alembic upgrade head` is the production migration path.

```bash
cd backend
alembic upgrade head                                      # apply pending migrations
alembic revision --autogenerate -m "describe the change"  # after changing a model
```

`Base.metadata.create_all()` also runs once on startup as a convenience for fresh dev databases —
it only ever *adds missing tables* and never alters existing ones, so it is a safety net, not a
substitute for migrations against a database that already holds data.

---

## Configuration

All configuration is read from the environment. Never hardcode credentials — every real `.env`
file is gitignored, and only `.env.example` templates are committed.

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `JWT_SECRET_KEY` | JWT signing secret. The app **refuses to start** if left at its insecure default while `ENVIRONMENT=production` |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins |
| `FRONTEND_URL` | Base URL used to build links in emails (e.g. password reset) |
| `NOTIFICATION_CHANNELS` | Comma-separated enabled channels (`email` works out of the box) |
| `SMTP_*`, `ALERT_RECIPIENTS` | Email delivery configuration |
| `UPLOADS_DIR`, `MAX_UPLOAD_SIZE_MB`, `ALLOWED_VIDEO_EXTENSIONS` | Video upload limits |
| `MAX_UPLOAD_STORAGE_PER_USER_MB` | Per-account storage quota (0 disables it) |
| `MAX_CONCURRENT_VIDEO_ANALYSES` | Upper bound on simultaneous upload analyses; the rest queue |
| `FALL_DETECTOR_MODEL_PATH` | The trained fall detector. Defaults to the committed artifact; empty forces the pose heuristic |
| `FALL_DETECTION_MODE` | `auto` \| `model` \| `heuristic` \| `hybrid` — see [Machine learning](#machine-learning-models--datasets) |
| `FALL_DETECTOR_MIN_CONFIDENCE`, `FALL_DETECTOR_MIN_SUSTAINED_SECONDS` | Detection threshold and how long a detection must persist before it alerts |
| `FALL_CLASSIFIER_MODEL_PATH` | Optional path to the earlier ONNX keypoint classifier (empty = off, the default) |
| `AUTH_*_MAX_REQUESTS` / `AUTH_*_WINDOW_SECONDS` | Per-IP rate limits on signup, login, forgot-password and reset-password |
| `VITE_API_URL` *(frontend)* | Backend base URL; the WebSocket URL is derived from it automatically |

The full commented list lives in [`.env.example`](.env.example),
[`backend/.env.example`](backend/.env.example) and `frontend/.env.example`.

---

## Testing

**261 automated tests across three suites**, all currently passing, plus a four-job CI pipeline.

| Suite | Count | What it covers |
|---|---:|---|
| **Backend** (pytest) | **169** | Security primitives, fall-detection logic (both the trained-model gate and the pose heuristic), the committed model artifact's contract, cross-user isolation, auth & RBAC, cameras, alerts, recordings, upload validation, HTTP Range handling, realtime delivery scoping, analytics, rate limiting, Alembic migrations + schema-drift detection, email notifier, and an end-to-end video analysis over a real decoded video |
| **Frontend** (Vitest + Testing Library) | **76** | Auth pages, route guards, upload workflow (drag-drop, validation, progress, delete), the results screen's in-video timestamps and detector labelling, system status including the model-fallback warning, dashboard, formatting helpers, Zustand stores |
| **End-to-end** (Playwright) | **16** | Real Chromium against the real Docker Compose stack — signup, login, logout, forgot/reset password (reading the actual email out of Mailpit's API), camera CRUD + RBAC, alert acknowledgement, recording playback, and the full upload → process → view-results workflow |
| **Docker** | 4 services | `postgres`, `backend`, `frontend`, `mailpit` — all healthy under `docker compose up` |

```bash
# Backend — runs against a real PostgreSQL database, not a mock
cd backend && source venv/bin/activate
createdb sentinelcam_test        # one-time
pytest
ruff check .

# Frontend
cd frontend
npm run lint && npm run test && npm run build

# End-to-end (brings the Docker stack up, seeds fixtures, runs, tears down)
cd e2e
npm install
npx playwright install --with-deps chromium   # one-time
npm test
```

Backend tests deliberately run against real Postgres rather than a sqlite substitute — the app
relies on Postgres-specific behaviour (boolean filters, timezone-aware timestamps) in enough
places that sqlite would give false confidence.

No coverage percentage is published here, because none has been measured.

---

## Machine learning: models & datasets

The `ml/` directory is a **standalone training pipeline**. It is not imported by the running
backend — it produces the model artifacts the backend loads.

### The production fall detector

**`ml/exported/fall_detector_v1.pt` — YOLOv8n, single class `Fall`, 5.4 MB.** This is the model
the application runs. Full details, honest limitations and reproduction steps live in
**[`ml/MODEL_CARD.md`](ml/MODEL_CARD.md)**; the short version:

| | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Validation (2,189 images) | 0.812 | 0.710 | 0.826 | 0.528 |
| **Held-out test (2,176 images)** | **0.846** | **0.800** | **0.877** | **0.557** |

Trained for 80 epochs (18.5 h on Apple Silicon) from COCO-pretrained weights on a public Roboflow
fall-detection dataset (CC BY 4.0), prepared into 11,074 train / 2,189 val / 2,176 test images.
Test scores sit slightly *above* validation across all four metrics — no sign of overfitting, and
307 near-duplicate images were moved out of val/test into train first so the test split is
genuinely held out.

Two preparation decisions shape what this model is, and both came out of a dataset analysis run
*before* any training ([`ml/reports/fall_detector_dataset_analysis.md`](ml/reports/fall_detector_dataset_analysis.md)):

- **The raw export's second class, `Person`, was dropped.** Not one image of 15,439 contained both
  a `Fall` and a `Person` box — `Person` was entirely PASCAL VOC, `Fall` entirely separate fall
  datasets. A two-class detector could have scored well by learning *which dataset an image came
  from*. Dropping `Person` removes the shortcut and repurposes those 6,887 VOC images as background
  negatives: examples of upright people that must not fire.
- **Augmentation follows from the task, not from defaults.** This class is defined by orientation
  relative to gravity, so vertical flips are disabled outright and rotation is capped at 5°;
  anything more would rotate standing people into lying poses while keeping the "not a fall" label.
  Mosaic is kept at 1.0 precisely *because* no source image contains both a fall and an upright
  person — compositing four images synthesises the co-occurrence the dataset structurally lacks.

**Known limitations, stated plainly:** no video-level evaluation (all metrics are per-frame on
stills), no hard-negative validation against floor-level activity like sit-ups or crouching, and
source data that is not care-home footage. See the model card's Limitations section in full before
deploying it anywhere real.

### Integration

```
frame ─▶ fall_object_detector.detect()   YOLO inference, conf ≥ 0.4
      ─▶ ModelFallDetector.update()       tracking + sustain gate + debounce
      ─▶ Event(detector, confidence, video_timestamp_seconds)
```

| Setting | Default | Effect |
|---|---|---|
| `FALL_DETECTOR_MODEL_PATH` | `ml/exported/fall_detector_v1.pt` | Empty forces the pose heuristic |
| `FALL_DETECTION_MODE` | `auto` | `auto` \| `model` \| `heuristic` \| `hybrid` |
| `FALL_DETECTOR_MIN_CONFIDENCE` | `0.4` | Per-box confidence floor |
| `FALL_DETECTOR_MIN_SUSTAINED_SECONDS` | `0.6` | How long a detection must persist |

`auto` runs the trained model when it loads and falls back to the pose heuristic when it doesn't,
so a deployment without the checkpoint still detects falls. That fallback is **never silent**:
`GET /api/system/status` and the System Status page report both the configured and the *active*
mode, and flag the degradation explicitly.

Inference costs ~120 ms per 640 px frame on CPU. For live cameras this is a *third* model per
processed frame alongside pose and object detection — budget for it, or raise
`DETECTION_FRAME_STRIDE`.

### The earlier keypoint classifier (secondary, disabled by default)

`ml/exported/fall_classifier_v1.onnx` is a 10-feature keypoint MLP from an earlier iteration. It is
**not** the production detector and has a much weaker standing: its 1.000 test metrics are flagged
in [`ml/reports/eval_report.md`](ml/reports/eval_report.md) as likely domain-shortcut artifacts. It
was only ever wired in as a confidence nudge on top of the heuristic's gate, never as a source of
truth, and remains off unless `FALL_CLASSIFIER_MODEL_PATH` is set.

### What is and is not in this repository

| Artifact | In Git? | Why |
|---|---|---|
| Detector + classifier pipeline source (`ml/detector/`, `ml/scripts/`) | ✅ | The reproducible part |
| **Trained fall detector** (`ml/exported/fall_detector_v1.pt` + metadata) | ✅ | 5.4 MB — the actual deliverable, so a fresh clone runs the real model |
| Detector metrics: results CSV, PR/F1 curves, confusion matrices | ✅ | So the model card's numbers are checkable without retraining |
| Exported keypoint classifier (`ml/exported/fall_classifier_v1.*`) | ✅ | Small, and the deliverable of that earlier pipeline |
| Training **datasets** (`ml/data/…`) | ❌ | Large, redistributable only under their own licences — re-fetch with the download scripts |
| Training **runs & checkpoints** (`ml/runs/`) | ❌ | ~16 MB per run of per-epoch checkpoints and batch mosaics; the selected artifact is promoted to `ml/exported/` |
| Training **logs** (`ml/logs/`) | ❌ | Machine-local noise |
| Auto-downloaded YOLO base weights (`*.pt` at repo root) | ❌ | Fetched on demand by Ultralytics |

The committed weights are fine-tuned from **AGPL-3.0** Ultralytics COCO weights on a **CC BY 4.0**
dataset — see [`NOTICE.md`](NOTICE.md) before any commercial use or redistribution.

Reproduce the model with `download_dataset.py` → `analyze_dataset.py` → `prepare_dataset.py` →
`train.py` in `ml/detector/` (a stratified smoke-test subset is available for fast iteration). Do
not overwrite `fall_detector_v1.pt`; promote a retrained model as `v2` alongside it.

---

## Engineering notes & design decisions

Three of the four live-camera detectors were specified against research-grade models that require
specific datasets and substantial GPU time (RWF-2000 for violence, ShanghaiTech/CSRNet for crowd
density, DeepSORT for tracking). Rather than ship an untrained model behind a confident-sounding
name, each was implemented as a practical, honestly-scoped equivalent that satisfies the same
interface and can be swapped later **without touching the rest of the system**:

| Detector | Research-grade approach | What is actually implemented | Upgrade path |
|---|---|---|---|
| **Fall** | YOLOv8-Pose, 3-signal heuristic | Pose heuristic (aspect ratio / keypoint alignment / hip velocity) gated by a sustained on-ground-posture requirement (≥1.2 s continuous, not a single frame) to reject bending and sitting, plus an optional trained classifier as a corroborating signal | Retrain on real footage; add the YOLO fall detector from `ml/detector/` |
| **Violence** | CNN+LSTM trained on RWF-2000 | Heuristic over the same pose keypoints: erratic high-velocity wrist/elbow motion between people in close proximity | Swap in a trained temporal model via `VIOLENCE_MODEL_PATH` |
| **Crowd counting** | CSRNet density-map estimation | YOLOv8 person-class detection and count, smoothed over frames, thresholded per camera | Swap in CSRNet for very dense or occluded scenes |
| **Abandoned object** | YOLO + DeepSORT | YOLO object detection plus a small centroid tracker | Swap in DeepSORT/ByteTrack for robust ID persistence through occlusion |

Other decisions worth calling out:

- **One capture thread per camera, not per viewer** — the naive implementation opens a new
  `VideoCapture` per connected browser and collapses at three viewers.
- **Sustained-posture gating over per-frame classification** — a single-frame "on the ground"
  signal fires constantly on people bending, sitting and crouching. The time gate is what makes
  the alert stream usable.
- **Shared detection path for live and uploaded video** — the upload feature runs the production
  detector against the file's own timeline, so it is a genuine test of the real pipeline.
- **Notifier interface with real stubs** — `SmsNotifier` and `WhatsAppNotifier` log an explicit
  "not configured" warning rather than silently no-op, so wiring a real provider later is a config
  change plus one class, not a call-site rewrite.

---

## Security

- **Secrets live in environment variables only.** No credential is committed; every real `.env`
  is gitignored and only documented `.env.example` templates ship with the repository.
- **Production start-up guard.** The backend refuses to boot with `ENVIRONMENT=production` while
  `JWT_SECRET_KEY` is still the insecure default.
- **Authentication.** JWT with bcrypt-hashed passwords. Password-reset tokens are single-use and
  stored only as SHA-256 hashes — the raw token exists solely in the email.
- **Authorization.** Role checks are enforced server-side on every mutating endpoint; the frontend
  guards are UX, not security.
- **Rate limiting.** Per-IP sliding-window limits on signup, login, forgot-password and
  reset-password, all tunable via environment variables.
- **Dependency auditing.** `pip-audit` and `npm audit` run on every CI build (informational, so a
  new advisory surfaces for review rather than silently blocking unrelated work).
- **For production deployments,** use a managed secret store (AWS Secrets Manager, GCP Secret
  Manager, Vault, or your platform's equivalent) rather than `.env` files on disk, terminate TLS at
  a reverse proxy, and restrict `CORS_ORIGINS` to your real frontend origins.

Found a vulnerability? Please report it privately — see [SECURITY.md](SECURITY.md). Do not open a
public issue for security reports.

---

## Roadmap

Recently completed:

- [x] **Trained YOLO fall detector integrated as the primary detector**, with a published
      held-out evaluation — see [`ml/MODEL_CARD.md`](ml/MODEL_CARD.md).
- [x] **In-video fall timestamps persisted**, so uploaded-video results seek to the exact moment
      of each detection rather than showing a wall-clock time.
- [x] **Row-level authorization** on alerts, recordings, analytics and the realtime channel.

Planned:

- [ ] **Video-level model evaluation** — the largest remaining gap. Every published metric is
      per-frame on stills; the numbers that matter operationally (falls detected per fall that
      occurred, false alerts per hour of ordinary footage) require a labelled fall *video* corpus
      and are currently unmeasured.
- [ ] **Hard-negative validation** — evaluate against realistic floor-level activity (sit-ups,
      crouching, a child playing, someone lying on a sofa), none of which the current test split
      contains.
- [ ] **Inference performance** — batched/strided frame processing and optional GPU acceleration
      for faster analysis of long recordings.
- [ ] **Horizontally scalable realtime** — move the rate limiter and event broadcaster to Redis so
      the backend can run multiple workers or instances.
- [ ] **Pagination** on the alert, recording and upload listings. They are now bounded (500
      newest by default, 1000 max) rather than unbounded, but a deployment past that cap needs
      real paging rather than a larger ceiling.
- [ ] **Richer analytics** — per-camera heatmaps, time-of-day distributions, exportable reports.
- [ ] **More notification channels** — SMS and WhatsApp providers behind the existing interfaces.
- [ ] **Observability** — structured logging, metrics and health dashboards.
- [ ] **Model version management** — track, compare and roll back deployed detector versions.
      The artifact + metadata sidecar convention (`fall_detector_v1.pt` alongside
      `fall_detector_v1.metadata.json`) is the groundwork; a registry and a UI are not.
- [ ] **Deployment reference** — a documented production deploy (managed Postgres, object storage
      for clips, TLS termination).

---

## Known limitations

Stated plainly, because they matter when reading the rest of this document:

1. **The fall detector has never been evaluated on video.** Its published metrics (precision
   0.846 / recall 0.800 / mAP@50 0.877 on a held-out test split) are **per frame, on still
   images**. Per-*incident* recall is certainly higher — a real fall is sampled dozens of times —
   but it is not measured, and neither is the false-alert rate over ordinary footage. It also has
   no validation against floor-level hard negatives (sit-ups, crouching, lying on a sofa). See
   [`ml/MODEL_CARD.md`](ml/MODEL_CARD.md) for the full list.
2. **Deleting a video while it is being analysed is an unresolved backend race.** The analysis
   worker and the delete endpoint can interleave; the failure is contained (the worker records a
   failed status rather than crashing the service), but the correct fix — cooperative cancellation
   of the worker — is not implemented.
3. **Analysis duration varies with machine load.** Everything here is CPU-verified with no GPU
   requirement, so wall-clock analysis time for a given video depends heavily on what else the
   host is doing.
4. **Violence, crowd and abandoned-object detectors are heuristics**, not trained models — a known
   and documented scope boundary, not an accidental gap.
5. **The earlier keypoint classifier's metrics are optimistic** — small, narrow, largely
   synthetic dataset; see [`ml/README.md`](ml/README.md). It is off by default and is not the
   production detector.
6. **Single-process assumptions.** The in-memory rate limiter and realtime broadcaster are
   per-process; multi-worker deployment needs a shared backing store.
7. **E2E coverage boundaries.** The Playwright suite does not cover live-camera streaming (no
   camera hardware in CI) or a genuine ML-detected fall — its upload fixture is deliberately
   person-free so the assertion stays honest. Both were verified manually in a real browser.
8. **Two residual dependency advisories are accepted rather than force-fixed:** `ecdsa` (reachable
   only via ECDSA JWT algorithms; this app uses HS256 exclusively) and `pyasn1` (pinned by
   `python-jose`'s own constraint). Both are documented rather than hidden.
9. **Untested externally:** a real RTSP IP camera (verified against a USB webcam instead), and
   real SMTP delivery (verified against Mailpit).

---

## Production deployment

1. Set `ENVIRONMENT=production` and a real random `JWT_SECRET_KEY` — the app refuses to boot
   otherwise.
2. Run `alembic upgrade head` against the production database as a deploy step; do not rely on
   `create_all`.
3. Run the backend behind a real ASGI configuration (`uvicorn app.main:app --workers N` behind a
   reverse proxy). Note that the in-memory rate limiter and realtime broadcaster are per-process —
   scaling past one worker needs a shared store (see the docstrings in `app/core/rate_limit.py`
   and `app/services/realtime.py`).
4. Serve the frontend build (`frontend/dist/`) from a static host or CDN, with `VITE_API_URL`
   baked in at build time.
5. Configure real SMTP credentials for `NOTIFICATION_CHANNELS=email`, and set `CORS_ORIGINS` to
   your real frontend origins.

### External services you will need for full functionality

| Dependency | Needed for | Without it |
|---|---|---|
| SMTP credentials | Alert and password-reset email | Logs a warning and continues — everything except delivery works |
| An RTSP/IP or USB camera | The live-camera pipeline | Video upload and analysis still work end to end |
| SMS / WhatsApp provider | Those notification channels | Stubs log "not configured"; no silent failures |
| A GPU | Faster inference and training | Everything here is CPU-verified; a GPU is an optimisation, not a requirement |

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `JWT_SECRET_KEY is still set to its insecure default` on boot | `ENVIRONMENT=production` with the placeholder secret. Generate one: `python3 -c "import secrets; print(secrets.token_hex(32))"` |
| `alembic upgrade head` fails on a fresh database | `DATABASE_URL` must point at a database that already exists (`createdb sentinelcam`) with table-creation privileges |
| Live camera stream never loads | Check the camera `url` (numeric string for a USB index, full `rtsp://…` for IP) and the backend log for `cv2.VideoCapture` open failures. The tile degrades to "Stream unavailable" rather than breaking the page |
| Recording won't play in Safari | Check which codec the backend logged — `mp4v` (the fallback) is far less broadly compatible than `avc1` |
| Uploaded video stuck at "processing" | `video_analysis.py` records real failures as `status=failed` with an `error_message`, so a genuinely *stuck* item is usually a long file still processing on CPU |
| Realtime badge stays on "Reconnecting" | The WebSocket uses the same JWT as everything else — check expiry, then the backend log for the `/api/ws/events` connection attempt |
| `pytest` fails with a connection error | `createdb sentinelcam_test` first; the suite needs a real reachable Postgres at the `DATABASE_URL` its `conftest.py` defaults to |

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the development
setup, coding conventions, commit-message format and the checks a pull request must pass.

## License

Released under the [MIT License](LICENSE).

Third-party models and datasets carry their own licences — in particular, Ultralytics YOLOv8 is
AGPL-3.0 licensed, and the fall-detection dataset is governed by the terms of its Roboflow export.
See [NOTICE.md](NOTICE.md) and review those terms separately before any commercial use.
