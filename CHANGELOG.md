# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### In progress

- Training a single-class YOLOv8n `Fall` object detector on a prepared fall-detection dataset, to
  sit alongside the existing pose heuristic. Source code for the download, analysis, preparation
  and training steps lives in `ml/detector/`; the resulting weights are not distributed here.

## [0.2.0] — 2026-09-07

### Added

- **Role-based access control** — `admin`, `operator` and `viewer` roles enforced server-side by
  FastAPI dependencies on every mutating endpoint, with role-aware frontend navigation and route
  guards, and an admin user-management page (create, disable, change role).
- **Video upload and offline analysis** — drag-and-drop upload with client-side validation, size
  and extension limits, progress tracking, and server-side analysis that runs the same pose model
  and fall detector used for live cameras, against the video's own timeline. Detected events
  produce pre/post-event clips exactly like a live camera would.
- **Realtime event delivery** — a WebSocket endpoint (`/api/ws/events`) with an in-process
  broadcaster, pushing `alert.created`, `camera.status`, `upload.progress` and `upload.completed`
  to every connected browser tab without polling.
- **Password reset flow** — single-use, SHA-256-hashed tokens delivered by email; the raw token is
  never persisted.
- **Rate limiting** — configurable per-IP sliding-window limits on signup, login, forgot-password
  and reset-password.
- **Notification architecture** — a `Notifier` interface with an SMTP `EmailNotifier`
  implementation, fanned out by a `NotificationService`, plus explicit SMS and WhatsApp stubs that
  warn rather than silently no-op.
- **Corroborating fall classifier** — a keypoint-feature model trained in `ml/` and exported to
  ONNX, loadable via `FALL_CLASSIFIER_MODEL_PATH` as a confidence adjustment on top of the
  heuristic's sustained-posture gate (never as a bypass).
- **Alembic migrations**, including a data-preserving RBAC migration that backfills `role` from
  the previous `is_admin` column before dropping it.
- **Docker Compose stack** — Postgres, backend, nginx-served frontend build, and a Mailpit SMTP
  catcher, with a documented `.env.example`.
- **Playwright end-to-end suite** (16 tests) driving the real stack in a real browser, including
  reading password-reset email out of Mailpit's API.
- **CI pipeline** — GitHub Actions jobs for backend (lint + pytest against real Postgres),
  frontend (lint + Vitest + build), end-to-end, and dependency auditing.
- **Analytics dashboard** and a system-status endpoint reporting model, detection, notification
  and storage state.

### Changed

- Production start-up now aborts if `JWT_SECRET_KEY` is left at its insecure default while
  `ENVIRONMENT=production`.
- Fall detection gained a sustained on-ground-posture requirement (≥1.2 s continuous) to reject
  the bending and sitting false positives that per-frame classification produces.

### Known issues

See [Known limitations](README.md#known-limitations) — most notably that in-video fall offsets are
not persisted, and that deleting a video mid-analysis remains an unresolved backend race.

## [0.1.0] — 2026-09-06

### Added

- Initial prototype: FastAPI backend with JWT authentication, camera CRUD, MJPEG live streaming
  with one shared capture thread per camera, a YOLOv8-pose detection engine with fall, violence,
  crowd and abandoned-object detectors, a recording engine with a rolling pre-event buffer, and
  email alerting.
- React + Vite frontend with dashboard, cameras, alerts, recordings and settings pages.
