# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- **Media tokens replace the session JWT in media URLs.** `<video src>`, `<img src>` and download
  links cannot send an `Authorization` header, so those four endpoints took the full 24-hour JWT
  in a `?token=` query parameter — and from there into Uvicorn's access log. The client now mints
  a short-lived token per resource (`POST /api/{recordings|video-uploads|cameras}/{id}/media-token`,
  default 300 s) that is scoped to that one clip, upload or camera, carries no role, and is
  refused as a bearer credential on every JSON route and on the realtime socket. A session JWT is
  no longer accepted in a query string anywhere.
- **The realtime WebSocket rejects media tokens.** It previously accepted any valid JWT, so a
  token handed out for a video element opened a live feed of every alert the account could see.
- **Credentials are redacted from every log record.** A log-record factory plus filters on
  Uvicorn's loggers replace `token=…` and `Bearer …` with `[REDACTED]`, covering the access log,
  the WebSocket handshake line and any third-party library that logs a full URL.

### Added

- **Fall clips are annotated with the tracked person who fell.** Every fall clip an uploaded video
  produces now carries a bounding box that follows its subject frame by frame, labelled
  `FALL DETECTED` with the track id and the event's own confidence (`ID 1 | 0.74`). A short-term
  tracker gives each person in the scene a temporary id from the pose boxes the pipeline already
  computes, and the fall is attributed to one of them by containment, IoU, centre distance and
  continuity across the sustain window — so with several people on screen only the one who fell is
  boxed. Where no person can be credibly named (an even split between two candidates, or a
  detection that fired on something that is not a person), the clip shows the detector's own region
  as a dashed box marked `UNMATCHED` rather than inventing a track id. Purely an annotation layer:
  it runs no additional inference, and fall timings, confidences, thresholds, clip boundaries,
  events, alerts and rows are byte-for-byte what they were without it. Measured cost on a 29 s
  576×1024 video: 7.284 s → 7.359 s (+1.0%), of which tracking and association are 3 ms.

- **Real pagination** on the alert, recording, video-upload and admin-user listings. Offset paging
  at the database (never fetching all rows and slicing in Python) with a
  `{items, page, page_size, total, pages}` envelope, ownership filtering applied in SQL before the
  count, and a primary-key tiebreaker on every sort so a row cannot appear on two pages when
  timestamps collide. Replaces the `limit` parameter, which bounded the response but left older
  rows unreachable. Previous/next controls, server-side filters and preserved filter state in the
  UI.
- **`trigger_action` filter on `GET /api/recordings`**, so the Recordings page's event-type filter
  keeps working now that the client no longer receives every row.
- **ML validation: F1, confidence distributions, per-condition analysis and corpus coverage.** The
  evaluator now reports incident F1, the confidence distribution of true detections versus false
  alerts (with a histogram straddling the production threshold), recall split by fall direction
  and speed, and results split by lighting, camera angle, distance, occlusion and resolution.
  `Corpus.missing_coverage()` checks a corpus mechanically against the dataset specification in
  `ml/validation/README.md` and every report states what is missing on its own front page.

### Changed

- **Fall clips encode with VP8 instead of VP9** — 4.8x faster to encode, 18% smaller, and 0.27 dB
  apart on PSNR (with a better worst-frame figure), equally playable in every browser targeted.
  Clip encoding runs inline in the upload analyser and was ~70% of its wall clock: a real 29 s
  video went from 77.6 s to 36.3 s end to end, **53% faster with identical detection output**.

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
