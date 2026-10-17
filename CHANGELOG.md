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

- **Human pose estimation and multi-person tracking.** The YOLOv8n-pose pass that already ran on
  every processed frame now also feeds a per-session tracker (`SceneTracker`, wrapping the existing
  `PersonTracker`): persistent track IDs with tentative/confirmed/lost states, 17 COCO keypoints per
  person, and per-track lifetimes - no extra inference. Live cameras publish each processed frame
  as a `tracks.update` WebSocket message; uploaded videos get a `tracking.json` record beside their
  clips, served by `GET /api/video-uploads/{id}/tracking` and deleted with the upload. Fall events
  are attributed to the tracked person they belong to (`person_track_id`) with pose evidence.
- **Pose/tracking overlays** on the live camera view and the uploaded-video results: boxes,
  skeletons, track IDs and confidence as one SVG layer aligned with the media at any size, each
  layer toggleable; a per-person timeline in the results; a Pose & tracking card on System Status.
- **Fall assists from pose and tracking:** deferred pose verification (`FALL_POSE_VERIFICATION`,
  on by default) and per-person dedupe (`FALL_TRACK_DEDUPE`, off), with
  `python -m ml.validation.compare_fall_assist` to evaluate them. An immediate veto cut false alerts
  on the URFD corpus (11 -> 3-4) but lost 2-3 of 10 real falls, so it was not shipped. The deferred
  form holds an event whose person is still upright for up to 2 s and withholds it only if they stay
  upright. On 22 clips (11 falls) it caught 11/11 with 7 false alerts, against 9/11 and 17 with
  verification off. That corpus is far too small to claim a real-world accuracy change (see README).
- **Uploads in codecs browsers cannot decode now play in the results view.** An upload stored as
  MPEG-4 Part 2 (common from OpenCV, cameras and public datasets) analysed fine, but its player
  stayed blank. Analysis now also writes a frame-aligned VP8/H.264 preview, which the stream
  endpoint serves in its place and which is deleted with the upload.
- **Trained YOLOv8n fall detector as the primary fall signal.** `ml/exported/fall_detector_v1.pt`
  (single class `Fall`, trained by `ml/detector/`) is committed with its metadata and loaded by
  default; the pose heuristic remains as a reported fallback. Metrics and limitations are in
  `ml/MODEL_CARD.md`.
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

- **Analysis results view.** A wider two-column layout: the source video, overlay controls and
  per-person timeline on the left, fall-event evidence on the right, so the overlay and the evidence
  for it are visible together. Pose evidence is a readable sentence rather than an all-caps badge,
  and the event list states that each event is a detector lead for review, not a confirmed fall.
- **Dialogs** are announced as modal dialogs named by their title, close on Escape, and move focus
  in on open and back on close.
- **Event markers** use a coloured status dot from the design tokens instead of emoji, on the
  dashboard, alerts, recordings and camera pages. Dashboard alert rows now show their source and
  confidence.
- **System Status** lays its cards out in two columns on wide screens.
- **The camera form no longer shows an "Abandoned object" timer.** That detector was removed from
  the live pipeline, so the field configured nothing. The column remains for compatibility.
- **Fall clips encode with VP8 instead of VP9** — 4.8x faster to encode, 18% smaller, and 0.27 dB
  apart on PSNR (with a better worst-frame figure), equally playable in every browser targeted.
  Clip encoding runs inline in the upload analyser and was ~70% of its wall clock: a real 29 s
  video went from 77.6 s to 36.3 s end to end, **53% faster with identical detection output**.

### Fixed

- On the live view, a fall event held by pose verification drew its fire-time box where the person
  *had been* up to 2 s earlier. Its attributed person's current box is drawn instead.
- Dashboard and analytics charts: axis labels drifted out from under their bars and truncated to
  `10-…`, an all-zero series rendered as a bare baseline, and a single category filled the whole
  card as one slab.
- The navbar wrapped the user name onto two lines at common laptop widths, and labelled the first
  realtime connection of a page load "Reconnecting".
- A fall event held by pose verification was stamped with the time it was *released* (up to 2 s
  late, or the last frame when the video ended), shifting its in-video timestamp, overlay highlight
  and clip. It is now placed where the detector fired.
- A held event's clip re-matched the fire-time box against the people visible at release, and could
  annotate a different person than the event named. It now reuses the event's attribution.
- `POSE_ESTIMATION_ENABLED=false` also stopped pose on live cameras, silently disabling violence
  detection, which works from keypoints. The flag now only affects uploaded-video analysis.
- A second thread could get past model loading while the first was still inside `YOLO(...)`, and
  run inference against an unloaded model for those frames.
- **Concurrent first inference on a shared model crashed.** Two threads making their first
  `predict()` on the same freshly loaded Ultralytics model (two uploads, or a camera and an upload,
  after a restart) crashed one of them with `'Conv' object has no attribute 'bn'` in 15 of 15
  trials. Inference on each shared model is now serialized; 0 of 15, no measurable throughput cost.
- **A model that failed to load stopped detection.** A missing or corrupt pose/object model now
  disables only what needs it and is reported on System Status, instead of failing every analysis.
- **Live camera views no longer exhaust the database pool.** `GET /api/cameras/{id}/stream` kept
  its request's database session checked out for the life of the stream, so about 15 concurrent
  views took the whole SQLAlchemy pool and every database-backed endpoint hung. The stream now
  releases its connection before streaming, yields from an async generator, and ends when the
  viewer disconnects — including on a camera that never produces a frame, which previously looped
  forever and blocked graceful shutdown. `SecurityHeadersMiddleware` is now pure ASGI, because
  `BaseHTTPMiddleware` hid that disconnect from the stream.
- **Oversized uploads are refused while they arrive.** The 500 MB limit was checked only after
  the multipart body had been spooled to disk in full. A new `RequestBodyLimitMiddleware` rejects
  a declared `Content-Length` over the limit immediately and cuts off a chunked body at the limit.
- **Deleting a user who owns analysed uploads no longer fails with a 500.** The staged upload
  deletes are flushed before the user row, so the `video_uploads.user_id` foreign key is satisfied.
  Their per-upload clip directories are now removed too.
- **Snapshots are deleted with their clips.** Snapshot JPEGs outlived their upload, recording,
  user and the retention script. Writers and deleters now share `snapshot_path_for_clip`.
- **Editing a running camera's URL takes effect immediately.** The capture thread kept retrying
  the old URL until the process restarted.
- **Unplayable videos report the failure.** The player's retry re-minted a media token within the
  same second, received a byte-identical token, and never reloaded — leaving a silent blank
  player instead of the "could not be played" message.

### Documentation

- Rewrote the README around the verified implementation, with genuine application screenshots,
  an architecture diagram and current test results; documented safe E2E isolation.

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
