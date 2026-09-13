# Security Policy

## Reporting a vulnerability

**Please do not open a public GitHub issue for a security vulnerability.**

Report it privately through
[GitHub's private vulnerability reporting](https://github.com/longlivewama/SentinelCam/security/advisories/new)
on this repository. If that is unavailable to you, open a public issue containing only the words
"security report — please make contact" and no technical detail, and a maintainer will arrange a
private channel.

When reporting, please include:

- The affected component (backend route, frontend, ML pipeline, Docker configuration).
- Steps to reproduce, or a proof of concept.
- The impact you believe it has.
- Any suggested remediation, if you have one.

You can expect an acknowledgement within **7 days** and an assessment within **30 days**. This is
a personal open-source project maintained on a best-effort basis, not a commercial product with a
funded security team — please calibrate expectations accordingly. Please give a reasonable window
for a fix before public disclosure.

## Supported versions

Only the current `main` branch receives security fixes. There are no maintained release branches.

## Scope

**In scope:** authentication and session handling, the role-based access-control checks,
**row-level authorization** (one user reaching another user's uploads, alerts, clips or analytics,
over REST or the WebSocket), the password-reset flow, file upload handling, path traversal in
media serving, SQL injection, injection into the notification pipeline, the WebSocket
authentication path, and secrets handling in the Docker Compose configuration.

**Out of scope:** the two dependency advisories already documented as accepted risks in the README
(`ecdsa`, reachable only through ECDSA JWT algorithms this app never uses, and `pyasn1`, pinned by
`python-jose`'s own constraint); missing hardening on the intentionally permissive local
development defaults; and denial of service achieved purely by exhausting CPU with very large
uploads on an unlimited configuration — `MAX_UPLOAD_SIZE_MB` exists for that.

## Handling secrets

This repository has never contained a real credential, and must not start.

- Every real `.env` file is gitignored; only documented `.env.example` templates are committed.
- Configuration is read from the environment through `backend/app/config.py`. No credential,
  host or path should ever be hardcoded in source.
- The backend refuses to start with `ENVIRONMENT=production` while `JWT_SECRET_KEY` is still its
  insecure default. Generate a real one with
  `python3 -c "import secrets; print(secrets.token_hex(32))"`.
- For production, use a managed secret store (AWS Secrets Manager, GCP Secret Manager, HashiCorp
  Vault, or your platform's equivalent) rather than `.env` files on disk.
- If you believe a secret has been committed, treat it as compromised: rotate it first, then
  worry about removing it from history.

## Security features in the application

Documented so you know what exists — not as a claim that the implementation is flawless:

- **Authentication.** JWT (HS256, 24 h expiry) with bcrypt-hashed passwords.
- **Password reset.** Single-use tokens stored only as SHA-256 hashes; the raw token exists only
  in the email that is sent.
- **Password reset invalidation.** Consuming a reset token, or changing the password from
  Settings, marks *every* outstanding token for that account used — an older link left in a
  mailbox cannot be replayed against the new password.
- **Role authorization.** `admin` / `operator` / `viewer` roles, enforced by FastAPI dependencies
  on every mutating endpoint. Frontend route guards are a usability layer, not the security
  boundary.
- **Row-level authorization.** Alerts, recordings, analytics and realtime events derived from a
  user's uploaded video are visible only to that user and to operators; rows from a shared camera
  are visible to every authenticated user. The rule lives in one place
  (`backend/app/core/scoping.py`) so every route enforces the same one, and a resource the caller
  may not see returns 404 rather than 403 — confirming an ID exists is itself a disclosure.
- **Media tokens.** A browser cannot attach an `Authorization` header to `<video src>`,
  `<img src>` or an `<a href>` download, so those four endpoints have to take a credential in the
  URL — and a URL is written into every access log, proxy and browser history on the path. What
  travels there is deliberately close to worthless: the client mints a **media token**
  (`POST /api/{recordings|video-uploads|cameras}/{id}/media-token`) that is scoped to that single
  resource by a `res` claim, expires in `MEDIA_TOKEN_EXPIRE_SECONDS` (default 300 s), carries no
  role or email, and is refused everywhere else. The session JWT is **not** accepted in a query
  string on any endpoint, and a media token is refused as a bearer credential on every JSON route
  and on the realtime socket. Minting runs the same ownership check the media endpoint runs, so a
  token can never be obtained for a resource the caller could not already fetch, and the owning
  account is re-resolved on every request — deactivating a user kills their outstanding media
  tokens immediately.
- **Credential redaction in logs.** Uvicorn's access log writes the full request line, query
  string included. `backend/app/core/logging_utils.py` installs a log-record factory plus filters
  on Uvicorn's own loggers, so `token=…` and `Bearer …` are replaced with `[REDACTED]` in every
  record in the process — regardless of which library emitted it — while leaving the path and the
  rest of the query string intact for debugging. The same filter strips the inline userinfo of a
  stream URL (`rtsp://admin:…@host/…` → `rtsp://***:***@host/…`), so a camera's password cannot
  reach a log file even from a call site that does not know it is holding one.
- **Camera credentials and camera sources.** An IP camera is addressed with its credentials
  inline (`rtsp://admin:…@host/…`), so a camera's `url` *is* its password. Cameras are shared
  infrastructure — every authenticated user may list them and watch their streams — but the URL is
  masked for anyone below `operator`, so a `viewer` cannot read the credentials out of the API and
  connect to the hardware directly. Operators and admins still receive the real value, because they
  are the roles that may edit a camera and the edit form round-trips it. Separately, a camera `url`
  must be a stream URL whose scheme is one a camera actually speaks (`rtsp`, `rtsps`, `rtmp`,
  `rtmps`, `http`, `https`, `udp`, `rtp`) or a USB device index: OpenCV hands the string to FFmpeg,
  which otherwise treats `file:`, `concat:`, a bare path and friends as things to open on the
  server's behalf. Private and loopback addresses are deliberately still allowed — IP cameras live
  on private LANs, and narrowing the protocol removes the capability without breaking the product.
- **WebSocket authorization.** Delivery is scoped per subscriber using that same rule, so the
  realtime channel cannot hand a client data the REST API would refuse it. The connection also
  closes itself when the token it was opened with expires, since a long-lived socket cannot
  re-authorize per message, and it rejects media tokens outright — a credential handed out for a
  `<video src>` must not become a live feed of the account. The handshake URL does still carry the
  session token (the native WebSocket API has no header); that is covered by the redaction above
  and recorded as a known limitation in the README.
- **Rate limiting.** Per-IP sliding-window limits on signup, login, forgot-password and
  reset-password, configurable through environment variables. Two caveats: the limiter is
  in-memory and therefore per-process (a multi-worker deployment needs a shared store), and it
  keys on the immediate peer address — behind a reverse proxy every client shares one bucket
  unless the proxy is configured to preserve the client address.
- **CORS.** Restricted to the origins listed in `CORS_ORIGINS`; never `*`.
- **Security headers.** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
  `Referrer-Policy: no-referrer` (the stream and video endpoints carry a scoped media token in
  the URL)
  and a restrictive `Permissions-Policy` on every response. The SPA's own nginx
  (`frontend/nginx.conf`) sets the same four plus a `Content-Security-Policy` whose working part is
  `script-src 'self'` — that origin is the one holding the session token, so an injected script
  there is an account takeover. The media/connect directives stay wide because the API origin is
  chosen at build time (`VITE_API_URL`) and this file cannot name it; narrow them if you template
  the config per deployment.
- **Interactive docs.** `/docs`, `/redoc` and `/openapi.json` are served in development and test
  and are **not mounted** when `ENVIRONMENT=production`, where they would hand an unauthenticated
  visitor a complete map of the API surface.
- **Upload validation.** Server-side extension, size and **container-signature** checks — a file
  is verified to actually be one of the accepted video containers, not merely named like one —
  plus a per-user storage quota, a generated on-disk filename (the client's is never used as a
  path), and a containment check before the file is read back out.
- **Error handling.** Unhandled exceptions return a generic body with a correlation ID; the
  traceback goes to the logs only. Recording responses do not expose server filesystem paths.
- **Container hardening.** The backend server process runs as an unprivileged user (uid 10001);
  the trained model is mounted read-only.
- **Dependency auditing.** `pip-audit` and `npm audit` run on every CI build, alongside a blocking
  scan of each diff for credential-shaped strings and a check that no `.env` or oversized binary
  has been committed.

## Privacy note

SentinelCam processes video that may contain identifiable people. If you deploy it, that is
personal data and you are responsible for the legal basis, retention policy, access controls and
any signage or consent your jurisdiction requires. Recorded clips and snapshots are stored
unencrypted on the configured storage path by default — encrypt that volume, restrict access to
it, and set a retention policy before pointing this at a real camera.
