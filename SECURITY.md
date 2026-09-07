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

**In scope:** authentication and session handling, the role-based access-control checks, the
password-reset flow, file upload handling, path traversal in media serving, SQL injection,
injection into the notification pipeline, the WebSocket authentication path, and secrets handling
in the Docker Compose configuration.

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
- **Authorization.** `admin` / `operator` / `viewer` roles, enforced by FastAPI dependencies on
  every mutating endpoint. Frontend route guards are a usability layer, not the security boundary.
- **Rate limiting.** Per-IP sliding-window limits on signup, login, forgot-password and
  reset-password, configurable through environment variables. Note: the limiter is in-memory and
  therefore per-process — a multi-worker deployment needs a shared store to be effective.
- **CORS.** Restricted to the origins listed in `CORS_ORIGINS`.
- **Upload validation.** Extension and size limits enforced server-side as well as in the browser.
- **Dependency auditing.** `pip-audit` and `npm audit` run on every CI build.

## Privacy note

SentinelCam processes video that may contain identifiable people. If you deploy it, that is
personal data and you are responsible for the legal basis, retention policy, access controls and
any signage or consent your jurisdiction requires. Recorded clips and snapshots are stored
unencrypted on the configured storage path by default — encrypt that volume, restrict access to
it, and set a retention policy before pointing this at a real camera.
