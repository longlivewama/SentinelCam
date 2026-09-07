# SentinelCam E2E Suite

Playwright tests that drive the real app in a real browser against the real
Docker Compose stack (Postgres, backend, frontend, and a Mailpit SMTP
catcher) — no mocking. Covers: signup, login, logout, forgot/reset password
(including reading the actual password-reset email out of Mailpit), camera
CRUD + RBAC (viewer can't manage cameras), alert acknowledgment, recording
playback, and the full video-upload → processing → results workflow.

## Running

```bash
cd e2e
npm install
npx playwright install --with-deps chromium   # one-time
npm test
```

That's it — `npm test` brings the whole stack up with `docker compose up
--build --wait` (see `global-setup.js`), seeds fixture data (an admin user,
one camera, one recording+alert — see `backend/scripts/seed_e2e_fixtures.py`),
runs the tests, and tears the stack down afterward. Uses whatever
`docker-compose.yml` in the repo root defines, so it's the same topology
in CI and locally.

View the HTML report after a run:

```bash
npm run report
```

## Running against already-running dev servers (faster iteration)

If you're actively working on a test and don't want to rebuild/restart
Docker every time:

```bash
# Terminal 1: point your local backend at a throwaway e2e database
cd backend && source venv/bin/activate
DATABASE_URL=postgresql://sentinelcam:sentinelcam@localhost:5433/sentinelcam_e2e \
  AUTH_SIGNUP_MAX_REQUESTS=1000 AUTH_LOGIN_MAX_REQUESTS=1000 \
  AUTH_FORGOT_PASSWORD_MAX_REQUESTS=1000 AUTH_RESET_PASSWORD_MAX_REQUESTS=1000 \
  NOTIFICATION_CHANNELS=email SMTP_HOST=localhost SMTP_PORT=1025 SMTP_USE_TLS=false \
  uvicorn app.main:app --reload

# Terminal 2: a local Mailpit (or `docker run -p 1025:1025 -p 8025:8025 axllent/mailpit`)
# Terminal 3: frontend as usual
cd frontend && npm run dev

# Terminal 4
cd e2e && E2E_SKIP_DOCKER=1 npm test
```

`E2E_SKIP_DOCKER=1` skips the `docker compose up`/`down` calls and seeds
fixtures directly via the local backend venv instead of `docker compose
exec` — everything else is identical.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `E2E_BASE_URL` | `http://localhost:5173` | Frontend URL Playwright drives |
| `E2E_MAILPIT_URL` | `http://localhost:8025` | Mailpit's HTTP API, used to fetch the password-reset email |
| `E2E_ADMIN_EMAIL` / `E2E_ADMIN_PASSWORD` | `e2e-admin@example.com` / `e2e-admin-password123` | Seeded admin account credentials |
| `E2E_SKIP_DOCKER` | unset | Skip Docker orchestration (see above) |
| `E2E_KEEP_STACK` | unset | Skip `docker compose down` in global-teardown, for debugging a failure |

The backend rate-limits auth endpoints per client IP, and this whole suite
comes from one IP. `global-setup.js` relaxes those limits for the Docker
stack it brings up; in `E2E_SKIP_DOCKER` mode you start the backend
yourself, so pass the `AUTH_*_MAX_REQUESTS` overrides shown above or the
signup-heavy specs will hit `429`.

## What this does NOT cover

- Live-camera streaming (no camera hardware in CI) — covered by manual
  browser QA during development instead (see root README).
- A genuine ML-detected fall (the upload fixture is a person-free
  synthetic clip, so the workflow test asserts 0 persons/0 falls, which is
  what actually happens — it does not fabricate a fall). The
  acknowledge-alert test uses a seeded fixture event instead of a real
  detection, for a deterministic assertion.
