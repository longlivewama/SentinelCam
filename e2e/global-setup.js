import { execSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { ADMIN_EMAIL, ADMIN_PASSWORD } from './fixtures.js'

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

const RELAXED_AUTH_RATE_LIMITS = {
  AUTH_SIGNUP_MAX_REQUESTS: '1000',
  AUTH_LOGIN_MAX_REQUESTS: '1000',
  AUTH_FORGOT_PASSWORD_MAX_REQUESTS: '1000',
  AUTH_RESET_PASSWORD_MAX_REQUESTS: '1000',
}

function run(cmd, opts = {}) {
  console.log(`[e2e global-setup] $ ${cmd}`)
  execSync(cmd, { stdio: 'inherit', cwd: REPO_ROOT, ...opts })
}

export default async function globalSetup() {
  if (process.env.E2E_SKIP_DOCKER) {
    // Local-iteration mode: backend/frontend are already running outside
    // Docker (e.g. `uvicorn --reload` + `npm run dev`), pointed at
    // whatever DATABASE_URL is set in the shell running this. Seed
    // directly via the local venv instead of `docker compose exec`.
    console.log('[e2e global-setup] E2E_SKIP_DOCKER set - assuming backend/frontend are already running locally.')
    run(
      'venv/bin/python scripts/seed_e2e_fixtures.py',
      { cwd: path.join(REPO_ROOT, 'backend'), env: { ...process.env, E2E_ADMIN_EMAIL: ADMIN_EMAIL, E2E_ADMIN_PASSWORD: ADMIN_PASSWORD } },
    )
    return
  }

  // The backend rate-limits auth endpoints per client IP, and every
  // request in this suite arrives from one IP - so the production limits
  // (5 signups/60s) trip partway through a run. Relax them for the stack
  // this suite drives; docker-compose.yml keeps the production values as
  // its defaults, so this opt-out is confined to the e2e environment.
  const env = { ...process.env, ...RELAXED_AUTH_RATE_LIMITS }

  // --wait relies on the healthchecks defined in docker-compose.yml
  // (postgres, backend, frontend) to know when the stack is actually
  // ready, not just "containers started".
  run('docker compose up -d --build --wait', { env })

  run(
    `docker compose exec -T -e E2E_ADMIN_EMAIL=${ADMIN_EMAIL} -e E2E_ADMIN_PASSWORD=${ADMIN_PASSWORD} ` +
      'backend python scripts/seed_e2e_fixtures.py',
  )
}
