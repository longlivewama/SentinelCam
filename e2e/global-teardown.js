import { execSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

export default async function globalTeardown() {
  if (process.env.E2E_SKIP_DOCKER || process.env.E2E_KEEP_STACK) {
    return
  }
  console.log('[e2e global-teardown] $ docker compose down')
  try {
    execSync('docker compose down', { stdio: 'inherit', cwd: REPO_ROOT })
  } catch (err) {
    console.error('[e2e global-teardown] docker compose down failed:', err.message)
  }
}
