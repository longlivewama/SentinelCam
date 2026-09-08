/**
 * Resolves the URLs this suite talks to.
 *
 * The stack's host ports are configurable (FRONTEND_PORT, MAILPIT_WEB_PORT
 * in the repo-root .env that docker compose reads), so hardcoding 5173/8025
 * here silently breaks the whole suite with ERR_CONNECTION_REFUSED the
 * moment anyone remaps a port to avoid a local collision. Read the same
 * .env docker compose does, so the suite follows the stack it just started.
 *
 * Precedence: explicit E2E_* override > shell environment > repo-root .env
 * > the compose default.
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

function readRootEnv() {
  const envPath = path.join(REPO_ROOT, '.env')
  if (!fs.existsSync(envPath)) return {}
  const values = {}
  for (const line of fs.readFileSync(envPath, 'utf8').split('\n')) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    const eq = trimmed.indexOf('=')
    if (eq === -1) continue
    values[trimmed.slice(0, eq).trim()] = trimmed.slice(eq + 1).trim()
  }
  return values
}

const rootEnv = readRootEnv()

function port(name, fallback) {
  return process.env[name] || rootEnv[name] || fallback
}

export const BASE_URL = process.env.E2E_BASE_URL || `http://localhost:${port('FRONTEND_PORT', '5173')}`
export const MAILPIT_URL = process.env.E2E_MAILPIT_URL || `http://localhost:${port('MAILPIT_WEB_PORT', '8025')}`
