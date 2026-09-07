// Shared constants between global-setup.js and the test specs. Must match
// the defaults in backend/scripts/seed_e2e_fixtures.py.
export const ADMIN_EMAIL = process.env.E2E_ADMIN_EMAIL || 'e2e-admin@example.com'
export const ADMIN_PASSWORD = process.env.E2E_ADMIN_PASSWORD || 'e2e-admin-password123'

export const MAILPIT_URL = process.env.E2E_MAILPIT_URL || 'http://localhost:8025'

/** A unique-per-run email so signup/login tests never collide with each
 * other or with a previous run's leftover data. */
export function uniqueEmail(label) {
  return `${label}-${Date.now()}-${Math.floor(Math.random() * 1e6)}@example.com`
}
