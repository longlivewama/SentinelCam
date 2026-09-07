import { expect, test } from '@playwright/test'
import { ADMIN_EMAIL, ADMIN_PASSWORD } from '../fixtures.js'
import { loginAs } from '../helpers.js'

test.describe('Alerts and recordings (seeded fixture data)', () => {
  test.beforeEach(async ({ page }) => {
    await loginAs(page, ADMIN_EMAIL, ADMIN_PASSWORD)
  })

  test('seeded fall alert can be viewed and acknowledged', async ({ page }) => {
    await page.goto('/alerts')

    const row = page.getByRole('row').filter({ hasText: 'fall' }).filter({ hasText: 'NEW' }).first()
    await expect(row).toBeVisible()

    await row.getByRole('button', { name: 'Acknowledge' }).click()
    await expect(page.getByText('Alert acknowledged.')).toBeVisible()

    const acknowledgedRow = page.getByRole('row').filter({ hasText: 'fall' }).filter({ hasText: 'ACKNOWLEDGED' })
    await expect(acknowledgedRow.first()).toBeVisible()
  })

  test('seeded recording appears and its video player opens', async ({ page }) => {
    await page.goto('/recordings')

    const row = page.getByRole('row').filter({ hasText: 'E2E Test Camera' }).first()
    await expect(row).toBeVisible()

    await row.getByRole('button', { name: 'View' }).click()
    await expect(page.locator('video')).toBeVisible()
  })

  test('recordings can be filtered by camera', async ({ page }) => {
    await page.goto('/recordings')
    await page.getByLabel('Camera').selectOption({ label: 'E2E Test Camera' })
    await expect(page.getByRole('row').filter({ hasText: 'E2E Test Camera' }).first()).toBeVisible()
  })

  test('dashboard reflects the seeded alert count', async ({ page }) => {
    await page.goto('/dashboard')
    await expect(page.getByText(/of \d+ total alerts/)).toBeVisible()
  })
})
