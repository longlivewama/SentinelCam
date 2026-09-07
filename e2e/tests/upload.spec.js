import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test } from '@playwright/test'
import { ADMIN_EMAIL, ADMIN_PASSWORD } from '../fixtures.js'
import { loginAs } from '../helpers.js'

const FIXTURE_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'fixtures')
const FIXTURE_VIDEO = path.join(FIXTURE_DIR, 'tiny-test-video.mp4')
const FIXTURE_NOT_A_VIDEO = path.join(FIXTURE_DIR, 'not-a-video.txt')

test.describe('Video upload and analysis', () => {
  test.beforeEach(async ({ page }) => {
    await loginAs(page, ADMIN_EMAIL, ADMIN_PASSWORD)
  })

  // Real CPU decode + pose inference over the fixture, measured on a dev
  // laptop: ~9s with the machine otherwise idle, but 45-75s when running
  // as part of the full suite, where Playwright's own video/trace capture
  // competes for the same cores. The budget below covers the slow end -
  // the analysis itself always completes, it just isn't fast.
  test('upload a video, watch it process, and view its results', async ({ page }) => {
    test.setTimeout(150_000)
    await page.goto('/upload')

    await page.locator('input[type="file"]').setInputFiles(FIXTURE_VIDEO)

    await expect(page.getByText('tiny-test-video.mp4')).toBeVisible()

    // The fixture video has no people in it, so a real run through the
    // detection pipeline should genuinely finish at 0 persons / 0 falls -
    // this isn't a fabricated expectation, it's what actually happens.
    // `.sc-card` is the single outer wrapper per upload row (see
    // VideoUpload.jsx), so this - unlike a bare `div` filter - can't
    // accidentally match one of the several nested divs inside it.
    const row = page.locator('.sc-card').filter({ hasText: 'tiny-test-video.mp4' })
    await expect(row).toContainText('completed', { timeout: 100_000, ignoreCase: true })
    await expect(row).toContainText('0 falls')
    await expect(row).toContainText('0 persons')

    await row.getByRole('button', { name: 'View' }).click()
    await expect(page.getByText('Persons detected')).toBeVisible()
    await expect(page.locator('video').first()).toBeVisible()
    await expect(page.getByText('No fall events detected.')).toBeVisible()
  })

  test('deleting an upload removes it from the list', async ({ page }) => {
    await page.goto('/upload')

    // Other tests in this file may have left their own upload of the same
    // fixture behind, so count rows rather than assuming there's exactly
    // one "tiny-test-video.mp4" on the page. The list is fetched after
    // mount, so wait for that fetch to land first - counting straight
    // after goto() reads 0 every time and makes the assertions below
    // depend on how fast the request happened to be.
    await expect(page.getByText('Loading uploads…')).toHaveCount(0)
    const matchingRows = page.locator('.sc-card').filter({ hasText: 'tiny-test-video.mp4' })
    const countBefore = await matchingRows.count()

    await page.locator('input[type="file"]').setInputFiles(FIXTURE_VIDEO)
    await expect(matchingRows).toHaveCount(countBefore + 1)

    // Uploads are listed newest-first, so the one just added is first.
    page.once('dialog', (dialog) => dialog.accept())
    await matchingRows.first().getByRole('button', { name: 'Delete' }).click()

    await expect(page.getByText('Video deleted.')).toBeVisible()
    await expect(matchingRows).toHaveCount(countBefore)
  })

  test('rejects an unsupported file without sending it to the server', async ({ page }) => {
    await page.goto('/upload')
    await expect(page.getByText('Loading uploads…')).toHaveCount(0)

    // Scope to actual upload rows: toasts are `.sc-card` too, and the
    // rejection toast quotes the filename, so a bare text filter would
    // match the toast rather than a row.
    const uploadRows = page
      .locator('.sc-card')
      .filter({ has: page.getByRole('button', { name: 'View' }) })
    expect(await uploadRows.filter({ hasText: 'not-a-video.txt' }).count()).toBe(0)

    await page.locator('input[type="file"]').setInputFiles(FIXTURE_NOT_A_VIDEO)

    await expect(page.getByText(/supported video formats are/i)).toBeVisible()
    // Rejected client-side, so no upload row is ever created for it.
    await expect(uploadRows.filter({ hasText: 'not-a-video.txt' })).toHaveCount(0)
  })

  test('the dashboard surfaces the most recent analysis', async ({ page }) => {
    // The first test in this file leaves a completed analysis behind.
    await page.goto('/dashboard')

    const analyses = page.locator('.sc-card').filter({ hasText: 'Recent analyses' })
    await expect(analyses).toContainText('tiny-test-video.mp4')
    await expect(analyses).toContainText('Latest')
  })
})
