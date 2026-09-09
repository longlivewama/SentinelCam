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

  // The bug this guards against rendered a *visible* <video> that decoded
  // nothing: the clip was MPEG-4 Part 2, which no browser supports, so the
  // element mounted, the range requests returned 206, and the user saw a black
  // rectangle. Asserting the player is visible - as the test above does - passes
  // happily in that state, which is how it shipped. Only decoding proves it.
  //
  // The seeded clip is written by the same `open_writer` codec ladder that
  // writes production clips (see backend/scripts/seed_e2e_fixtures.py), so this
  // covers the real encoder path without needing a committed video fixture or a
  // model that has to detect something on cue.
  test('the seeded clip actually decodes in the browser, not just mounts', async ({ page }) => {
    await page.goto('/recordings')

    const row = page.getByRole('row').filter({ hasText: 'E2E Test Camera' }).first()
    await row.getByRole('button', { name: 'View' }).click()
    const video = page.locator('video')
    await expect(video).toBeVisible()

    const playback = await video.evaluate(
      (el) =>
        new Promise((resolve) => {
          const report = () => resolve({
            readyState: el.readyState,
            width: el.videoWidth,
            height: el.videoHeight,
            error: el.error ? el.error.message : null,
          })
          if (el.readyState >= 2 || el.error) return report()
          const timer = setTimeout(report, 15000)
          el.onloadeddata = () => { clearTimeout(timer); report() }
          el.onerror = () => { clearTimeout(timer); report() }
        }),
    )

    expect(playback.error, `the browser refused the clip: ${playback.error}`).toBeNull()
    // HAVE_CURRENT_DATA or better - the decoder produced a frame.
    expect(playback.readyState).toBeGreaterThanOrEqual(2)
    // A decoded frame has real dimensions; an undecodable one reports 0x0.
    expect(playback.width).toBeGreaterThan(0)
    expect(playback.height).toBeGreaterThan(0)
  })

  test('the clip is served as the media type it actually is', async ({ page, request }) => {
    // The other half of the same bug: serving WebM bytes labelled video/mp4.
    // With X-Content-Type-Options: nosniff - which this API sets - a mismatched
    // Content-Type is fatal rather than cosmetic, so the header has to follow
    // the file's real container.
    await page.goto('/recordings')
    const row = page.getByRole('row').filter({ hasText: 'E2E Test Camera' }).first()
    await row.getByRole('button', { name: 'View' }).click()

    const src = await page.locator('video').getAttribute('src')
    const response = await request.get(src, { headers: { Range: 'bytes=0-63' } })

    expect(response.status()).toBe(206)
    expect(response.headers()['content-range']).toMatch(/^bytes 0-63\/\d+$/)
    expect(response.headers()['accept-ranges']).toBe('bytes')

    const body = await response.body()
    const contentType = response.headers()['content-type']
    if (contentType.startsWith('video/webm')) {
      // EBML magic - what a real WebM starts with.
      expect([...body.subarray(0, 4)]).toEqual([0x1a, 0x45, 0xdf, 0xa3])
    } else {
      expect(contentType).toMatch(/^video\/mp4/)
      expect(body.subarray(4, 8).toString('latin1')).toBe('ftyp')
    }
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
