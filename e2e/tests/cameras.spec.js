import { expect, test } from '@playwright/test'
import { ADMIN_EMAIL, ADMIN_PASSWORD } from '../fixtures.js'
import { loginAs } from '../helpers.js'

test.describe('Camera workflow', () => {
  test.beforeEach(async ({ page }) => {
    await loginAs(page, ADMIN_EMAIL, ADMIN_PASSWORD)
  })

  test('create, edit, and delete a camera', async ({ page }) => {
    const name = `E2E Camera ${Date.now()}`

    await page.goto('/cameras')
    await page.getByRole('button', { name: '+ Add Camera' }).click()
    await page.getByPlaceholder('Front Entrance').fill(name)
    await page.getByPlaceholder('rtsp://192.168.1.20/stream').fill('0')
    await page.getByPlaceholder('Lobby').fill('E2E Wing')
    await page.getByRole('button', { name: 'Create Camera' }).click()

    // The new camera shows up as a tile on the list page; click into its
    // detail page for the edit/delete actions.
    await expect(page.getByRole('heading', { name })).toBeVisible()
    await page.getByRole('heading', { name }).click()
    await expect(page).toHaveURL(/\/cameras\/\d+$/)

    await page.getByRole('button', { name: 'Edit' }).click()
    await page.getByPlaceholder('Lobby').fill('E2E Wing Updated')
    await page.getByRole('button', { name: 'Save Changes' }).click()
    await expect(page.getByText('E2E Wing Updated')).toBeVisible()

    page.once('dialog', (dialog) => dialog.accept())
    await page.getByRole('button', { name: 'Delete' }).click()
    await expect(page).toHaveURL(/\/cameras$/)
    await expect(page.getByRole('heading', { name })).not.toBeVisible()
  })

  test('viewer cannot see the Add Camera button', async ({ page }) => {
    // Sign up a fresh viewer account (default role for self-signup) and
    // confirm the management action is hidden for them.
    const email = `viewer-cam-${Date.now()}@example.com`
    await page.goto('/signup')
    await page.getByLabel('Full name').fill('Viewer Cam Test')
    await page.getByLabel('Email').fill(email)
    await page.getByLabel('Password', { exact: true }).fill('password123')
    await page.getByLabel('Confirm password').fill('password123')
    await page.getByRole('button', { name: 'Create Account' }).click()
    await expect(page).toHaveURL(/\/dashboard$/)

    await page.goto('/cameras')
    await expect(page.getByRole('button', { name: '+ Add Camera' })).not.toBeVisible()
  })
})
