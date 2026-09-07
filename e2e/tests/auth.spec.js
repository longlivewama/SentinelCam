import { expect, test } from '@playwright/test'
import { MAILPIT_URL, uniqueEmail } from '../fixtures.js'

test.describe('Authentication', () => {
  test('signup creates an account and logs straight in', async ({ page }) => {
    const email = uniqueEmail('signup')

    await page.goto('/signup')
    await page.getByLabel('Full name').fill('E2E Signup User')
    await page.getByLabel('Email').fill(email)
    await page.getByLabel('Password', { exact: true }).fill('password123')
    await page.getByLabel('Confirm password').fill('password123')
    await page.getByRole('button', { name: 'Create Account' }).click()

    await expect(page).toHaveURL(/\/dashboard$/)
    await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible()
  })

  test('signup rejects a duplicate email with a clear error', async ({ page }) => {
    const email = uniqueEmail('dup')

    await page.goto('/signup')
    await page.getByLabel('Full name').fill('First')
    await page.getByLabel('Email').fill(email)
    await page.getByLabel('Password', { exact: true }).fill('password123')
    await page.getByLabel('Confirm password').fill('password123')
    await page.getByRole('button', { name: 'Create Account' }).click()
    await expect(page).toHaveURL(/\/dashboard$/)

    await page.getByRole('button', { name: 'Logout' }).click()
    await expect(page).toHaveURL(/\/login$/)

    await page.goto('/signup')
    await page.getByLabel('Full name').fill('Second')
    await page.getByLabel('Email').fill(email)
    await page.getByLabel('Password', { exact: true }).fill('password456')
    await page.getByLabel('Confirm password').fill('password456')
    await page.getByRole('button', { name: 'Create Account' }).click()

    await expect(page.getByText(/already in use/i)).toBeVisible()
  })

  test('login, logout, and login again', async ({ page }) => {
    const email = uniqueEmail('loginflow')

    await page.goto('/signup')
    await page.getByLabel('Full name').fill('Login Flow User')
    await page.getByLabel('Email').fill(email)
    await page.getByLabel('Password', { exact: true }).fill('password123')
    await page.getByLabel('Confirm password').fill('password123')
    await page.getByRole('button', { name: 'Create Account' }).click()
    await expect(page).toHaveURL(/\/dashboard$/)

    await page.getByRole('button', { name: 'Logout' }).click()
    await expect(page).toHaveURL(/\/login$/)

    // A protected page must bounce back to /login while logged out.
    await page.goto('/dashboard')
    await expect(page).toHaveURL(/\/login$/)

    await page.getByLabel('Email').fill(email)
    await page.getByLabel('Password').fill('password123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    await expect(page).toHaveURL(/\/dashboard$/)
  })

  test('login shows an error for wrong credentials', async ({ page }) => {
    await page.goto('/login')
    await page.getByLabel('Email').fill('nobody-at-all@example.com')
    await page.getByLabel('Password').fill('wrongpassword')
    await page.getByRole('button', { name: 'Sign In' }).click()
    await expect(page.getByText(/invalid email or password/i)).toBeVisible()
    await expect(page).toHaveURL(/\/login$/)
  })

  test('forgot password -> real email via Mailpit -> reset -> login with new password', async ({ page, request }) => {
    const email = uniqueEmail('resetflow')

    await page.goto('/signup')
    await page.getByLabel('Full name').fill('Reset Flow User')
    await page.getByLabel('Email').fill(email)
    await page.getByLabel('Password', { exact: true }).fill('originalpass123')
    await page.getByLabel('Confirm password').fill('originalpass123')
    await page.getByRole('button', { name: 'Create Account' }).click()
    await expect(page).toHaveURL(/\/dashboard$/)
    await page.getByRole('button', { name: 'Logout' }).click()

    await page.goto('/forgot-password')
    await page.getByLabel('Email').fill(email)
    await page.getByRole('button', { name: 'Send Reset Link' }).click()
    await expect(page.getByText(/if an account with that email exists/i)).toBeVisible()

    const resetUrl = await waitForResetLink(request, email)

    await page.goto(resetUrl)
    await page.getByLabel('New password', { exact: true }).fill('brandnewpass456')
    await page.getByLabel('Confirm new password').fill('brandnewpass456')
    await page.getByRole('button', { name: 'Reset Password' }).click()
    await expect(page.getByText(/reset successfully/i)).toBeVisible()

    await page.goto('/login')
    await page.getByLabel('Email').fill(email)
    await page.getByLabel('Password').fill('originalpass123')
    await page.getByRole('button', { name: 'Sign In' }).click()
    await expect(page.getByText(/invalid email or password/i)).toBeVisible()

    await page.getByLabel('Email').fill(email)
    await page.getByLabel('Password').fill('brandnewpass456')
    await page.getByRole('button', { name: 'Sign In' }).click()
    await expect(page).toHaveURL(/\/dashboard$/)
  })

  test('reset-password page shows an error for an invalid/expired token', async ({ page }) => {
    await page.goto('/reset-password?token=this-token-does-not-exist')
    await page.getByLabel('New password', { exact: true }).fill('somepassword123')
    await page.getByLabel('Confirm new password').fill('somepassword123')
    await page.getByRole('button', { name: 'Reset Password' }).click()
    await expect(page.getByText(/invalid or has expired/i)).toBeVisible()
  })
})

/** Polls Mailpit's REST API for the password-reset email sent to `email`
 * and extracts the reset link's path+query from its body. */
async function waitForResetLink(request, email) {
  const deadline = Date.now() + 15_000
  while (Date.now() < deadline) {
    const res = await request.get(`${MAILPIT_URL}/api/v1/search`, {
      params: { query: `to:${email}` },
    })
    if (res.ok()) {
      const data = await res.json()
      const message = data.messages?.[0]
      if (message) {
        const detail = await (await request.get(`${MAILPIT_URL}/api/v1/message/${message.ID}`)).json()
        const match = detail.Text.match(/https?:\/\/[^\s]+\/reset-password\?token=[^\s]+/)
        if (match) {
          return match[0].replace(/^https?:\/\/[^/]+/, '')
        }
      }
    }
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error(`Timed out waiting for a password-reset email to ${email} in Mailpit`)
}
