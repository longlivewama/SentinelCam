import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ResetPassword from './ResetPassword'
import apiClient from '../api/client'

vi.mock('../api/client', () => ({
  default: { post: vi.fn(), get: vi.fn() },
  API_URL: 'http://localhost:8000',
}))

function renderAt(path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/reset-password" element={<ResetPassword />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('ResetPassword page', () => {
  beforeEach(() => vi.clearAllMocks())

  it('shows an error state when no token is present in the URL', () => {
    renderAt('/reset-password')
    expect(screen.getByText(/missing a reset token/i)).toBeInTheDocument()
  })

  it('rejects mismatched passwords before calling the API', async () => {
    const user = userEvent.setup()
    renderAt('/reset-password?token=abc123')

    await user.type(screen.getByLabelText(/^new password$/i), 'password123')
    await user.type(screen.getByLabelText(/confirm new password/i), 'different123')
    await user.click(screen.getByRole('button', { name: /reset password/i }))

    expect(await screen.findByText(/do not match/i)).toBeInTheDocument()
    expect(apiClient.post).not.toHaveBeenCalled()
  })

  it('submits the token and new password, showing success', async () => {
    const user = userEvent.setup()
    apiClient.post.mockResolvedValueOnce({ data: { message: 'ok' } })
    renderAt('/reset-password?token=abc123')

    await user.type(screen.getByLabelText(/^new password$/i), 'password123')
    await user.type(screen.getByLabelText(/confirm new password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /reset password/i }))

    expect(await screen.findByText(/reset successfully/i)).toBeInTheDocument()
    expect(apiClient.post).toHaveBeenCalledWith('/api/auth/reset-password', {
      token: 'abc123',
      new_password: 'password123',
    })
  })

  it('shows the server error message for an invalid/expired token', async () => {
    const user = userEvent.setup()
    apiClient.post.mockRejectedValueOnce({ response: { data: { detail: 'This link has expired.' } } })
    renderAt('/reset-password?token=expired')

    await user.type(screen.getByLabelText(/^new password$/i), 'password123')
    await user.type(screen.getByLabelText(/confirm new password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /reset password/i }))

    expect(await screen.findByText(/this link has expired/i)).toBeInTheDocument()
  })
})
