import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ForgotPassword from './ForgotPassword'
import apiClient from '../api/client'

vi.mock('../api/client', () => ({
  default: { post: vi.fn(), get: vi.fn() },
  API_URL: 'http://localhost:8000',
}))

describe('ForgotPassword page', () => {
  beforeEach(() => vi.clearAllMocks())

  it('shows a generic confirmation message after submit, regardless of whether the email exists', async () => {
    const user = userEvent.setup()
    apiClient.post.mockResolvedValueOnce({ data: { message: 'ok' } })

    render(
      <MemoryRouter>
        <ForgotPassword />
      </MemoryRouter>,
    )
    await user.type(screen.getByLabelText(/email/i), 'someone@example.com')
    await user.click(screen.getByRole('button', { name: /send reset link/i }))

    expect(await screen.findByText(/if an account with that email exists/i)).toBeInTheDocument()
    expect(apiClient.post).toHaveBeenCalledWith('/api/auth/forgot-password', { email: 'someone@example.com' })
  })
})
