import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Signup from './Signup'
import apiClient from '../api/client'
import { useAuthStore } from '../store/authStore'

vi.mock('../api/client', () => ({
  default: { post: vi.fn(), get: vi.fn() },
  API_URL: 'http://localhost:8000',
}))

const mockNavigate = vi.fn()
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return { ...actual, useNavigate: () => mockNavigate }
})

describe('Signup page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAuthStore.setState({ token: null, user: null })
  })

  function renderSignup() {
    return render(
      <MemoryRouter>
        <Signup />
      </MemoryRouter>,
    )
  }

  it('rejects a password shorter than 8 characters client-side', async () => {
    const user = userEvent.setup()
    renderSignup()
    await user.type(screen.getByLabelText(/full name/i), 'Jane Doe')
    await user.type(screen.getByLabelText(/email/i), 'jane@example.com')
    await user.type(screen.getByLabelText(/^password$/i), 'short')
    await user.type(screen.getByLabelText(/confirm password/i), 'short')
    await user.click(screen.getByRole('button', { name: /create account/i }))

    expect(await screen.findByText(/at least 8 characters/i)).toBeInTheDocument()
    expect(apiClient.post).not.toHaveBeenCalled()
  })

  it('rejects mismatched passwords', async () => {
    const user = userEvent.setup()
    renderSignup()
    await user.type(screen.getByLabelText(/full name/i), 'Jane Doe')
    await user.type(screen.getByLabelText(/email/i), 'jane@example.com')
    await user.type(screen.getByLabelText(/^password$/i), 'password123')
    await user.type(screen.getByLabelText(/confirm password/i), 'password456')
    await user.click(screen.getByRole('button', { name: /create account/i }))

    expect(await screen.findByText(/do not match/i)).toBeInTheDocument()
    expect(apiClient.post).not.toHaveBeenCalled()
  })

  it('creates an account and logs in on success', async () => {
    const user = userEvent.setup()
    apiClient.post.mockResolvedValueOnce({
      data: { access_token: 'newtok', user: { id: 2, email: 'jane@example.com', role: 'viewer' } },
    })

    renderSignup()
    await user.type(screen.getByLabelText(/full name/i), 'Jane Doe')
    await user.type(screen.getByLabelText(/email/i), 'jane@example.com')
    await user.type(screen.getByLabelText(/^password$/i), 'password123')
    await user.type(screen.getByLabelText(/confirm password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /create account/i }))

    await waitFor(() => expect(apiClient.post).toHaveBeenCalledWith('/api/auth/signup', {
      email: 'jane@example.com',
      password: 'password123',
      full_name: 'Jane Doe',
    }))
    await waitFor(() => expect(useAuthStore.getState().token).toBe('newtok'))
    expect(mockNavigate).toHaveBeenCalledWith('/dashboard', { replace: true })
  })

  it('shows a friendly error for a duplicate email', async () => {
    const user = userEvent.setup()
    apiClient.post.mockRejectedValueOnce({ response: { status: 400, data: { detail: 'Email already in use' } } })

    renderSignup()
    await user.type(screen.getByLabelText(/full name/i), 'Jane Doe')
    await user.type(screen.getByLabelText(/email/i), 'jane@example.com')
    await user.type(screen.getByLabelText(/^password$/i), 'password123')
    await user.type(screen.getByLabelText(/confirm password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /create account/i }))

    expect(await screen.findByText(/already in use/i)).toBeInTheDocument()
  })
})
