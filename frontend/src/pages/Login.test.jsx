import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Login from './Login'
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

describe('Login page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAuthStore.setState({ token: null, user: null })
  })

  function renderLogin() {
    return render(
      <MemoryRouter>
        <Login />
      </MemoryRouter>,
    )
  }

  it('renders email and password fields', () => {
    renderLogin()
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument()
  })

  it('logs in successfully and redirects to the dashboard', async () => {
    const user = userEvent.setup()
    apiClient.post.mockResolvedValueOnce({
      data: { access_token: 'tok123', user: { id: 1, email: 'a@example.com', role: 'viewer' } },
    })

    renderLogin()
    await user.type(screen.getByLabelText(/email/i), 'a@example.com')
    await user.type(screen.getByLabelText(/password/i), 'password123')
    await user.click(screen.getByRole('button', { name: /sign in/i }))

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith('/api/auth/login', {
        email: 'a@example.com',
        password: 'password123',
      })
    })
    await waitFor(() => expect(useAuthStore.getState().token).toBe('tok123'))
    expect(mockNavigate).toHaveBeenCalledWith('/dashboard', { replace: true })
  })

  it('shows an error message on invalid credentials', async () => {
    const user = userEvent.setup()
    apiClient.post.mockRejectedValueOnce({ response: { status: 401 } })

    renderLogin()
    await user.type(screen.getByLabelText(/email/i), 'a@example.com')
    await user.type(screen.getByLabelText(/password/i), 'wrongpassword')
    await user.click(screen.getByRole('button', { name: /sign in/i }))

    expect(await screen.findByText(/invalid email or password/i)).toBeInTheDocument()
    expect(useAuthStore.getState().token).toBeNull()
  })

  it('links to signup and forgot-password', () => {
    renderLogin()
    expect(screen.getByText(/forgot password/i)).toBeInTheDocument()
    expect(screen.getByText(/create an account/i)).toBeInTheDocument()
  })
})
