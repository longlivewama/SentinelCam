import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it } from 'vitest'
import ProtectedRoute from './ProtectedRoute'
import AdminRoute from './AdminRoute'
import OperatorRoute from './OperatorRoute'
import { useAuthStore } from '../store/authStore'

function renderGuarded(GuardComponent, { protectedPath = '/protected' } = {}) {
  return render(
    <MemoryRouter initialEntries={[protectedPath]}>
      <Routes>
        <Route path="/login" element={<div>Login Page</div>} />
        <Route path="/dashboard" element={<div>Dashboard Page</div>} />
        <Route path="/cameras" element={<div>Cameras Page</div>} />
        <Route element={<GuardComponent />}>
          <Route path={protectedPath} element={<div>Protected Content</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  )
}

describe('ProtectedRoute', () => {
  beforeEach(() => useAuthStore.setState({ token: null, user: null }))

  it('redirects to /login when logged out', () => {
    renderGuarded(ProtectedRoute)
    expect(screen.getByText('Login Page')).toBeInTheDocument()
  })

  it('renders the protected content when logged in', () => {
    useAuthStore.setState({ token: 'tok', user: { id: 1, role: 'viewer' } })
    renderGuarded(ProtectedRoute)
    expect(screen.getByText('Protected Content')).toBeInTheDocument()
  })
})

describe('OperatorRoute', () => {
  beforeEach(() => useAuthStore.setState({ token: null, user: null }))

  it('redirects viewers to /dashboard', () => {
    useAuthStore.setState({ token: 'tok', user: { id: 1, role: 'viewer' } })
    renderGuarded(OperatorRoute)
    expect(screen.getByText('Dashboard Page')).toBeInTheDocument()
  })

  it('allows operators through', () => {
    useAuthStore.setState({ token: 'tok', user: { id: 1, role: 'operator' } })
    renderGuarded(OperatorRoute)
    expect(screen.getByText('Protected Content')).toBeInTheDocument()
  })

  it('allows admins through', () => {
    useAuthStore.setState({ token: 'tok', user: { id: 1, role: 'admin' } })
    renderGuarded(OperatorRoute)
    expect(screen.getByText('Protected Content')).toBeInTheDocument()
  })
})

describe('AdminRoute', () => {
  beforeEach(() => useAuthStore.setState({ token: null, user: null }))

  it('redirects non-admins to /dashboard', () => {
    useAuthStore.setState({ token: 'tok', user: { id: 1, role: 'operator' } })
    renderGuarded(AdminRoute)
    expect(screen.getByText('Dashboard Page')).toBeInTheDocument()
  })

  it('allows admins through', () => {
    useAuthStore.setState({ token: 'tok', user: { id: 1, role: 'admin' } })
    renderGuarded(AdminRoute)
    expect(screen.getByText('Protected Content')).toBeInTheDocument()
  })
})
