import { beforeEach, describe, expect, it } from 'vitest'
import { useAuthStore } from './authStore'

describe('authStore', () => {
  beforeEach(() => {
    localStorage.clear()
    useAuthStore.setState({ token: null, user: null })
  })

  it('starts logged out', () => {
    expect(useAuthStore.getState().token).toBeNull()
    expect(useAuthStore.getState().user).toBeNull()
  })

  it('login stores token and user', () => {
    useAuthStore.getState().login('abc123', { id: 1, email: 'a@example.com', role: 'viewer' })
    expect(useAuthStore.getState().token).toBe('abc123')
    expect(useAuthStore.getState().user.email).toBe('a@example.com')
  })

  it('logout clears token and user', () => {
    useAuthStore.getState().login('abc123', { id: 1, email: 'a@example.com', role: 'admin' })
    useAuthStore.getState().logout()
    expect(useAuthStore.getState().token).toBeNull()
    expect(useAuthStore.getState().user).toBeNull()
  })

  it.each([
    ['admin', true, true],
    ['operator', false, true],
    ['viewer', false, false],
  ])('role=%s -> isAdmin=%s, isOperator=%s', (role, expectedAdmin, expectedOperator) => {
    useAuthStore.getState().login('tok', { id: 1, email: 'a@example.com', role })
    expect(useAuthStore.getState().isAdmin()).toBe(expectedAdmin)
    expect(useAuthStore.getState().isOperator()).toBe(expectedOperator)
  })

  it('hasRole checks against arbitrary role lists', () => {
    useAuthStore.getState().login('tok', { id: 1, email: 'a@example.com', role: 'operator' })
    expect(useAuthStore.getState().hasRole('admin', 'operator')).toBe(true)
    expect(useAuthStore.getState().hasRole('admin')).toBe(false)
  })
})
