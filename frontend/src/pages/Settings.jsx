import { useState } from 'react'
import apiClient from '../api/client'
import { useAuthStore } from '../store/authStore'

export default function Settings() {
  const user = useAuthStore((s) => s.user)
  const token = useAuthStore((s) => s.token)
  const login = useAuthStore((s) => s.login)

  const [emailForm, setEmailForm] = useState({ new_email: '', current_password: '' })
  const [emailStatus, setEmailStatus] = useState({ type: '', message: '' })
  const [emailSubmitting, setEmailSubmitting] = useState(false)

  const [passwordForm, setPasswordForm] = useState({
    current_password: '',
    new_password: '',
    confirm_password: '',
  })
  const [passwordStatus, setPasswordStatus] = useState({ type: '', message: '' })
  const [passwordSubmitting, setPasswordSubmitting] = useState(false)

  const handleEmailSubmit = async (e) => {
    e.preventDefault()
    setEmailStatus({ type: '', message: '' })
    setEmailSubmitting(true)
    try {
      const { data } = await apiClient.put('/api/auth/settings', {
        current_password: emailForm.current_password,
        new_email: emailForm.new_email,
      })
      login(token, data)
      setEmailStatus({ type: 'success', message: 'Email updated successfully.' })
      setEmailForm({ new_email: '', current_password: '' })
    } catch (err) {
      setEmailStatus({
        type: 'error',
        message: err?.response?.data?.detail || 'Failed to update email.',
      })
    } finally {
      setEmailSubmitting(false)
    }
  }

  const handlePasswordSubmit = async (e) => {
    e.preventDefault()
    setPasswordStatus({ type: '', message: '' })

    if (passwordForm.new_password !== passwordForm.confirm_password) {
      setPasswordStatus({ type: 'error', message: 'New passwords do not match.' })
      return
    }

    setPasswordSubmitting(true)
    try {
      const { data } = await apiClient.put('/api/auth/settings', {
        current_password: passwordForm.current_password,
        new_password: passwordForm.new_password,
      })
      login(token, data)
      setPasswordStatus({ type: 'success', message: 'Password updated successfully.' })
      setPasswordForm({ current_password: '', new_password: '', confirm_password: '' })
    } catch (err) {
      setPasswordStatus({
        type: 'error',
        message: err?.response?.data?.detail || 'Failed to update password.',
      })
    } finally {
      setPasswordSubmitting(false)
    }
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-8 sm:px-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-100">Settings</h1>
        <p className="mt-1 text-sm text-slate-400">
          Signed in as {user?.email}
        </p>
      </div>

      <div className="sc-card mb-6 p-6">
        <h2 className="mb-4 text-lg font-semibold text-slate-100">Change Email</h2>
        <form onSubmit={handleEmailSubmit} className="flex flex-col gap-4">
          <div>
            <label htmlFor="settings-new-email" className="sc-label">New Email</label>
            <input
              id="settings-new-email"
              type="email"
              required
              className="sc-input"
              value={emailForm.new_email}
              onChange={(e) => setEmailForm((f) => ({ ...f, new_email: e.target.value }))}
              placeholder="new@sentinelcam.io"
            />
          </div>
          <div>
            <label htmlFor="settings-email-current-password" className="sc-label">Current Password</label>
            <input
              id="settings-email-current-password"
              type="password"
              required
              className="sc-input"
              value={emailForm.current_password}
              onChange={(e) => setEmailForm((f) => ({ ...f, current_password: e.target.value }))}
            />
          </div>

          {emailStatus.message && (
            <div
              className={`rounded-lg border px-3 py-2 text-sm ${
                emailStatus.type === 'success'
                  ? 'border-status-ok/30 bg-status-ok/10 text-green-300'
                  : 'border-status-error/30 bg-status-error/10 text-red-300'
              }`}
            >
              {emailStatus.message}
            </div>
          )}

          <div>
            <button type="submit" disabled={emailSubmitting} className="sc-btn-primary">
              {emailSubmitting ? 'Updating…' : 'Update Email'}
            </button>
          </div>
        </form>
      </div>

      <div className="sc-card p-6">
        <h2 className="mb-4 text-lg font-semibold text-slate-100">Change Password</h2>
        <form onSubmit={handlePasswordSubmit} className="flex flex-col gap-4">
          <div>
            <label htmlFor="settings-password-current-password" className="sc-label">Current Password</label>
            <input
              id="settings-password-current-password"
              type="password"
              required
              className="sc-input"
              value={passwordForm.current_password}
              onChange={(e) =>
                setPasswordForm((f) => ({ ...f, current_password: e.target.value }))
              }
            />
          </div>
          <div>
            <label htmlFor="settings-new-password" className="sc-label">New Password</label>
            <input
              id="settings-new-password"
              type="password"
              required
              className="sc-input"
              value={passwordForm.new_password}
              onChange={(e) => setPasswordForm((f) => ({ ...f, new_password: e.target.value }))}
            />
          </div>
          <div>
            <label htmlFor="settings-confirm-new-password" className="sc-label">Confirm New Password</label>
            <input
              id="settings-confirm-new-password"
              type="password"
              required
              className="sc-input"
              value={passwordForm.confirm_password}
              onChange={(e) =>
                setPasswordForm((f) => ({ ...f, confirm_password: e.target.value }))
              }
            />
          </div>

          {passwordStatus.message && (
            <div
              className={`rounded-lg border px-3 py-2 text-sm ${
                passwordStatus.type === 'success'
                  ? 'border-status-ok/30 bg-status-ok/10 text-green-300'
                  : 'border-status-error/30 bg-status-error/10 text-red-300'
              }`}
            >
              {passwordStatus.message}
            </div>
          )}

          <div>
            <button type="submit" disabled={passwordSubmitting} className="sc-btn-primary">
              {passwordSubmitting ? 'Updating…' : 'Update Password'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
