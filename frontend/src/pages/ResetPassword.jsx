import { useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import apiClient from '../api/client'

export default function ResetPassword() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token') || ''

  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (newPassword.length < 8) {
      setError('Password must be at least 8 characters.')
      return
    }
    if (newPassword !== confirmPassword) {
      setError('Passwords do not match.')
      return
    }

    setLoading(true)
    try {
      await apiClient.post('/api/auth/reset-password', { token, new_password: newPassword })
      setSuccess(true)
      setTimeout(() => navigate('/login', { replace: true }), 2000)
    } catch (err) {
      setError(err?.response?.data?.detail || 'This reset link is invalid or has expired.')
    } finally {
      setLoading(false)
    }
  }

  if (!token) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-surface-950 px-4">
        <div className="sc-card w-full max-w-sm p-8 text-center">
          <p className="text-sm text-red-300">
            This link is missing a reset token. Please request a new password reset.
          </p>
          <Link to="/forgot-password" className="mt-4 inline-block text-sm text-accent-cyan hover:underline">
            Request new link
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-surface-950 px-4">
      <div className="sc-card w-full max-w-sm p-8">
        <div className="mb-6 text-center">
          <h1 className="text-xl font-bold tracking-tight text-slate-100">Choose a new password</h1>
        </div>

        {success ? (
          <div className="rounded-lg border border-status-ok/30 bg-status-ok/10 px-4 py-3 text-sm text-green-300">
            Password reset successfully. Redirecting you to sign in…
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div>
              <label htmlFor="new_password" className="sc-label">
                New password
              </label>
              <input
                id="new_password"
                type="password"
                required
                minLength={8}
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className="sc-input"
                placeholder="At least 8 characters"
              />
            </div>
            <div>
              <label htmlFor="confirm_password" className="sc-label">
                Confirm new password
              </label>
              <input
                id="confirm_password"
                type="password"
                required
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                className="sc-input"
                placeholder="••••••••"
              />
            </div>

            {error && (
              <div className="rounded-lg border border-status-error/30 bg-status-error/10 px-3 py-2 text-sm text-red-300">
                {error}
              </div>
            )}

            <button type="submit" disabled={loading} className="sc-btn-primary mt-2 w-full">
              {loading ? 'Resetting…' : 'Reset Password'}
            </button>
          </form>
        )}
      </div>
    </div>
  )
}
