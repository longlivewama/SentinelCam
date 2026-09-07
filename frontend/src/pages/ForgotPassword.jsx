import { useState } from 'react'
import { Link } from 'react-router-dom'
import apiClient from '../api/client'

export default function ForgotPassword() {
  const [email, setEmail] = useState('')
  const [loading, setLoading] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await apiClient.post('/api/auth/forgot-password', { email })
      setSubmitted(true)
    } catch {
      setError('Something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-surface-950 px-4">
      <div className="sc-card relative w-full max-w-sm p-8">
        <div className="mb-6 text-center">
          <h1 className="text-xl font-bold tracking-tight text-slate-100">Reset your password</h1>
          <p className="mt-1 text-sm text-slate-400">
            Enter your email and we&rsquo;ll send you a link to reset your password.
          </p>
        </div>

        {submitted ? (
          <div className="rounded-lg border border-status-ok/30 bg-status-ok/10 px-4 py-3 text-sm text-green-300">
            If an account with that email exists, a password reset link has been sent. Check your inbox.
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div>
              <label htmlFor="email" className="sc-label">
                Email
              </label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="sc-input"
                placeholder="you@sentinelcam.io"
              />
            </div>

            {error && (
              <div className="rounded-lg border border-status-error/30 bg-status-error/10 px-3 py-2 text-sm text-red-300">
                {error}
              </div>
            )}

            <button type="submit" disabled={loading} className="sc-btn-primary mt-2 w-full">
              {loading ? 'Sending…' : 'Send Reset Link'}
            </button>
          </form>
        )}

        <div className="mt-6 text-center text-sm">
          <Link to="/login" className="text-slate-400 hover:text-accent-cyan">
            ← Back to sign in
          </Link>
        </div>
      </div>
    </div>
  )
}
