import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import apiClient from '../api/client'
import { useAuthStore } from '../store/authStore'

export default function Signup() {
  const navigate = useNavigate()
  const login = useAuthStore((s) => s.login)
  const [form, setForm] = useState({ full_name: '', email: '', password: '', confirm_password: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const update = (field, value) => setForm((f) => ({ ...f, [field]: value }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')

    if (form.password.length < 8) {
      setError('Password must be at least 8 characters.')
      return
    }
    if (form.password !== form.confirm_password) {
      setError('Passwords do not match.')
      return
    }

    setLoading(true)
    try {
      const { data } = await apiClient.post('/api/auth/signup', {
        email: form.email,
        password: form.password,
        full_name: form.full_name,
      })
      login(data.access_token, data.user)
      navigate('/dashboard', { replace: true })
    } catch (err) {
      if (err.response?.status === 400) {
        setError(err.response.data?.detail || 'An account with that email already exists.')
      } else if (err.response?.status === 422) {
        setError('Please check your details and try again.')
      } else {
        setError('Unable to create your account. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-surface-950 px-4 py-12">
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.07]"
        style={{
          backgroundImage:
            'linear-gradient(rgba(34,211,238,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.5) 1px, transparent 1px)',
          backgroundSize: '48px 48px',
        }}
      />
      <div className="sc-card relative w-full max-w-sm p-8">
        <div className="mb-8 flex flex-col items-center gap-3">
          <span className="relative flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-accent-blue to-accent-cyan shadow-glow">
            <span className="absolute h-4 w-4 rounded-full bg-surface-950" />
            <span className="h-2 w-2 rounded-full bg-accent-cyan animate-pulse" />
          </span>
          <div className="text-center">
            <h1 className="text-xl font-bold tracking-tight text-slate-100">Create your account</h1>
            <p className="mt-1 text-xs uppercase tracking-widest text-slate-500">Sentinel Cam</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div>
            <label htmlFor="full_name" className="sc-label">
              Full name
            </label>
            <input
              id="full_name"
              required
              value={form.full_name}
              onChange={(e) => update('full_name', e.target.value)}
              className="sc-input"
              placeholder="Jane Doe"
            />
          </div>
          <div>
            <label htmlFor="email" className="sc-label">
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              required
              value={form.email}
              onChange={(e) => update('email', e.target.value)}
              className="sc-input"
              placeholder="you@sentinelcam.io"
            />
          </div>
          <div>
            <label htmlFor="password" className="sc-label">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="new-password"
              required
              minLength={8}
              value={form.password}
              onChange={(e) => update('password', e.target.value)}
              className="sc-input"
              placeholder="At least 8 characters"
            />
          </div>
          <div>
            <label htmlFor="confirm_password" className="sc-label">
              Confirm password
            </label>
            <input
              id="confirm_password"
              type="password"
              autoComplete="new-password"
              required
              value={form.confirm_password}
              onChange={(e) => update('confirm_password', e.target.value)}
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
            {loading ? 'Creating account…' : 'Create Account'}
          </button>
        </form>

        <div className="mt-6 text-center text-sm">
          <Link to="/login" className="text-slate-400 hover:text-accent-cyan">
            Already have an account? Sign in
          </Link>
        </div>
      </div>
    </div>
  )
}
