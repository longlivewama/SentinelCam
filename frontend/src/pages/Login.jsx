import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import apiClient from '../api/client'
import { useAuthStore } from '../store/authStore'

export default function Login() {
  const navigate = useNavigate()
  const login = useAuthStore((s) => s.login)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const { data } = await apiClient.post('/api/auth/login', { email, password })
      login(data.access_token, data.user)
      navigate('/cameras', { replace: true })
    } catch (err) {
      if (err.response && err.response.status === 401) {
        setError('Invalid email or password.')
      } else {
        setError('Unable to sign in. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-surface-950 px-4">
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
            <h1 className="text-xl font-bold tracking-tight text-slate-100">
              Sentinel<span className="text-accent-cyan">Cam</span>
            </h1>
            <p className="mt-1 text-xs uppercase tracking-widest text-slate-500">
              Surveillance Console
            </p>
          </div>
        </div>

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
          <div>
            <label htmlFor="password" className="sc-label">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
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
            {loading ? 'Signing in…' : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  )
}
