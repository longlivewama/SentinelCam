import { useEffect } from 'react'
import { Navigate, Outlet, Route, Routes } from 'react-router-dom'
import ProtectedRoute from './components/ProtectedRoute'
import AdminRoute from './components/AdminRoute'
import OperatorRoute from './components/OperatorRoute'
import Navbar from './components/Navbar'
import ToastContainer from './components/ToastContainer'
import Login from './pages/Login'
import Signup from './pages/Signup'
import ForgotPassword from './pages/ForgotPassword'
import ResetPassword from './pages/ResetPassword'
import Dashboard from './pages/Dashboard'
import Cameras from './pages/Cameras'
import CameraDetail from './pages/CameraDetail'
import Recordings from './pages/Recordings'
import Alerts from './pages/Alerts'
import Analytics from './pages/Analytics'
import VideoUpload from './pages/VideoUpload'
import SystemStatus from './pages/SystemStatus'
import AdminUsers from './pages/AdminUsers'
import Settings from './pages/Settings'
import apiClient from './api/client'
import { connectRealtime, disconnectRealtime, onRealtimeEvent } from './lib/realtime'
import { useAlertsBadgeStore } from './store/alertsBadgeStore'
import { useAuthStore } from './store/authStore'
import { toast } from './store/toastStore'

function AppLayout() {
  const increment = useAlertsBadgeStore((s) => s.increment)
  const token = useAuthStore((s) => s.token)
  const login = useAuthStore((s) => s.login)

  useEffect(() => {
    // Refreshes the cached user object (role, is_active, etc.) from the
    // server on every app load, rather than trusting whatever was cached
    // in localStorage at last login - important after role changes made
    // by an admin, or after a backend upgrade that changes the user
    // schema (e.g. this app's own is_admin -> role migration), so a
    // still-logged-in session doesn't silently act on stale permissions.
    apiClient
      .get('/api/auth/me')
      .then(({ data }) => login(token, data))
      .catch(() => {})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    connectRealtime()
    const unsubscribe = onRealtimeEvent((event) => {
      if (event.type === 'alert.created') {
        increment()
        const source = event.data.camera_name || event.data.source_name || 'a source'
        toast.warning(`${(event.data.event_type || 'Event').toUpperCase()} detected on ${source}`)
      } else if (event.type === 'camera.status' && event.data.status === 'error') {
        toast.error(`Camera "${event.data.camera_name}" reported an error`)
      } else if (event.type === 'upload.completed') {
        toast.success(
          `Video analysis finished: ${event.data.fall_events_count} fall event(s) detected`,
        )
      }
    })
    return () => {
      unsubscribe()
      disconnectRealtime()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="min-h-screen bg-surface-950">
      <Navbar />
      <main>
        <Outlet />
      </main>
      <ToastContainer />
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/forgot-password" element={<ForgotPassword />} />
      <Route path="/reset-password" element={<ResetPassword />} />

      <Route element={<ProtectedRoute />}>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/cameras" element={<Cameras />} />
          <Route path="/cameras/:id" element={<CameraDetail />} />
          <Route path="/recordings" element={<Recordings />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/upload" element={<VideoUpload />} />
          <Route path="/settings" element={<Settings />} />

          <Route element={<OperatorRoute />}>
            <Route path="/system" element={<SystemStatus />} />
          </Route>

          <Route element={<AdminRoute />}>
            <Route path="/admin/users" element={<AdminUsers />} />
          </Route>
        </Route>
      </Route>

      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  )
}
