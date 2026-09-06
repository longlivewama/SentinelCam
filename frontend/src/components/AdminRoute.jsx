import { Navigate, Outlet } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'

export default function AdminRoute() {
  const isAdmin = useAuthStore((s) => s.isAdmin())

  if (!isAdmin) {
    return <Navigate to="/cameras" replace />
  }

  return <Outlet />
}
