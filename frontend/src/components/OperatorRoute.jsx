import { Navigate, Outlet } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'

export default function OperatorRoute() {
  const isOperator = useAuthStore((s) => s.isOperator())

  if (!isOperator) {
    return <Navigate to="/dashboard" replace />
  }

  return <Outlet />
}
