import { Navigate, Outlet, Route, Routes } from 'react-router-dom'
import ProtectedRoute from './components/ProtectedRoute'
import AdminRoute from './components/AdminRoute'
import Navbar from './components/Navbar'
import Login from './pages/Login'
import Cameras from './pages/Cameras'
import CameraDetail from './pages/CameraDetail'
import Recordings from './pages/Recordings'
import AdminUsers from './pages/AdminUsers'
import Settings from './pages/Settings'

function AppLayout() {
  return (
    <div className="min-h-screen bg-surface-950">
      <Navbar />
      <main>
        <Outlet />
      </main>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />

      <Route element={<ProtectedRoute />}>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Navigate to="/cameras" replace />} />
          <Route path="/cameras" element={<Cameras />} />
          <Route path="/cameras/:id" element={<CameraDetail />} />
          <Route path="/recordings" element={<Recordings />} />
          <Route path="/settings" element={<Settings />} />

          <Route element={<AdminRoute />}>
            <Route path="/admin/users" element={<AdminUsers />} />
          </Route>
        </Route>
      </Route>

      <Route path="*" element={<Navigate to="/cameras" replace />} />
    </Routes>
  )
}
