import { NavLink, useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'

const navLinkClass = ({ isActive }) =>
  `px-3 py-2 rounded-md text-sm font-medium transition ${
    isActive
      ? 'bg-surface-800 text-accent-cyan'
      : 'text-slate-400 hover:text-slate-100 hover:bg-surface-800/60'
  }`

export default function Navbar() {
  const user = useAuthStore((s) => s.user)
  const isAdmin = useAuthStore((s) => s.isAdmin())
  const logout = useAuthStore((s) => s.logout)
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <header className="sticky top-0 z-40 border-b border-surface-700 bg-surface-950/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-2.5">
            <span className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-accent-blue to-accent-cyan shadow-glow">
              <span className="absolute h-2.5 w-2.5 rounded-full bg-surface-950" />
              <span className="h-1.5 w-1.5 rounded-full bg-accent-cyan animate-pulse" />
            </span>
            <span className="text-lg font-bold tracking-tight text-slate-100">
              Sentinel<span className="text-accent-cyan">Cam</span>
            </span>
          </div>
          <nav className="hidden items-center gap-1 sm:flex">
            <NavLink to="/cameras" className={navLinkClass}>
              Cameras
            </NavLink>
            <NavLink to="/recordings" className={navLinkClass}>
              Recordings
            </NavLink>
            <NavLink to="/settings" className={navLinkClass}>
              Settings
            </NavLink>
            {isAdmin && (
              <NavLink to="/admin/users" className={navLinkClass}>
                Admin Users
              </NavLink>
            )}
          </nav>
        </div>
        <div className="flex items-center gap-3">
          {user && (
            <span className="hidden text-sm text-slate-400 sm:inline">
              {user.full_name || user.email}
              {isAdmin && (
                <span className="ml-2 sc-badge bg-accent-blue/10 text-accent-blue">Admin</span>
              )}
            </span>
          )}
          <button onClick={handleLogout} className="sc-btn-secondary text-xs">
            Logout
          </button>
        </div>
      </div>
      <nav className="flex items-center gap-1 overflow-x-auto border-t border-surface-800 px-4 py-1.5 sm:hidden">
        <NavLink to="/cameras" className={navLinkClass}>
          Cameras
        </NavLink>
        <NavLink to="/recordings" className={navLinkClass}>
          Recordings
        </NavLink>
        <NavLink to="/settings" className={navLinkClass}>
          Settings
        </NavLink>
        {isAdmin && (
          <NavLink to="/admin/users" className={navLinkClass}>
            Admin Users
          </NavLink>
        )}
      </nav>
    </header>
  )
}
