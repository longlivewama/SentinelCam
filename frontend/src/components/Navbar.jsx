import { NavLink, useNavigate } from 'react-router-dom'
import { useAuthStore } from '../store/authStore'
import { useAlertsBadgeStore } from '../store/alertsBadgeStore'
import { useRealtimeStatus } from '../lib/realtime'

const navLinkClass = ({ isActive }) =>
  `relative px-3 py-2 rounded-md text-sm font-medium transition whitespace-nowrap ${
    isActive
      ? 'bg-surface-800 text-accent-cyan'
      : 'text-slate-400 hover:text-slate-100 hover:bg-surface-800/60'
  }`

function NavLinks({ isOperator, isAdmin, unreadCount, onAlertsClick }) {
  return (
    <>
      <NavLink to="/dashboard" className={navLinkClass}>
        Dashboard
      </NavLink>
      <NavLink to="/cameras" className={navLinkClass}>
        Cameras
      </NavLink>
      <NavLink to="/upload" className={navLinkClass}>
        Upload
      </NavLink>
      <NavLink to="/recordings" className={navLinkClass}>
        Recordings
      </NavLink>
      <NavLink to="/alerts" className={navLinkClass} onClick={onAlertsClick}>
        Alerts
        {unreadCount > 0 && (
          <span className="absolute -right-1 -top-1 flex h-4 min-w-4 items-center justify-center rounded-full bg-status-error px-1 text-[10px] font-bold text-white">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </NavLink>
      <NavLink to="/analytics" className={navLinkClass}>
        Analytics
      </NavLink>
      {isOperator && (
        <NavLink to="/system" className={navLinkClass}>
          System
        </NavLink>
      )}
      <NavLink to="/settings" className={navLinkClass}>
        Settings
      </NavLink>
      {isAdmin && (
        <NavLink to="/admin/users" className={navLinkClass}>
          Users
        </NavLink>
      )}
    </>
  )
}

export default function Navbar() {
  const user = useAuthStore((s) => s.user)
  const isAdmin = useAuthStore((s) => s.isAdmin())
  const isOperator = useAuthStore((s) => s.isOperator())
  const logout = useAuthStore((s) => s.logout)
  const navigate = useNavigate()
  const unreadCount = useAlertsBadgeStore((s) => s.unreadCount)
  const resetUnread = useAlertsBadgeStore((s) => s.reset)
  const connected = useRealtimeStatus((s) => s.connected)

  const handleLogout = () => {
    logout()
    navigate('/login', { replace: true })
  }

  return (
    <header className="sticky top-0 z-40 border-b border-surface-700 bg-surface-950/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-2.5">
            <span className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-accent-blue to-accent-cyan shadow-glow">
              <span className="absolute h-2.5 w-2.5 rounded-full bg-surface-950" />
              <span className="h-1.5 w-1.5 rounded-full bg-accent-cyan animate-pulse" />
            </span>
            <span className="hidden text-lg font-bold tracking-tight text-slate-100 sm:inline">
              Sentinel<span className="text-accent-cyan">Cam</span>
            </span>
          </div>
          <nav className="hidden items-center gap-1 lg:flex">
            <NavLinks
              isOperator={isOperator}
              isAdmin={isAdmin}
              unreadCount={unreadCount}
              onAlertsClick={resetUnread}
            />
          </nav>
        </div>
        <div className="flex items-center gap-3">
          <span
            className="hidden items-center gap-1.5 text-xs text-slate-500 sm:flex"
            title={connected ? 'Realtime connection active' : 'Realtime connection lost - reconnecting…'}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${connected ? 'bg-status-ok' : 'bg-status-warn animate-pulse'}`} />
            {connected ? 'Live' : 'Reconnecting'}
          </span>
          {user && (
            <span className="hidden text-sm text-slate-400 sm:inline">
              {user.full_name || user.email}
              <span className="ml-2 sc-badge bg-accent-blue/10 text-accent-blue capitalize">{user.role}</span>
            </span>
          )}
          <button onClick={handleLogout} className="sc-btn-secondary text-xs">
            Logout
          </button>
        </div>
      </div>
      <nav className="flex items-center gap-1 overflow-x-auto border-t border-surface-800 px-4 py-1.5 lg:hidden">
        <NavLinks
          isOperator={isOperator}
          isAdmin={isAdmin}
          unreadCount={unreadCount}
          onAlertsClick={resetUnread}
        />
      </nav>
    </header>
  )
}
