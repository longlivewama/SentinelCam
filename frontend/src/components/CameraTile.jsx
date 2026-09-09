import { useNavigate } from 'react-router-dom'
import CameraStream from './CameraStream'

const STATUS_STYLES = {
  active: 'bg-status-ok/10 text-status-ok border-status-ok/30',
  online: 'bg-status-ok/10 text-status-ok border-status-ok/30',
  inactive: 'bg-status-warn/10 text-status-warn border-status-warn/30',
  offline: 'bg-status-warn/10 text-status-warn border-status-warn/30',
  error: 'bg-status-error/10 text-status-error border-status-error/30',
}

function statusStyle(status) {
  return STATUS_STYLES[String(status).toLowerCase()] || STATUS_STYLES.inactive
}

export default function CameraTile({ camera }) {
  const navigate = useNavigate()

  return (
    <button
      onClick={() => navigate(`/cameras/${camera.id}`)}
      className="sc-card group flex flex-col overflow-hidden text-left transition hover:border-accent-cyan/50 hover:shadow-glow"
    >
      <div className="relative aspect-video w-full overflow-hidden bg-surface-800">
        {camera.is_active !== false ? (
          <CameraStream
            cameraId={camera.id}
            alt={`${camera.name} live stream`}
            className="h-full w-full object-cover"
            onError={(e) => {
              e.currentTarget.style.display = 'none'
              e.currentTarget.nextSibling.style.display = 'flex'
            }}
          />
        ) : null}
        <div
          className="absolute inset-0 hidden items-center justify-center bg-surface-850 text-xs text-slate-500"
          style={{ display: camera.is_active !== false ? 'none' : 'flex' }}
        >
          Stream unavailable
        </div>
        <div className="absolute left-2 top-2 flex items-center gap-1.5 rounded-md bg-surface-950/70 px-2 py-1 backdrop-blur">
          <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-300">
            Live
          </span>
        </div>
        {camera.ai_detection_enabled && (
          <div className="absolute right-2 top-2 flex items-center gap-1.5 rounded-md border border-accent-cyan/40 bg-surface-950/70 px-2 py-1 backdrop-blur">
            <span className="h-1.5 w-1.5 rounded-full bg-accent-cyan" />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-accent-cyan">
              AI Detect
            </span>
          </div>
        )}
      </div>
      <div className="flex flex-1 flex-col gap-1.5 p-4">
        <div className="flex items-center justify-between gap-2">
          <h3 className="truncate font-semibold text-slate-100 group-hover:text-accent-cyan">
            {camera.name}
          </h3>
          <span className={`sc-badge border ${statusStyle(camera.status)}`}>{camera.status}</span>
        </div>
        <p className="truncate text-sm text-slate-400">{camera.location || 'No location set'}</p>
      </div>
    </button>
  )
}
