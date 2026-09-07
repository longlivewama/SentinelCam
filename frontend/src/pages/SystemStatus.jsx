import { useEffect, useState } from 'react'
import apiClient from '../api/client'

function StatusRow({ label, ok, okLabel = 'Ready', badLabel = 'Not configured' }) {
  return (
    <div className="flex items-center justify-between border-b border-surface-800 py-2.5 last:border-b-0">
      <span className="text-sm text-slate-300">{label}</span>
      <span
        className={`sc-badge border ${
          ok
            ? 'border-status-ok/30 bg-status-ok/10 text-status-ok'
            : 'border-surface-600 bg-surface-800 text-slate-400'
        }`}
      >
        {ok ? okLabel : badLabel}
      </span>
    </div>
  )
}

export default function SystemStatus() {
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const fetchStatus = async () => {
      setLoading(true)
      setError('')
      try {
        const { data } = await apiClient.get('/api/system/status')
        setStatus(data)
      } catch {
        setError('Failed to load system status.')
      } finally {
        setLoading(false)
      }
    }
    fetchStatus()
    const interval = setInterval(fetchStatus, 15000)
    return () => clearInterval(interval)
  }, [])

  if (loading) return <div className="py-24 text-center text-slate-500">Loading system status…</div>

  if (error || !status) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8">
        <div className="rounded-lg border border-status-error/30 bg-status-error/10 px-4 py-3 text-sm text-red-300">
          {error || 'No status available.'}
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-100">System Status</h1>
        <p className="mt-1 text-sm text-slate-400">
          Environment: <span className="capitalize text-slate-300">{status.environment}</span> · Inference device:{' '}
          <span className="uppercase text-slate-300">{status.model_device}</span>
        </p>
      </div>

      <div className="sc-card mb-6 p-6">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Models</h2>
        <StatusRow label="Pose + object models loaded" ok={status.models.pose_object_models_loaded} badLabel="Not loaded yet" />
        <StatusRow label={`Fall classifier (corroborating signal)`} ok={status.models.fall_classifier_loaded} />
        <StatusRow label="Violence detection model" ok={status.models.violence_model_configured} okLabel="Trained model" badLabel="Heuristic only" />
      </div>

      <div className="sc-card mb-6 p-6">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Detection</h2>
        <StatusRow
          label="Active camera detection loops"
          ok={status.detection.active_camera_detection_loops > 0}
          okLabel={`${status.detection.active_camera_detection_loops} running`}
          badLabel="None running"
        />
        <div className="flex items-center justify-between border-b border-surface-800 py-2.5">
          <span className="text-sm text-slate-300">Live detection frame stride</span>
          <span className="text-sm text-slate-400">every {status.detection.detection_frame_stride} frames</span>
        </div>
        <div className="flex items-center justify-between py-2.5">
          <span className="text-sm text-slate-300">Video analysis frame stride</span>
          <span className="text-sm text-slate-400">every {status.detection.video_analysis_frame_stride} frames</span>
        </div>
      </div>

      <div className="sc-card mb-6 p-6">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Notifications</h2>
        <div className="flex items-center justify-between border-b border-surface-800 py-2.5">
          <span className="text-sm text-slate-300">Enabled channels</span>
          <span className="text-sm text-slate-400">
            {status.notifications.channels_enabled.length > 0 ? status.notifications.channels_enabled.join(', ') : 'None'}
          </span>
        </div>
        <StatusRow label="Alert recipients configured" ok={status.notifications.alert_recipients_configured} />
      </div>

      <div className="sc-card p-6">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-400">Storage</h2>
        <StatusRow label="Recordings directory writable" ok={status.storage.recordings_dir_writable} />
        <StatusRow label="Uploads directory writable" ok={status.storage.uploads_dir_writable} />
      </div>
    </div>
  )
}
