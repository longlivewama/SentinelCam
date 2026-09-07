import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import apiClient from '../api/client'
import StatTile from '../components/StatTile'
import BarChart from '../components/BarChart'
import { onRealtimeEvent } from '../lib/realtime'
import { eventTypeMeta, formatDateTime } from '../lib/format'

const RECENT_ANALYSES_LIMIT = 4

const UPLOAD_STATUS_STYLES = {
  completed: 'border-status-ok/30 bg-status-ok/10 text-status-ok',
  processing: 'border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan',
  pending: 'border-status-warn/30 bg-status-warn/10 text-status-warn',
  failed: 'border-status-error/30 bg-status-error/10 text-status-error',
}

export default function Dashboard() {
  const [summary, setSummary] = useState(null)
  const [recentAlerts, setRecentAlerts] = useState([])
  const [recentUploads, setRecentUploads] = useState([])
  const [uploadsError, setUploadsError] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const fetchData = async () => {
    setError('')
    try {
      const [summaryRes, alertsRes] = await Promise.all([
        apiClient.get('/api/analytics/summary'),
        apiClient.get('/api/alerts', { params: { limit: 6 } }),
      ])
      setSummary(summaryRes.data)
      setRecentAlerts(alertsRes.data)
    } catch {
      setError('Failed to load dashboard data.')
    } finally {
      setLoading(false)
    }
  }

  // Kept out of fetchData's Promise.all deliberately: the uploads list is
  // supplementary, so failing to load it should degrade that one card
  // rather than blanking the whole dashboard.
  const fetchUploads = async () => {
    setUploadsError('')
    try {
      const { data } = await apiClient.get('/api/video-uploads')
      setRecentUploads(data)
    } catch {
      setUploadsError('Could not load recent analyses.')
    }
  }

  useEffect(() => {
    fetchData()
    fetchUploads()
    const unsubscribe = onRealtimeEvent((event) => {
      if (event.type === 'alert.created' || event.type === 'camera.status') {
        fetchData()
      } else if (event.type === 'upload.completed') {
        fetchUploads()
      } else if (event.type === 'upload.progress') {
        setRecentUploads((prev) =>
          prev.map((u) => (u.id === event.data.id ? { ...u, ...event.data } : u)),
        )
      }
    })
    return unsubscribe
  }, [])

  if (loading) {
    return <div className="py-24 text-center text-slate-500">Loading dashboard…</div>
  }

  if (error) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        <div className="rounded-lg border border-status-error/30 bg-status-error/10 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-100">Dashboard</h1>
        <p className="mt-1 text-sm text-slate-400">Live operational overview across all cameras.</p>
      </div>

      <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatTile label="Cameras" value={`${summary.cameras.active}/${summary.cameras.total}`} hint="active / total" />
        <StatTile
          label="Unacknowledged"
          value={summary.alerts.unacknowledged}
          hint={`of ${summary.alerts.total} total alerts`}
          accent={summary.alerts.unacknowledged > 0 ? 'warn' : 'ok'}
        />
        <StatTile
          label={`Falls (${summary.window_days}d)`}
          value={summary.falls.over_time.reduce((sum, d) => sum + d.count, 0)}
          accent="error"
        />
        <StatTile label="Recordings" value={summary.recordings.total} hint="event clips stored" />
      </div>

      <div className="mb-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="sc-card p-5">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-400">
            Falls over time ({summary.window_days}d)
          </h2>
          <BarChart data={summary.falls.over_time.map((d) => ({ label: d.date.slice(5), value: d.count }))} />
        </div>
        <div className="sc-card p-5">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-400">Falls by camera</h2>
          <BarChart
            data={summary.falls.by_camera.map((c) => ({ label: c.camera_name, value: c.count }))}
            emptyLabel="No falls detected in this window."
          />
        </div>
      </div>

      <div className="sc-card mb-6 p-5">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">Recent analyses</h2>
          <Link to="/upload" className="text-sm text-accent-cyan hover:underline">
            Upload a video →
          </Link>
        </div>
        {uploadsError ? (
          <p className="py-8 text-center text-sm text-red-300">{uploadsError}</p>
        ) : recentUploads.length === 0 ? (
          <p className="py-8 text-center text-sm text-slate-500">
            No videos analyzed yet. Upload one to get started.
          </p>
        ) : (
          <div className="flex flex-col divide-y divide-surface-800">
            {recentUploads.slice(0, RECENT_ANALYSES_LIMIT).map((upload, index) => (
              <div key={upload.id} className="flex flex-wrap items-center justify-between gap-3 py-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <p className="truncate text-sm font-medium text-slate-200">
                      {upload.original_filename}
                    </p>
                    {index === 0 && (
                      <span className="sc-badge border border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan">
                        Latest
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-500">{formatDateTime(upload.created_at)}</p>
                  {upload.status === 'failed' && (
                    <p className="mt-1 text-xs text-red-300">
                      {upload.error_message || 'Analysis failed. Try uploading the file again.'}
                    </p>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  {upload.status === 'completed' && (
                    <span className="text-xs text-slate-400">
                      {upload.fall_events_count} fall{upload.fall_events_count === 1 ? '' : 's'} ·{' '}
                      {upload.persons_detected} person{upload.persons_detected === 1 ? '' : 's'}
                    </span>
                  )}
                  {upload.status === 'processing' && (
                    <span className="text-xs text-slate-400">{upload.progress_percent}%</span>
                  )}
                  <span
                    className={`sc-badge border capitalize ${UPLOAD_STATUS_STYLES[upload.status] || ''}`}
                  >
                    {upload.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="sc-card p-5">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">Recent alerts</h2>
          <Link to="/alerts" className="text-sm text-accent-cyan hover:underline">
            View all →
          </Link>
        </div>
        {recentAlerts.length === 0 ? (
          <p className="py-8 text-center text-sm text-slate-500">No alerts yet.</p>
        ) : (
          <div className="flex flex-col divide-y divide-surface-800">
            {recentAlerts.map((alert) => {
              const meta = eventTypeMeta(alert.event_type)
              return (
                <div key={alert.id} className="flex items-center justify-between gap-3 py-3">
                  <div className="flex items-center gap-3">
                    <span className={`text-lg ${meta.color}`} aria-hidden>
                      {meta.icon}
                    </span>
                    <div>
                      <p className="text-sm font-medium text-slate-200">{meta.label}</p>
                      <p className="text-xs text-slate-500">{formatDateTime(alert.timestamp)}</p>
                    </div>
                  </div>
                  {alert.acknowledged ? (
                    <span className="sc-badge border border-status-ok/30 bg-status-ok/10 text-status-ok">
                      Acknowledged
                    </span>
                  ) : (
                    <span className="sc-badge border border-status-warn/30 bg-status-warn/10 text-status-warn">
                      New
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
