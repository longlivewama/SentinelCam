import { useEffect, useMemo, useState } from 'react'
import apiClient from '../api/client'
import VideoModal from '../components/VideoModal'
import { useAuthStore } from '../store/authStore'
import { useAlertsBadgeStore } from '../store/alertsBadgeStore'
import { toast } from '../store/toastStore'
import { onRealtimeEvent } from '../lib/realtime'
import { eventTypeMeta, formatDateTime } from '../lib/format'

export default function Alerts() {
  const isOperator = useAuthStore((s) => s.isOperator())
  const resetUnread = useAlertsBadgeStore((s) => s.reset)

  const [alerts, setAlerts] = useState([])
  const [cameras, setCameras] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all')
  const [busyId, setBusyId] = useState(null)
  const [viewingRecording, setViewingRecording] = useState(null)

  const fetchData = async () => {
    setLoading(true)
    setError('')
    try {
      const [alertsRes, camerasRes] = await Promise.all([
        apiClient.get('/api/alerts'),
        apiClient.get('/api/cameras'),
      ])
      setAlerts(alertsRes.data)
      setCameras(camerasRes.data)
    } catch {
      setError('Failed to load alerts.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    resetUnread()
    fetchData()

    const unsubscribe = onRealtimeEvent((event) => {
      if (event.type === 'alert.created') {
        fetchData()
      }
    })
    return unsubscribe
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const cameraNameById = useMemo(() => {
    const map = {}
    cameras.forEach((c) => {
      map[c.id] = c.name
    })
    return map
  }, [cameras])

  const filtered = useMemo(() => {
    return alerts
      .filter((a) => typeFilter === 'all' || a.event_type === typeFilter)
      .filter(
        (a) =>
          statusFilter === 'all' ||
          (statusFilter === 'acknowledged' && a.acknowledged) ||
          (statusFilter === 'unacknowledged' && !a.acknowledged),
      )
  }, [alerts, typeFilter, statusFilter])

  const eventTypes = useMemo(() => Array.from(new Set(alerts.map((a) => a.event_type))), [alerts])

  const handleAcknowledge = async (alert) => {
    setBusyId(alert.id)
    try {
      const { data } = await apiClient.put(`/api/alerts/${alert.id}/acknowledge`)
      setAlerts((prev) => prev.map((a) => (a.id === alert.id ? data : a)))
      toast.success('Alert acknowledged.')
    } catch {
      toast.error('Failed to acknowledge alert.')
    } finally {
      setBusyId(null)
    }
  }

  const sourceLabel = (alert) => {
    if (alert.camera_id) return cameraNameById[alert.camera_id] || `Camera #${alert.camera_id}`
    if (alert.video_upload_id) return `Uploaded video #${alert.video_upload_id}`
    return 'Unknown source'
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-100">Alerts</h1>
        <p className="mt-1 text-sm text-slate-400">
          {filtered.length} alert{filtered.length === 1 ? '' : 's'}
        </p>
      </div>

      <div className="mb-6 flex flex-wrap gap-4">
        <div className="w-full max-w-xs">
          <label className="sc-label">Event Type</label>
          <select className="sc-input" value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}>
            <option value="all">All Types</option>
            {eventTypes.map((t) => (
              <option key={t} value={t}>
                {eventTypeMeta(t).label}
              </option>
            ))}
          </select>
        </div>
        <div className="w-full max-w-xs">
          <label className="sc-label">Status</label>
          <select className="sc-input" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="all">All</option>
            <option value="unacknowledged">Unacknowledged</option>
            <option value="acknowledged">Acknowledged</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-status-error/30 bg-status-error/10 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="py-24 text-center text-slate-500">Loading alerts…</div>
      ) : filtered.length === 0 ? (
        <div className="sc-card py-24 text-center text-slate-500">No alerts found.</div>
      ) : (
        <div className="sc-card overflow-x-auto">
          <table className="w-full min-w-[760px] text-left">
            <thead>
              <tr className="border-b border-surface-700 text-xs uppercase tracking-wide text-slate-500">
                <th className="px-4 py-3 font-medium">Event</th>
                <th className="px-4 py-3 font-medium">Source</th>
                <th className="px-4 py-3 font-medium">Time</th>
                <th className="px-4 py-3 font-medium">Confidence</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((alert) => {
                const meta = eventTypeMeta(alert.event_type)
                return (
                  <tr key={alert.id} className="border-b border-surface-800 last:border-b-0 hover:bg-surface-800/40">
                    <td className="whitespace-nowrap px-4 py-3">
                      <span className={`inline-flex items-center gap-1.5 text-sm ${meta.color}`}>
                        <span aria-hidden>{meta.icon}</span>
                        {meta.label}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-300">{sourceLabel(alert)}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-400">
                      {formatDateTime(alert.timestamp)}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-400">
                      {Math.round(alert.confidence_score * 100)}%
                    </td>
                    <td className="whitespace-nowrap px-4 py-3">
                      {alert.acknowledged ? (
                        <span className="sc-badge border border-status-ok/30 bg-status-ok/10 text-status-ok">
                          Acknowledged
                        </span>
                      ) : (
                        <span className="sc-badge border border-status-warn/30 bg-status-warn/10 text-status-warn">
                          New
                        </span>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3">
                      <div className="flex items-center gap-2">
                        {alert.recording_id && (
                          <button
                            onClick={() => setViewingRecording({ id: alert.recording_id, filename: meta.label })}
                            className="sc-btn-secondary px-3 py-1.5 text-xs"
                          >
                            Watch
                          </button>
                        )}
                        {isOperator && !alert.acknowledged && (
                          <button
                            onClick={() => handleAcknowledge(alert)}
                            disabled={busyId === alert.id}
                            className="sc-btn-primary px-3 py-1.5 text-xs"
                          >
                            {busyId === alert.id ? 'Saving…' : 'Acknowledge'}
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      <VideoModal recording={viewingRecording} onClose={() => setViewingRecording(null)} />
    </div>
  )
}
