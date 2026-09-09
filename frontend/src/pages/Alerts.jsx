import { useCallback, useEffect, useMemo, useState } from 'react'
import apiClient from '../api/client'
import Pagination from '../components/Pagination'
import VideoModal from '../components/VideoModal'
import { useAuthStore } from '../store/authStore'
import { useAlertsBadgeStore } from '../store/alertsBadgeStore'
import { toast } from '../store/toastStore'
import { onRealtimeEvent } from '../lib/realtime'
import { DETECTION_EVENT_TYPES, eventTypeMeta, formatDateTime } from '../lib/format'

const PAGE_SIZE = 20

// Both filters run on the SERVER now. Filtering the twenty rows that
// happen to be on the current page would hide matches on every other one
// and leave the page count describing the unfiltered list.
export default function Alerts() {
  const isOperator = useAuthStore((s) => s.isOperator())
  const resetUnread = useAlertsBadgeStore((s) => s.reset)

  const [alerts, setAlerts] = useState([])
  const [pageInfo, setPageInfo] = useState({ page: 1, pages: 0, total: 0 })
  const [cameras, setCameras] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [typeFilter, setTypeFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all')
  const [page, setPage] = useState(1)
  const [busyId, setBusyId] = useState(null)
  const [viewingRecording, setViewingRecording] = useState(null)

  const fetchAlerts = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const params = { page, page_size: PAGE_SIZE }
      if (typeFilter !== 'all') params.event_type = typeFilter
      if (statusFilter !== 'all') params.acknowledged = statusFilter === 'acknowledged'
      const { data } = await apiClient.get('/api/alerts', { params })
      setAlerts(data.items)
      setPageInfo({ page: data.page, pages: data.pages, total: data.total })
    } catch {
      setError('Failed to load alerts.')
    } finally {
      setLoading(false)
    }
  }, [page, typeFilter, statusFilter])

  useEffect(() => {
    fetchAlerts()
  }, [fetchAlerts])

  useEffect(() => {
    resetUnread()
    apiClient
      .get('/api/cameras')
      .then(({ data }) => setCameras(data))
      .catch(() => setCameras([]))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const unsubscribe = onRealtimeEvent((event) => {
      if (event.type === 'alert.created') {
        fetchAlerts()
      }
    })
    return unsubscribe
  }, [fetchAlerts])

  // A filter change makes the current page number meaningless - page 4 of
  // the unfiltered list is very unlikely to exist in the filtered one.
  const applyFilter = (setter) => (value) => {
    setter(value)
    setPage(1)
  }

  const cameraNameById = useMemo(() => {
    const map = {}
    cameras.forEach((c) => {
      map[c.id] = c.name
    })
    return map
  }, [cameras])

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
          {pageInfo.total} alert{pageInfo.total === 1 ? '' : 's'}
        </p>
      </div>

      <div className="mb-6 flex flex-wrap gap-4">
        <div className="w-full max-w-xs">
          <label htmlFor="alerts-event-type-filter" className="sc-label">Event Type</label>
          <select
            id="alerts-event-type-filter"
            className="sc-input"
            value={typeFilter}
            onChange={(e) => applyFilter(setTypeFilter)(e.target.value)}
          >
            <option value="all">All Types</option>
            {DETECTION_EVENT_TYPES.map((t) => (
              <option key={t} value={t}>
                {eventTypeMeta(t).label}
              </option>
            ))}
          </select>
        </div>
        <div className="w-full max-w-xs">
          <label htmlFor="alerts-status-filter" className="sc-label">Status</label>
          <select
            id="alerts-status-filter"
            className="sc-input"
            value={statusFilter}
            onChange={(e) => applyFilter(setStatusFilter)(e.target.value)}
          >
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
      ) : alerts.length === 0 ? (
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
              {alerts.map((alert) => {
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

      <Pagination
        page={pageInfo.page}
        pages={pageInfo.pages}
        total={pageInfo.total}
        pageSize={PAGE_SIZE}
        onChange={setPage}
        busy={loading}
        noun="alerts"
      />

      <VideoModal recording={viewingRecording} onClose={() => setViewingRecording(null)} />
    </div>
  )
}
