import { useEffect, useState } from 'react'
import apiClient from '../api/client'
import StatTile from '../components/StatTile'
import BarChart from '../components/BarChart'
import { formatBytes } from '../lib/format'

const WINDOW_OPTIONS = [7, 14, 30, 90]

export default function Analytics() {
  const [days, setDays] = useState(14)
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      setError('')
      try {
        const { data } = await apiClient.get('/api/analytics/summary', { params: { days } })
        setSummary(data)
      } catch {
        setError('Failed to load analytics.')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [days])

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Analytics</h1>
          <p className="mt-1 text-sm text-slate-400">Operational metrics across cameras, alerts, and uploads.</p>
        </div>
        <div className="w-full max-w-[160px]">
          <label htmlFor="analytics-window" className="sc-label">Window</label>
          <select id="analytics-window" className="sc-input" value={days} onChange={(e) => setDays(Number(e.target.value))}>
            {WINDOW_OPTIONS.map((d) => (
              <option key={d} value={d}>
                Last {d} days
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-status-error/30 bg-status-error/10 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {loading || !summary ? (
        <div className="py-24 text-center text-slate-500">Loading analytics…</div>
      ) : (
        <>
          <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatTile label="Total Cameras" value={summary.cameras.total} />
            <StatTile label="Active Cameras" value={summary.cameras.active} accent="ok" />
            <StatTile label="Total Alerts" value={summary.alerts.total} />
            <StatTile label="Storage Used" value={formatBytes(summary.recordings.total_storage_bytes)} />
          </div>

          <div className="mb-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="sc-card p-5">
              <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Falls over time
              </h2>
              <BarChart data={summary.falls.over_time.map((d) => ({ label: d.date.slice(5), value: d.count }))} />
            </div>
            <div className="sc-card p-5">
              <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Falls by camera
              </h2>
              <BarChart
                data={summary.falls.by_camera.map((c) => ({ label: c.camera_name, value: c.count }))}
                emptyLabel="No falls in this window."
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="sc-card p-5">
              <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Avg. detection confidence by type
              </h2>
              <BarChart
                data={Object.entries(summary.confidence.avg_by_type_recent).map(([label, value]) => ({
                  label,
                  value: Math.round(value * 100),
                }))}
                valueFormatter={(v) => `${v}%`}
                emptyLabel="No detections in this window."
              />
            </div>
            <div className="sc-card p-5">
              <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-400">
                Video upload processing
              </h2>
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(summary.video_uploads.by_status).map(([status, count]) => (
                  <div key={status} className="rounded-lg border border-surface-700 p-3">
                    <p className="text-xs uppercase tracking-wide text-slate-500">{status}</p>
                    <p className="text-xl font-bold text-slate-100">{count}</p>
                  </div>
                ))}
                {summary.video_uploads.total === 0 && (
                  <p className="col-span-2 py-4 text-center text-sm text-slate-500">No videos uploaded yet.</p>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
