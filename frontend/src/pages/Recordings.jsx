import { useEffect, useMemo, useState } from 'react'
import apiClient from '../api/client'
import RecordingRow from '../components/RecordingRow'
import VideoModal from '../components/VideoModal'
import { eventTypeMeta } from '../lib/format'

export default function Recordings() {
  const [recordings, setRecordings] = useState([])
  const [cameras, setCameras] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [cameraFilter, setCameraFilter] = useState('all')
  const [eventFilter, setEventFilter] = useState('all')
  const [viewingRecording, setViewingRecording] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      setError('')
      try {
        const [recordingsRes, camerasRes] = await Promise.all([
          apiClient.get('/api/recordings'),
          apiClient.get('/api/cameras'),
        ])
        setRecordings(recordingsRes.data)
        setCameras(camerasRes.data)
      } catch {
        setError('Failed to load recordings.')
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  const cameraNameById = useMemo(() => {
    const map = {}
    cameras.forEach((c) => {
      map[c.id] = c.name
    })
    return map
  }, [cameras])

  const eventTypes = useMemo(() => {
    const set = new Set(recordings.map((r) => r.trigger_action).filter(Boolean))
    return Array.from(set)
  }, [recordings])

  const filtered = useMemo(() => {
    return recordings
      .filter((r) => cameraFilter === 'all' || String(r.camera_id) === String(cameraFilter))
      .filter((r) => eventFilter === 'all' || r.trigger_action === eventFilter)
      .sort((a, b) => {
        const aTime = new Date(a.event_timestamp || a.created_at).getTime()
        const bTime = new Date(b.event_timestamp || b.created_at).getTime()
        return bTime - aTime
      })
  }, [recordings, cameraFilter, eventFilter])

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-100">Recordings</h1>
        <p className="mt-1 text-sm text-slate-400">
          {filtered.length} recording{filtered.length === 1 ? '' : 's'}
        </p>
      </div>

      <div className="mb-6 flex flex-wrap gap-4">
        <div className="w-full max-w-xs">
          <label className="sc-label">Camera</label>
          <select
            className="sc-input"
            value={cameraFilter}
            onChange={(e) => setCameraFilter(e.target.value)}
          >
            <option value="all">All Cameras</option>
            {cameras.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </div>
        <div className="w-full max-w-xs">
          <label className="sc-label">Event Type</label>
          <select
            className="sc-input"
            value={eventFilter}
            onChange={(e) => setEventFilter(e.target.value)}
          >
            <option value="all">All Event Types</option>
            {eventTypes.map((type) => (
              <option key={type} value={type}>
                {eventTypeMeta(type).label}
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

      {loading ? (
        <div className="py-24 text-center text-slate-500">Loading recordings…</div>
      ) : filtered.length === 0 ? (
        <div className="sc-card py-24 text-center text-slate-500">No recordings found.</div>
      ) : (
        <div className="sc-card overflow-x-auto">
          <table className="w-full min-w-[720px] text-left">
            <thead>
              <tr className="border-b border-surface-700 text-xs uppercase tracking-wide text-slate-500">
                <th className="px-4 py-3 font-medium">Camera</th>
                <th className="px-4 py-3 font-medium">Event</th>
                <th className="px-4 py-3 font-medium">Date &amp; Time</th>
                <th className="px-4 py-3 font-medium">Duration</th>
                <th className="px-4 py-3 font-medium">Size</th>
                <th className="px-4 py-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((recording) => (
                <RecordingRow
                  key={recording.id}
                  recording={recording}
                  cameraName={cameraNameById[recording.camera_id]}
                  onView={setViewingRecording}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      <VideoModal recording={viewingRecording} onClose={() => setViewingRecording(null)} />
    </div>
  )
}
