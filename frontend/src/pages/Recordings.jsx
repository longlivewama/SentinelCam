import { useCallback, useEffect, useMemo, useState } from 'react'
import apiClient from '../api/client'
import Pagination from '../components/Pagination'
import RecordingRow from '../components/RecordingRow'
import VideoModal from '../components/VideoModal'
import { DETECTION_EVENT_TYPES, eventTypeMeta } from '../lib/format'

const PAGE_SIZE = 20

// Filters run on the SERVER now, not over the rows that happen to be on
// screen. Client-side filtering was correct while the API returned every
// row; with paging it would hide matches sitting on other pages and leave
// the page count describing the unfiltered list.
export default function Recordings() {
  const [recordings, setRecordings] = useState([])
  const [pageInfo, setPageInfo] = useState({ page: 1, pages: 0, total: 0 })
  const [cameras, setCameras] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [cameraFilter, setCameraFilter] = useState('all')
  const [eventFilter, setEventFilter] = useState('all')
  const [page, setPage] = useState(1)
  const [viewingRecording, setViewingRecording] = useState(null)

  const fetchRecordings = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const params = { page, page_size: PAGE_SIZE }
      if (cameraFilter !== 'all') params.camera_id = cameraFilter
      if (eventFilter !== 'all') params.trigger_action = eventFilter
      const { data } = await apiClient.get('/api/recordings', { params })
      setRecordings(data.items)
      setPageInfo({ page: data.page, pages: data.pages, total: data.total })
    } catch {
      setError('Failed to load recordings.')
    } finally {
      setLoading(false)
    }
  }, [page, cameraFilter, eventFilter])

  useEffect(() => {
    fetchRecordings()
  }, [fetchRecordings])

  useEffect(() => {
    apiClient
      .get('/api/cameras')
      .then(({ data }) => setCameras(data))
      .catch(() => setCameras([]))
  }, [])

  const cameraNameById = useMemo(() => {
    const map = {}
    cameras.forEach((c) => {
      map[c.id] = c.name
    })
    return map
  }, [cameras])

  // A filter change makes the current page number meaningless - page 4 of
  // the unfiltered list is very unlikely to exist in the filtered one.
  const applyFilter = (setter) => (value) => {
    setter(value)
    setPage(1)
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-100">Recordings</h1>
        <p className="mt-1 text-sm text-slate-400">
          {pageInfo.total} recording{pageInfo.total === 1 ? '' : 's'}
        </p>
      </div>

      <div className="mb-6 flex flex-wrap gap-4">
        <div className="w-full max-w-xs">
          <label htmlFor="recordings-camera-filter" className="sc-label">Camera</label>
          <select
            id="recordings-camera-filter"
            className="sc-input"
            value={cameraFilter}
            onChange={(e) => applyFilter(setCameraFilter)(e.target.value)}
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
          <label htmlFor="recordings-event-type-filter" className="sc-label">Event Type</label>
          <select
            id="recordings-event-type-filter"
            className="sc-input"
            value={eventFilter}
            onChange={(e) => applyFilter(setEventFilter)(e.target.value)}
          >
            <option value="all">All Event Types</option>
            {DETECTION_EVENT_TYPES.map((type) => (
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
      ) : recordings.length === 0 ? (
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
              {recordings.map((recording) => (
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

      <Pagination
        page={pageInfo.page}
        pages={pageInfo.pages}
        total={pageInfo.total}
        pageSize={PAGE_SIZE}
        onChange={setPage}
        busy={loading}
        noun="recordings"
      />

      <VideoModal recording={viewingRecording} onClose={() => setViewingRecording(null)} />
    </div>
  )
}
