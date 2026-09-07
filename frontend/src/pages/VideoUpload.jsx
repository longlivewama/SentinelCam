import { useCallback, useEffect, useRef, useState } from 'react'
import apiClient, { API_URL } from '../api/client'
import Modal from '../components/Modal'
import { useAuthStore } from '../store/authStore'
import { toast } from '../store/toastStore'
import { onRealtimeEvent } from '../lib/realtime'
import { formatBytes, formatDateTime } from '../lib/format'
import { ALLOWED_VIDEO_EXTENSIONS, MAX_UPLOAD_SIZE_MB, validateVideoFile } from '../lib/videoValidation'

const STATUS_STYLES = {
  completed: 'border-status-ok/30 bg-status-ok/10 text-status-ok',
  processing: 'border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan',
  pending: 'border-status-warn/30 bg-status-warn/10 text-status-warn',
  failed: 'border-status-error/30 bg-status-error/10 text-status-error',
}

// Plain-language explanation of each backend status, for users who won't
// know what "pending" means in this pipeline.
const STATUS_HINTS = {
  pending: 'Queued - analysis will start shortly.',
  processing: 'Analyzing the video for people and falls.',
  completed: 'Analysis finished.',
  failed: 'Analysis could not be completed.',
}

// Statuses the backend can still move away from on its own; while any
// upload is in one of these we keep asking the server for fresh state.
const ACTIVE_STATUSES = ['pending', 'processing']
const POLL_INTERVAL_MS = 3000
// Tolerate a couple of blips before telling the user live updates are off.
const POLL_FAILURES_BEFORE_WARNING = 3

function isActive(upload) {
  return ACTIVE_STATUSES.includes(upload.status)
}

function UploadDetailModal({ upload, onClose }) {
  const token = useAuthStore((s) => s.token)
  const [recordings, setRecordings] = useState([])
  const [fallEvents, setFallEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState('')

  useEffect(() => {
    let cancelled = false
    const fetchResults = async () => {
      setLoading(true)
      setLoadError('')
      try {
        // The per-fall confidence scores live on the event rows, the
        // playable clips on the recording rows; both are keyed by this
        // upload, so fetch them together and pair them up below.
        const [recordingsRes, eventsRes] = await Promise.all([
          apiClient.get('/api/recordings', { params: { video_upload_id: upload.id } }),
          apiClient.get('/api/alerts', { params: { video_upload_id: upload.id, event_type: 'fall' } }),
        ])
        if (cancelled) return
        setRecordings(recordingsRes.data)
        setFallEvents(eventsRes.data)
      } catch {
        if (cancelled) return
        // Must not fall through to the "no falls detected" branch - a
        // failed request is not a negative detection result, and showing
        // it as one would misreport the analysis.
        setLoadError('Could not load the fall events for this video.')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    fetchResults()
    return () => {
      cancelled = true
    }
  }, [upload.id])

  const sourceVideoUrl = `${API_URL}/api/video-uploads/${upload.id}/video?token=${token}`

  const recordingsById = new Map(recordings.map((rec) => [rec.id, rec]))
  // Oldest-first so the numbering below reads in the order the falls
  // were detected; the API returns newest-first.
  const orderedEvents = [...fallEvents].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
  const unpairedRecordings = recordings.filter(
    (rec) => !orderedEvents.some((evt) => evt.recording_id === rec.id),
  )

  return (
    <Modal title={upload.original_filename} onClose={onClose} maxWidth="max-w-2xl">
      <div className="mb-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div>
          <p className="sc-label">Status</p>
          <p className="text-sm capitalize text-slate-200">{upload.status}</p>
        </div>
        <div>
          <p className="sc-label">Persons detected</p>
          <p className="text-sm text-slate-200">{upload.persons_detected}</p>
        </div>
        <div>
          <p className="sc-label">Fall events</p>
          <p className="text-sm text-slate-200">{upload.fall_events_count}</p>
        </div>
        <div>
          <p className="sc-label">Duration</p>
          <p className="text-sm text-slate-200">
            {upload.duration_seconds ? `${upload.duration_seconds.toFixed(1)}s` : '—'}
          </p>
        </div>
      </div>

      {upload.status === 'failed' && (
        <div className="mb-4 rounded-lg border border-status-error/30 bg-status-error/10 px-4 py-3 text-sm text-red-300">
          <p className="font-medium">This video could not be analyzed.</p>
          <p className="mt-1 text-red-300/80">
            {upload.error_message || 'The server did not report a reason. Try uploading the file again.'}
          </p>
        </div>
      )}

      <div className="mb-4">
        <p className="sc-label mb-2">Source video</p>
        <video controls className="w-full rounded-lg bg-black" src={sourceVideoUrl}>
          Your browser does not support the video tag.
        </video>
      </div>

      <p className="sc-label mb-2">Fall event clips</p>
      {loading ? (
        <p className="text-sm text-slate-500">Loading…</p>
      ) : loadError ? (
        <p className="text-sm text-red-300">{loadError}</p>
      ) : orderedEvents.length === 0 && recordings.length === 0 ? (
        <p className="text-sm text-slate-500">
          {upload.status === 'completed'
            ? 'No fall events detected.'
            : upload.status === 'failed'
              ? 'Analysis failed before any results were produced.'
              : 'Processing not finished yet.'}
        </p>
      ) : (
        <div className="flex flex-col gap-4">
          {orderedEvents.map((event, index) => {
            const recording = recordingsById.get(event.recording_id)
            return (
              <div key={event.id} className="rounded-lg border border-surface-800 p-3">
                <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                  <p className="text-sm font-medium text-slate-200">Fall {index + 1}</p>
                  <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
                    <span>Detected {formatDateTime(event.timestamp)}</span>
                    <span className="sc-badge border border-status-error/30 bg-status-error/10 text-status-error">
                      {Math.round((event.confidence_score ?? 0) * 100)}% confidence
                    </span>
                  </div>
                </div>
                {recording ? (
                  <video
                    controls
                    className="w-full rounded-lg bg-black"
                    src={`${API_URL}/api/recordings/${recording.id}/video?token=${token}`}
                  />
                ) : (
                  <p className="text-xs text-slate-500">No clip was saved for this event.</p>
                )}
              </div>
            )
          })}
          {unpairedRecordings.map((rec) => (
            <video
              key={rec.id}
              controls
              className="w-full rounded-lg bg-black"
              src={`${API_URL}/api/recordings/${rec.id}/video?token=${token}`}
            />
          ))}
        </div>
      )}
    </Modal>
  )
}

export default function VideoUpload() {
  const fileInputRef = useRef(null)
  const [uploads, setUploads] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [uploadProgress, setUploadProgress] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [viewingUpload, setViewingUpload] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [pollFailures, setPollFailures] = useState(0)

  // Every list request gets a sequence number and only the newest one is
  // allowed to write to state. Without this a slow in-flight response can
  // land after a newer one - or after a local insert/delete - and resurrect
  // rows the user has already removed, or drop rows they just added.
  const requestSeq = useRef(0)

  const loadUploads = useCallback(async ({ showSpinner = false, background = false } = {}) => {
    const seq = (requestSeq.current += 1)
    if (showSpinner) setLoading(true)
    try {
      const { data } = await apiClient.get('/api/video-uploads')
      if (seq !== requestSeq.current) return
      setUploads(data)
      setPollFailures(0)
      if (!background) setError('')
    } catch {
      if (seq !== requestSeq.current) return
      if (background) {
        // A transient blip shouldn't wipe the list or spam toasts; the
        // next tick retries and the banner appears if it keeps failing.
        setPollFailures((n) => n + 1)
      } else {
        setError('Failed to load video uploads.')
      }
    } finally {
      if (showSpinner) setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadUploads({ showSpinner: true })
    const unsubscribe = onRealtimeEvent((event) => {
      if (['upload.progress', 'upload.completed'].includes(event.type)) {
        setUploads((prev) =>
          prev.map((u) => (u.id === event.data.id ? { ...u, ...event.data } : u)),
        )
      }
    })
    return unsubscribe
  }, [loadUploads])

  // Progress arrives over the realtime socket, but that socket can be
  // down (it reconnects with a backoff of up to 30s) - without this an
  // upload would sit at "processing" until the user reloaded the page.
  // Only runs while something is actually in flight, so an idle page
  // makes no requests.
  const hasActiveUpload = uploads.some(isActive)
  useEffect(() => {
    if (!hasActiveUpload) {
      setPollFailures(0)
      return undefined
    }
    const timer = setInterval(() => {
      loadUploads({ background: true })
    }, POLL_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [hasActiveUpload, loadUploads])

  const handleFile = useCallback(
    async (file) => {
      if (!file) return
      // One upload at a time: a second drop or picker selection while a
      // request is in flight would race the first and can double-submit
      // the same file.
      if (uploading) {
        toast.error('An upload is already in progress. Please wait for it to finish.')
        return
      }

      const validationError = validateVideoFile(file)
      if (validationError) {
        toast.error(validationError)
        if (fileInputRef.current) fileInputRef.current.value = ''
        return
      }

      setError('')
      setUploading(true)
      setUploadProgress(0)
      try {
        const formData = new FormData()
        formData.append('file', file)
        const { data } = await apiClient.post('/api/video-uploads', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (evt) => {
            if (evt.total) setUploadProgress(Math.round((evt.loaded / evt.total) * 100))
          },
        })
        toast.success('Video uploaded. Analysis started.')
        // Show it immediately, then reconcile - the refetch also bumps the
        // sequence, so the initial list request (which may still be in
        // flight and predates this upload) can no longer overwrite it.
        setUploads((prev) => [data, ...prev])
        loadUploads({ background: true })
      } catch (err) {
        const detail = err?.response?.data?.detail
        if (detail) {
          toast.error(detail)
        } else if (err?.response) {
          toast.error(`Upload failed (server error ${err.response.status}). Please try again.`)
        } else {
          toast.error('Upload failed - could not reach the server. Check your connection and try again.')
        }
      } finally {
        setUploading(false)
        setUploadProgress(null)
        if (fileInputRef.current) fileInputRef.current.value = ''
      }
    },
    [uploading, loadUploads],
  )

  const handleDelete = async (upload) => {
    if (!window.confirm(`Delete uploaded video "${upload.original_filename}"?`)) return
    try {
      await apiClient.delete(`/api/video-uploads/${upload.id}`)
      setUploads((prev) => prev.filter((u) => u.id !== upload.id))
      toast.success('Video deleted.')
      loadUploads({ background: true })
    } catch {
      toast.error('Failed to delete video.')
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-100">Video Upload &amp; Analysis</h1>
        <p className="mt-1 text-sm text-slate-400">
          Upload a video to run person detection and fall analysis offline. Analysis starts
          automatically once the upload finishes.
        </p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          if (!uploading) setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragActive(false)
          handleFile(e.dataTransfer.files?.[0])
        }}
        className={`sc-card mb-6 flex flex-col items-center justify-center gap-3 border-2 border-dashed p-10 text-center transition ${
          dragActive ? 'border-accent-cyan bg-accent-cyan/5' : 'border-surface-700'
        } ${uploading ? 'opacity-60' : ''}`}
      >
        <p className="text-sm text-slate-300">
          {uploading ? 'Upload in progress…' : 'Drag and drop a video file here, or'}
        </p>
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          className="sc-btn-primary disabled:cursor-not-allowed disabled:opacity-50"
        >
          {uploading ? 'Uploading…' : 'Choose Video'}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept={ALLOWED_VIDEO_EXTENSIONS.join(',')}
          className="hidden"
          disabled={uploading}
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <p className="text-xs text-slate-500">
          {ALLOWED_VIDEO_EXTENSIONS.map((e) => e.replace('.', '').toUpperCase()).join(', ')} · up to{' '}
          {MAX_UPLOAD_SIZE_MB}MB
        </p>
        {uploadProgress !== null && (
          <div className="w-full max-w-sm">
            <div className="h-2 w-full overflow-hidden rounded-full bg-surface-800">
              <div className="h-full bg-accent-cyan transition-all" style={{ width: `${uploadProgress}%` }} />
            </div>
            <p className="mt-1 text-xs text-slate-500">Uploading… {uploadProgress}%</p>
          </div>
        )}
      </div>

      {error && (
        <div className="mb-6 rounded-lg border border-status-error/30 bg-status-error/10 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {pollFailures >= POLL_FAILURES_BEFORE_WARNING && (
        <div className="mb-6 rounded-lg border border-status-warn/30 bg-status-warn/10 px-4 py-3 text-sm text-amber-300">
          Can&apos;t reach the server for status updates - analysis is still running on the server.
          Retrying…
        </div>
      )}

      {loading ? (
        <div className="py-24 text-center text-slate-500">Loading uploads…</div>
      ) : uploads.length === 0 ? (
        <div className="sc-card py-24 text-center text-slate-500">No videos uploaded yet.</div>
      ) : (
        <div className="flex flex-col gap-3">
          {uploads.map((upload) => (
            <div key={upload.id} className="sc-card flex flex-wrap items-center justify-between gap-3 p-4">
              <div className="min-w-0 flex-1">
                <p className="truncate font-medium text-slate-200">{upload.original_filename}</p>
                <p className="text-xs text-slate-500">
                  {formatDateTime(upload.created_at)} · {formatBytes(upload.file_size_bytes)}
                </p>
                {upload.status === 'processing' && (
                  <div className="mt-2 h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-surface-800">
                    <div
                      className="h-full bg-accent-cyan transition-all"
                      style={{ width: `${upload.progress_percent}%` }}
                    />
                  </div>
                )}
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
                <span
                  title={STATUS_HINTS[upload.status] || ''}
                  className={`sc-badge border capitalize ${STATUS_STYLES[upload.status] || ''}`}
                >
                  {upload.status}
                </span>
                <button onClick={() => setViewingUpload(upload)} className="sc-btn-secondary px-3 py-1.5 text-xs">
                  View
                </button>
                <button onClick={() => handleDelete(upload)} className="sc-btn-danger px-3 py-1.5 text-xs">
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {viewingUpload && (
        <UploadDetailModal
          upload={uploads.find((u) => u.id === viewingUpload.id) || viewingUpload}
          onClose={() => setViewingUpload(null)}
        />
      )}
    </div>
  )
}
