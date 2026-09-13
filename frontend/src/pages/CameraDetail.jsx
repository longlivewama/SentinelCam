import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import apiClient from '../api/client'
import { useAuthStore } from '../store/authStore'
import CameraForm from '../components/CameraForm'
import CameraStream from '../components/CameraStream'
import FallBoundingBoxOverlay from '../components/FallBoundingBoxOverlay'
import MediaDownloadLink from '../components/MediaDownloadLink'
import Modal from '../components/Modal'
import VideoModal from '../components/VideoModal'
import { onRealtimeEvent } from '../lib/realtime'
import { formatDateTime, formatDuration, formatBytes, eventTypeMeta } from '../lib/format'

// How long a live fall box stays drawn on screen after it fires. The
// underlying event is a single moment, not a stream of positions (see
// detection/engine.py's `fall.bbox` broadcast), so this is a "flash it up
// long enough for a human to see it" duration rather than a tracked
// duration.
const FALL_BBOX_DISPLAY_MS = 6000

// Small ON/OFF pill. Read-only (a badge) for viewers; an operator gets a
// clickable toggle that fires `onToggle` with the opposite of `on`.
function StateToggle({ label, on, onLabel = 'ON', offLabel = 'OFF', onToggle, busy }) {
  const badge = (
    <span
      className={`sc-badge border ${
        on ? 'bg-status-ok/10 text-status-ok border-status-ok/30' : 'bg-surface-800 text-slate-400 border-surface-600'
      }`}
    >
      {on ? onLabel : offLabel}
    </span>
  )
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs font-medium uppercase tracking-wider text-slate-400">{label}</span>
      {onToggle ? (
        <button
          type="button"
          onClick={() => onToggle(!on)}
          disabled={busy}
          className={`sc-badge border transition ${
            on
              ? 'bg-status-ok/10 text-status-ok border-status-ok/30 hover:bg-status-ok/20'
              : 'bg-surface-800 text-slate-400 border-surface-600 hover:bg-surface-700'
          } ${busy ? 'opacity-60' : ''}`}
        >
          {on ? onLabel : offLabel}
        </button>
      ) : (
        badge
      )}
    </div>
  )
}

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

export default function CameraDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const isOperator = useAuthStore((s) => s.isOperator())

  const [camera, setCamera] = useState(null)
  const [recordings, setRecordings] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showEditModal, setShowEditModal] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [viewingRecording, setViewingRecording] = useState(null)
  const [fallAlert, setFallAlert] = useState(null)
  const [togglingField, setTogglingField] = useState(null)
  const fallAlertTimeout = useRef(null)

  const fetchData = async () => {
    setLoading(true)
    setError('')
    try {
      const [cameraRes, recordingsRes] = await Promise.all([
        apiClient.get(`/api/cameras/${id}`),
        // A bounded 'recent recordings' panel; the full history is on the
        // Recordings page, which pages properly.
        apiClient.get('/api/recordings', { params: { camera_id: id, page_size: 10 } }),
      ])
      setCamera(cameraRes.data)
      setRecordings(recordingsRes.data.items)
    } catch {
      setError('Failed to load camera details.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id])

  // Live fall-box overlay: the backend broadcasts the exact box a fall
  // fired on (detection/engine.py's `fall.bbox`) the moment it happens -
  // this just displays it for a few seconds over the live view. No
  // subscription while the camera is off, and any box already showing is
  // cleared the instant it's switched off.
  useEffect(() => {
    if (!camera?.is_active) {
      setFallAlert(null)
      return undefined
    }
    const unsubscribe = onRealtimeEvent((message) => {
      if (message?.type !== 'fall.bbox' || message?.data?.camera_id !== Number(id)) return
      clearTimeout(fallAlertTimeout.current)
      setFallAlert(message.data)
      fallAlertTimeout.current = setTimeout(() => setFallAlert(null), FALL_BBOX_DISPLAY_MS)
    })
    return () => {
      unsubscribe()
      clearTimeout(fallAlertTimeout.current)
    }
  }, [id, camera?.is_active])

  const handleUpdate = async (payload) => {
    const { data } = await apiClient.put(`/api/cameras/${id}`, payload)
    setCamera(data)
    setShowEditModal(false)
  }

  const handleToggleField = (field) => async (nextValue) => {
    setTogglingField(field)
    setError('')
    try {
      const { data } = await apiClient.put(`/api/cameras/${id}`, { [field]: nextValue })
      setCamera(data)
    } catch {
      setError(`Failed to update camera ${field === 'is_active' ? 'status' : 'AI detection'}.`)
    } finally {
      setTogglingField(null)
    }
  }

  const handleDelete = async () => {
    if (!window.confirm(`Delete camera "${camera.name}"? This cannot be undone.`)) return
    setDeleting(true)
    try {
      await apiClient.delete(`/api/cameras/${id}`)
      navigate('/cameras', { replace: true })
    } catch {
      setError('Failed to delete camera.')
      setDeleting(false)
    }
  }

  if (loading) {
    return <div className="py-24 text-center text-slate-500">Loading camera…</div>
  }

  if (error && !camera) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8">
        <div className="rounded-lg border border-status-error/30 bg-status-error/10 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      </div>
    )
  }

  if (!camera) return null

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <button onClick={() => navigate('/cameras')} className="mb-4 text-sm text-slate-400 hover:text-accent-cyan">
        ← Back to Cameras
      </button>

      <div className="sc-card mb-6 overflow-hidden">
        <div className="relative aspect-video w-full bg-black">
          {camera.is_active ? (
            <>
              <CameraStream
                cameraId={camera.id}
                alt={`${camera.name} live stream`}
                className="h-full w-full object-contain"
              />
              <div className="absolute left-3 top-3 flex items-center gap-1.5 rounded-md bg-surface-950/70 px-2.5 py-1 backdrop-blur">
                <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">Live</span>
              </div>
              {fallAlert && (
                <FallBoundingBoxOverlay
                  bbox={fallAlert.bbox}
                  frameWidth={fallAlert.frame_width}
                  frameHeight={fallAlert.frame_height}
                  confidence={fallAlert.confidence}
                  trackId={fallAlert.track_id}
                />
              )}
            </>
          ) : (
            // Camera is intentionally off - a broken <img> would read as a
            // fault; this reads as a deliberate state instead.
            <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-slate-500">
              <span className="h-2.5 w-2.5 rounded-full bg-slate-600" />
              <p className="text-sm font-semibold uppercase tracking-wider">Camera Off</p>
              <p className="text-xs text-slate-600">Turn the camera on below to resume the live view.</p>
            </div>
          )}
        </div>
      </div>

      <div className="sc-card mb-6 p-6">
        <div className="mb-4 flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-100">{camera.name}</h1>
            <p className="mt-1 text-sm text-slate-400">{camera.location || 'No location set'}</p>
          </div>
          <div className="flex items-center gap-2">
            <span className={`sc-badge border ${statusStyle(camera.status)}`}>{camera.status}</span>
            {isOperator && (
              <>
                <button onClick={() => setShowEditModal(true)} className="sc-btn-secondary">
                  Edit
                </button>
                <button onClick={handleDelete} disabled={deleting} className="sc-btn-danger">
                  {deleting ? 'Deleting…' : 'Delete'}
                </button>
              </>
            )}
          </div>
        </div>

        {error && (
          <div className="mb-4 rounded-lg border border-status-error/30 bg-status-error/10 px-3 py-2 text-sm text-red-300">
            {error}
          </div>
        )}

        <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-3 rounded-lg border border-surface-700 bg-surface-900/40 px-4 py-3">
          <StateToggle
            label="Camera"
            on={camera.is_active}
            onToggle={isOperator ? handleToggleField('is_active') : undefined}
            busy={togglingField === 'is_active'}
          />
          <StateToggle
            label="AI Detection"
            on={camera.ai_detection_enabled}
            onToggle={isOperator ? handleToggleField('ai_detection_enabled') : undefined}
            busy={togglingField === 'ai_detection_enabled'}
          />
          <StateToggle label="Live Video" on={camera.is_active} onLabel="ACTIVE" offLabel="INACTIVE" />
          <StateToggle
            label="Fall Detection"
            on={camera.is_active && camera.ai_detection_enabled}
            onLabel="RUNNING"
            offLabel="STOPPED"
          />
        </div>

        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <dt className="sc-label">Type</dt>
            <dd className="text-sm text-slate-200 uppercase">{camera.camera_type}</dd>
          </div>
          <div>
            <dt className="sc-label">Connection</dt>
            <dd className="text-sm text-slate-200 capitalize">{camera.status}</dd>
          </div>
          <div>
            <dt className="sc-label">Crowd Threshold</dt>
            <dd className="text-sm text-slate-200">{camera.crowd_threshold}</dd>
          </div>
          <div>
            <dt className="sc-label">Abandoned Object (sec)</dt>
            <dd className="text-sm text-slate-200">{camera.abandoned_object_seconds}</dd>
          </div>
          <div>
            <dt className="sc-label">Added</dt>
            <dd className="text-sm text-slate-200">{formatDateTime(camera.created_at)}</dd>
          </div>
        </dl>
      </div>

      <div className="sc-card p-6">
        <h2 className="mb-4 text-lg font-semibold text-slate-100">Recent Recordings</h2>
        {recordings.length === 0 ? (
          <p className="py-8 text-center text-sm text-slate-500">No recordings for this camera yet.</p>
        ) : (
          <div className="flex flex-col divide-y divide-surface-800">
            {recordings.map((rec) => {
              const meta = eventTypeMeta(rec.trigger_action)
              return (
                <div key={rec.id} className="flex flex-wrap items-center justify-between gap-3 py-3">
                  <div className="flex items-center gap-3">
                    <span className={`text-lg ${meta.color}`} aria-hidden>
                      {meta.icon}
                    </span>
                    <div>
                      <p className="text-sm font-medium text-slate-200">{meta.label}</p>
                      <p className="text-xs text-slate-500">
                        {formatDateTime(rec.event_timestamp || rec.created_at)} · {formatDuration(rec.duration_seconds)} ·{' '}
                        {formatBytes(rec.file_size_bytes)}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={() => setViewingRecording(rec)} className="sc-btn-secondary px-3 py-1.5 text-xs">
                      View
                    </button>
                    <MediaDownloadLink kind="recording" id={rec.id} />
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {showEditModal && (
        <Modal title="Edit Camera" onClose={() => setShowEditModal(false)}>
          <CameraForm
            initialValues={camera}
            submitLabel="Save Changes"
            showStatusFields
            onSubmit={handleUpdate}
            onCancel={() => setShowEditModal(false)}
          />
        </Modal>
      )}

      <VideoModal recording={viewingRecording} onClose={() => setViewingRecording(null)} />
    </div>
  )
}
