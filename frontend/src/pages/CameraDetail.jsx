import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import apiClient, { API_URL } from '../api/client'
import { useAuthStore } from '../store/authStore'
import CameraForm from '../components/CameraForm'
import Modal from '../components/Modal'
import VideoModal from '../components/VideoModal'
import { formatDateTime, formatDuration, formatBytes, eventTypeMeta } from '../lib/format'

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
  const token = useAuthStore((s) => s.token)

  const [camera, setCamera] = useState(null)
  const [recordings, setRecordings] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showEditModal, setShowEditModal] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [viewingRecording, setViewingRecording] = useState(null)

  const fetchData = async () => {
    setLoading(true)
    setError('')
    try {
      const [cameraRes, recordingsRes] = await Promise.all([
        apiClient.get(`/api/cameras/${id}`),
        apiClient.get('/api/recordings', { params: { camera_id: id } }),
      ])
      setCamera(cameraRes.data)
      setRecordings(recordingsRes.data)
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

  const handleUpdate = async (payload) => {
    const { data } = await apiClient.put(`/api/cameras/${id}`, payload)
    setCamera(data)
    setShowEditModal(false)
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

  const streamUrl = `${API_URL}/api/cameras/${camera.id}/stream?token=${token}`

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <button onClick={() => navigate('/cameras')} className="mb-4 text-sm text-slate-400 hover:text-accent-cyan">
        ← Back to Cameras
      </button>

      <div className="sc-card mb-6 overflow-hidden">
        <div className="relative aspect-video w-full bg-black">
          <img src={streamUrl} alt={`${camera.name} live stream`} className="h-full w-full object-contain" />
          <div className="absolute left-3 top-3 flex items-center gap-1.5 rounded-md bg-surface-950/70 px-2.5 py-1 backdrop-blur">
            <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-pulse" />
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">Live</span>
          </div>
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

        <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <dt className="sc-label">Type</dt>
            <dd className="text-sm text-slate-200 uppercase">{camera.camera_type}</dd>
          </div>
          <div>
            <dt className="sc-label">Status</dt>
            <dd className="text-sm text-slate-200 capitalize">{camera.status}</dd>
          </div>
          <div>
            <dt className="sc-label">AI Detection</dt>
            <dd className="text-sm text-slate-200">
              {camera.ai_detection_enabled ? 'Enabled' : 'Disabled'}
            </dd>
          </div>
          <div>
            <dt className="sc-label">Active</dt>
            <dd className="text-sm text-slate-200">{camera.is_active ? 'Yes' : 'No'}</dd>
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
              const downloadUrl = `${API_URL}/api/recordings/${rec.id}/download?token=${token}`
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
                    <a href={downloadUrl} className="sc-btn-secondary px-3 py-1.5 text-xs">
                      Download
                    </a>
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
