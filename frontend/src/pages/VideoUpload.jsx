import { useEffect, useRef, useState } from 'react'
import apiClient, { API_URL } from '../api/client'
import Modal from '../components/Modal'
import { useAuthStore } from '../store/authStore'
import { toast } from '../store/toastStore'
import { onRealtimeEvent } from '../lib/realtime'
import { formatBytes, formatDateTime } from '../lib/format'

const STATUS_STYLES = {
  completed: 'border-status-ok/30 bg-status-ok/10 text-status-ok',
  processing: 'border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan',
  pending: 'border-status-warn/30 bg-status-warn/10 text-status-warn',
  failed: 'border-status-error/30 bg-status-error/10 text-status-error',
}

function UploadDetailModal({ upload, onClose }) {
  const token = useAuthStore((s) => s.token)
  const [recordings, setRecordings] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchRecordings = async () => {
      setLoading(true)
      try {
        const { data } = await apiClient.get('/api/recordings', { params: { video_upload_id: upload.id } })
        setRecordings(data)
      } finally {
        setLoading(false)
      }
    }
    fetchRecordings()
  }, [upload.id])

  const sourceVideoUrl = `${API_URL}/api/video-uploads/${upload.id}/video?token=${token}`

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

      <div className="mb-4">
        <p className="sc-label mb-2">Source video</p>
        <video controls className="w-full rounded-lg bg-black" src={sourceVideoUrl}>
          Your browser does not support the video tag.
        </video>
      </div>

      <p className="sc-label mb-2">Fall event clips</p>
      {loading ? (
        <p className="text-sm text-slate-500">Loading…</p>
      ) : recordings.length === 0 ? (
        <p className="text-sm text-slate-500">
          {upload.status === 'completed' ? 'No fall events detected.' : 'Processing not finished yet.'}
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {recordings.map((rec) => (
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
  const [viewingUpload, setViewingUpload] = useState(null)
  const [dragActive, setDragActive] = useState(false)

  const fetchUploads = async () => {
    setLoading(true)
    setError('')
    try {
      const { data } = await apiClient.get('/api/video-uploads')
      setUploads(data)
    } catch {
      setError('Failed to load video uploads.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchUploads()
    const unsubscribe = onRealtimeEvent((event) => {
      if (['upload.progress', 'upload.completed'].includes(event.type)) {
        setUploads((prev) =>
          prev.map((u) => (u.id === event.data.id ? { ...u, ...event.data } : u)),
        )
      }
    })
    return unsubscribe
  }, [])

  const handleFile = async (file) => {
    if (!file) return
    setError('')
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
      setUploads((prev) => [data, ...prev])
    } catch (err) {
      toast.error(err?.response?.data?.detail || 'Failed to upload video.')
    } finally {
      setUploadProgress(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleDelete = async (upload) => {
    if (!window.confirm(`Delete uploaded video "${upload.original_filename}"?`)) return
    try {
      await apiClient.delete(`/api/video-uploads/${upload.id}`)
      setUploads((prev) => prev.filter((u) => u.id !== upload.id))
      toast.success('Video deleted.')
    } catch {
      toast.error('Failed to delete video.')
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-100">Video Upload &amp; Analysis</h1>
        <p className="mt-1 text-sm text-slate-400">
          Upload a video to run person detection and fall analysis offline.
        </p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragActive(false)
          handleFile(e.dataTransfer.files?.[0])
        }}
        className={`sc-card mb-6 flex flex-col items-center justify-center gap-3 border-2 border-dashed p-10 text-center transition ${
          dragActive ? 'border-accent-cyan bg-accent-cyan/5' : 'border-surface-700'
        }`}
      >
        <p className="text-sm text-slate-300">Drag and drop a video file here, or</p>
        <button onClick={() => fileInputRef.current?.click()} className="sc-btn-primary">
          Choose Video
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <p className="text-xs text-slate-500">MP4, MOV, AVI, MKV, WEBM · up to 500MB</p>
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
              </div>
              <div className="flex items-center gap-3">
                {upload.status === 'completed' && (
                  <span className="text-xs text-slate-400">
                    {upload.fall_events_count} fall{upload.fall_events_count === 1 ? '' : 's'} ·{' '}
                    {upload.persons_detected} person{upload.persons_detected === 1 ? '' : 's'}
                  </span>
                )}
                <span className={`sc-badge border capitalize ${STATUS_STYLES[upload.status] || ''}`}>
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

      {viewingUpload && <UploadDetailModal upload={viewingUpload} onClose={() => setViewingUpload(null)} />}
    </div>
  )
}
