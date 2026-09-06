import { useState } from 'react'

const DEFAULTS = {
  name: '',
  url: '',
  camera_type: 'ip',
  location: '',
  ai_detection_enabled: false,
  crowd_threshold: 10,
  abandoned_object_seconds: 30,
  status: 'active',
  is_active: true,
}

export default function CameraForm({
  initialValues,
  onSubmit,
  onCancel,
  submitLabel = 'Save',
  showStatusFields = false,
}) {
  const [values, setValues] = useState({ ...DEFAULTS, ...initialValues })
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const update = (field, value) => setValues((v) => ({ ...v, [field]: value }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setSubmitting(true)
    try {
      const payload = {
        ...values,
        crowd_threshold: Number(values.crowd_threshold),
        abandoned_object_seconds: Number(values.abandoned_object_seconds),
      }
      await onSubmit(payload)
    } catch (err) {
      setError(err?.response?.data?.detail || 'Failed to save camera.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div>
        <label className="sc-label">Name</label>
        <input
          required
          className="sc-input"
          value={values.name}
          onChange={(e) => update('name', e.target.value)}
          placeholder="Front Entrance"
        />
      </div>
      <div>
        <label className="sc-label">Stream URL</label>
        <input
          required
          className="sc-input"
          value={values.url}
          onChange={(e) => update('url', e.target.value)}
          placeholder="rtsp://192.168.1.20/stream"
        />
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="sc-label">Camera Type</label>
          <select
            className="sc-input"
            value={values.camera_type}
            onChange={(e) => update('camera_type', e.target.value)}
          >
            <option value="ip">IP</option>
            <option value="usb">USB</option>
          </select>
        </div>
        <div>
          <label className="sc-label">Location</label>
          <input
            className="sc-input"
            value={values.location}
            onChange={(e) => update('location', e.target.value)}
            placeholder="Lobby"
          />
        </div>
      </div>

      {showStatusFields && (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="sc-label">Status</label>
            <select
              className="sc-input"
              value={values.status}
              onChange={(e) => update('status', e.target.value)}
            >
              <option value="active">Active</option>
              <option value="inactive">Inactive</option>
              <option value="error">Error</option>
            </select>
          </div>
          <div className="flex items-end pb-2">
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={values.is_active}
                onChange={(e) => update('is_active', e.target.checked)}
                className="h-4 w-4 rounded border-surface-600 bg-surface-800 text-accent-cyan focus:ring-accent-cyan"
              />
              Camera Active
            </label>
          </div>
        </div>
      )}

      <label className="flex items-center gap-2 text-sm text-slate-300">
        <input
          type="checkbox"
          checked={values.ai_detection_enabled}
          onChange={(e) => update('ai_detection_enabled', e.target.checked)}
          className="h-4 w-4 rounded border-surface-600 bg-surface-800 text-accent-cyan focus:ring-accent-cyan"
        />
        Enable AI Detection
      </label>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="sc-label">Crowd Threshold</label>
          <input
            type="number"
            min="0"
            className="sc-input"
            value={values.crowd_threshold}
            onChange={(e) => update('crowd_threshold', e.target.value)}
          />
        </div>
        <div>
          <label className="sc-label">Abandoned Object (sec)</label>
          <input
            type="number"
            min="0"
            className="sc-input"
            value={values.abandoned_object_seconds}
            onChange={(e) => update('abandoned_object_seconds', e.target.value)}
          />
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-status-error/30 bg-status-error/10 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="mt-2 flex justify-end gap-3">
        {onCancel && (
          <button type="button" onClick={onCancel} className="sc-btn-secondary">
            Cancel
          </button>
        )}
        <button type="submit" disabled={submitting} className="sc-btn-primary">
          {submitting ? 'Saving…' : submitLabel}
        </button>
      </div>
    </form>
  )
}
