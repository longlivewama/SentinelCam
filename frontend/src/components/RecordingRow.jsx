import { API_URL } from '../api/client'
import { useAuthStore } from '../store/authStore'
import { formatBytes, formatDateTime, formatDuration, eventTypeMeta } from '../lib/format'

export default function RecordingRow({ recording, cameraName, onView, onDelete }) {
  const token = useAuthStore((s) => s.token)
  const meta = eventTypeMeta(recording.trigger_action)
  const downloadUrl = `${API_URL}/api/recordings/${recording.id}/download?token=${token}`

  return (
    <tr className="border-b border-surface-800 last:border-b-0 hover:bg-surface-800/40">
      <td className="whitespace-nowrap px-4 py-3">
        <div className="font-medium text-slate-200">{cameraName || `Camera #${recording.camera_id}`}</div>
      </td>
      <td className="whitespace-nowrap px-4 py-3">
        <span className={`inline-flex items-center gap-1.5 text-sm ${meta.color}`}>
          <span aria-hidden>{meta.icon}</span>
          {meta.label}
        </span>
      </td>
      <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-400">
        {formatDateTime(recording.event_timestamp || recording.created_at)}
      </td>
      <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-400">
        {formatDuration(recording.duration_seconds)}
      </td>
      <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-400">
        {formatBytes(recording.file_size_bytes)}
      </td>
      <td className="whitespace-nowrap px-4 py-3">
        <div className="flex items-center gap-2">
          <button onClick={() => onView?.(recording)} className="sc-btn-secondary px-3 py-1.5 text-xs">
            View
          </button>
          <a href={downloadUrl} className="sc-btn-secondary px-3 py-1.5 text-xs">
            Download
          </a>
          {onDelete && (
            <button
              onClick={() => onDelete(recording)}
              className="sc-btn-danger px-3 py-1.5 text-xs"
            >
              Delete
            </button>
          )}
        </div>
      </td>
    </tr>
  )
}
