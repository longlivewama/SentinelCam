export function formatBytes(bytes) {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) return '—'
  if (bytes === 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  const value = bytes / Math.pow(1024, i)
  return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '—'
  const s = Math.round(seconds)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m ${String(sec).padStart(2, '0')}s`
  if (m > 0) return `${m}m ${String(sec).padStart(2, '0')}s`
  return `${sec}s`
}

/**
 * A position WITHIN a video, as m:ss (or h:mm:ss past an hour) - distinct
 * from formatDateTime, which renders a wall-clock instant. Keeping the two
 * visibly different matters: an event's `timestamp` is when the analysis
 * recorded it, and `video_timestamp_seconds` is where in the footage it
 * happened. Showing the former where the latter belongs tells the operator
 * the wrong thing about their own video.
 */
export function formatVideoTimestamp(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return null
  const total = Math.max(0, Math.floor(seconds))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const mm = h > 0 ? String(m).padStart(2, '0') : String(m)
  return h > 0
    ? `${h}:${mm}:${String(s).padStart(2, '0')}`
    : `${mm}:${String(s).padStart(2, '0')}`
}

export function formatDateTime(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export const EVENT_TYPE_META = {
  // "fall" is the product's primary event type; without an entry here it
  // rendered through the fallback below as a bare lowercase "fall" with a
  // generic bell icon.
  fall: { label: 'Fall Detected', icon: '\u{1F6A8}', color: 'text-status-error' },
  violence: { label: 'Violence Detected', icon: '\u{26A0}', color: 'text-status-error' },
  motion: { label: 'Motion', icon: '\u{1F3C3}', color: 'text-accent-blue' },
  person: { label: 'Person Detected', icon: '\u{1F9CD}', color: 'text-accent-cyan' },
  crowd: { label: 'Crowd Detected', icon: '\u{1F465}', color: 'text-status-warn' },
  abandoned_object: { label: 'Abandoned Object', icon: '\u{1F392}', color: 'text-status-error' },
  manual: { label: 'Manual', icon: '\u{1F4F9}', color: 'text-slate-300' },
  scheduled: { label: 'Scheduled', icon: '\u{23F0}', color: 'text-slate-300' },
}

export function eventTypeMeta(triggerAction) {
  const key = String(triggerAction || '').toLowerCase()
  return (
    EVENT_TYPE_META[key] || {
      label: triggerAction || 'Unknown',
      icon: '\u{1F514}',
      color: 'text-slate-300',
    }
  )
}

/**
 * How a fall event was decided. The trained detector and the pose
 * heuristic compute confidence differently, so a percentage is not
 * interpretable without knowing which produced it.
 */
export const DETECTOR_META = {
  model: {
    label: 'AI model',
    title: 'Detected by the trained YOLO fall-detection model',
  },
  heuristic: {
    label: 'Pose heuristic',
    title: 'Detected by the pose-geometry heuristic (the trained model was not active)',
  },
}

export function detectorMeta(detector) {
  return DETECTOR_META[detector] || null
}
