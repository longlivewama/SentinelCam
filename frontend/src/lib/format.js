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
