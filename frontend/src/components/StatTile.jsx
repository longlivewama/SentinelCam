export default function StatTile({ label, value, hint, accent = 'cyan' }) {
  const accentClass = accent === 'error' ? 'text-status-error' : accent === 'warn' ? 'text-status-warn' : accent === 'ok' ? 'text-status-ok' : 'text-accent-cyan'

  return (
    <div className="sc-card p-5">
      <p className="sc-label mb-2">{label}</p>
      <p className={`text-3xl font-bold tabular-nums ${accentClass}`}>{value}</p>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  )
}
