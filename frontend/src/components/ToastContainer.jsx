import { useToastStore } from '../store/toastStore'

const STYLES = {
  success: 'border-status-ok/30 bg-surface-900 text-status-ok',
  error: 'border-status-error/30 bg-surface-900 text-status-error',
  warning: 'border-status-warn/30 bg-surface-900 text-status-warn',
  info: 'border-accent-cyan/30 bg-surface-900 text-accent-cyan',
}

const ICONS = {
  success: '✓',
  error: '✕',
  warning: '!',
  info: 'i',
}

export default function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts)
  const dismiss = useToastStore((s) => s.dismiss)

  if (toasts.length === 0) return null

  return (
    <div
      className="fixed right-4 top-4 z-[100] flex w-full max-w-sm flex-col gap-2"
      role="region"
      aria-live="polite"
      aria-label="Notifications"
    >
      {toasts.map((t) => (
        <div
          key={t.id}
          role="alert"
          className={`sc-card flex items-start gap-3 border px-4 py-3 shadow-glow ${STYLES[t.type] || STYLES.info}`}
        >
          <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-current text-xs font-bold">
            {ICONS[t.type] || ICONS.info}
          </span>
          <p className="flex-1 text-sm text-slate-200">{t.message}</p>
          <button
            onClick={() => dismiss(t.id)}
            className="text-slate-500 hover:text-slate-200"
            aria-label="Dismiss notification"
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  )
}
