export default function Modal({ title, onClose, children, maxWidth = 'max-w-lg' }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-surface-950/80 p-4 backdrop-blur-sm">
      <div
        className={`sc-card relative w-full ${maxWidth} max-h-[90vh] overflow-y-auto p-6`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-100">{title}</h2>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-slate-400 hover:bg-surface-800 hover:text-slate-100"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        {children}
      </div>
    </div>
  )
}
