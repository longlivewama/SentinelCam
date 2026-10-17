import { useEffect, useId, useRef } from 'react'

// A dialog over the page: Escape closes it (a backdrop click deliberately
// does not - several of these hold forms a stray click would discard), focus
// moves into it on open and back to whatever opened it on close, and it is
// announced as a modal dialog named by its title.
export default function Modal({ title, onClose, children, maxWidth = 'max-w-lg' }) {
  const titleId = useId()
  const panelRef = useRef(null)
  // Callers pass onClose inline; reading it through a ref keeps the effect
  // below to mount/unmount, so a re-render never pulls focus out of a form
  // field inside the dialog.
  const onCloseRef = useRef(onClose)
  useEffect(() => {
    onCloseRef.current = onClose
  })

  useEffect(() => {
    const opener = document.activeElement
    panelRef.current?.focus()
    const onKeyDown = (event) => {
      if (event.key === 'Escape') onCloseRef.current?.()
    }
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      if (opener instanceof HTMLElement) opener.focus()
    }
  }, [])

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-surface-950/80 p-4 backdrop-blur-sm"
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        className={`sc-card relative w-full ${maxWidth} max-h-[90vh] overflow-y-auto p-6 focus:outline-none`}
      >
        <div className="mb-4 flex items-center justify-between gap-4">
          <h2 id={titleId} className="truncate text-lg font-semibold text-slate-100">{title}</h2>
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
