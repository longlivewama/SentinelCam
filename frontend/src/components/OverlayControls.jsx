import { OVERLAY_OPTION_LABELS } from '../lib/poseOverlay'

// Checkboxes rather than custom switches: native keyboard and screen-reader
// support, and each control names exactly what it shows or hides.
export default function OverlayControls({ options, onToggle, disabled = false, children }) {
  return (
    <fieldset className="flex flex-wrap items-center gap-x-4 gap-y-2" disabled={disabled}>
      <legend className="sc-label mb-1 w-full">Overlay</legend>
      {Object.entries(OVERLAY_OPTION_LABELS).map(([key, label]) => (
        <label
          key={key}
          className={`flex cursor-pointer items-center gap-2 text-sm ${disabled ? 'text-slate-600' : 'text-slate-300'}`}
        >
          <input
            type="checkbox"
            className="h-4 w-4 accent-cyan-400"
            checked={Boolean(options[key])}
            onChange={() => onToggle(key)}
          />
          {label}
        </label>
      ))}
      {children}
    </fieldset>
  )
}
