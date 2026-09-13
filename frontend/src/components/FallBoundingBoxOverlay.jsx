// Draws the box a live fall event fired on over a camera's MJPEG view.
//
// The stream's container is a fixed 16:9 box (Tailwind's `aspect-video`,
// used by both CameraTile and CameraDetail) with the actual frame shown
// inside it via `object-contain`, which letterboxes whenever the camera's
// own aspect ratio isn't 16:9. `bbox` arrives in the frame's own pixel
// coordinates (see detection/engine.py's `fall.bbox` broadcast), so it has
// to be re-projected through that same letterboxing to land in the right
// place - this component does that with plain CSS percentages rather than
// measuring the rendered <img>, so it works before the image has painted
// and never fights a layout read/write cycle.
const CONTAINER_ASPECT = 16 / 9

function letterboxRect(frameWidth, frameHeight) {
  const frameAspect = frameWidth / frameHeight
  if (frameAspect > CONTAINER_ASPECT) {
    const heightPct = (CONTAINER_ASPECT / frameAspect) * 100
    return { widthPct: 100, heightPct, leftPct: 0, topPct: (100 - heightPct) / 2 }
  }
  const widthPct = (frameAspect / CONTAINER_ASPECT) * 100
  return { widthPct, heightPct: 100, leftPct: (100 - widthPct) / 2, topPct: 0 }
}

export default function FallBoundingBoxOverlay({ bbox, frameWidth, frameHeight, confidence, trackId }) {
  if (!bbox || !frameWidth || !frameHeight) return null
  const [x1, y1, x2, y2] = bbox
  const { widthPct, heightPct, leftPct, topPct } = letterboxRect(frameWidth, frameHeight)

  const boxStyle = {
    position: 'absolute',
    left: `${leftPct + (x1 / frameWidth) * widthPct}%`,
    top: `${topPct + (y1 / frameHeight) * heightPct}%`,
    width: `${((x2 - x1) / frameWidth) * widthPct}%`,
    height: `${((y2 - y1) / frameHeight) * heightPct}%`,
    border: '3px solid #ef4444',
    boxShadow: '0 0 0 1px rgba(0,0,0,0.4)',
    pointerEvents: 'none',
  }

  const confidencePct = Math.round((confidence ?? 0) * 100)

  return (
    <div style={boxStyle} aria-hidden="true">
      <span className="absolute -top-6 left-0 whitespace-nowrap rounded-sm bg-red-600 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white">
        FALL {confidencePct}%{typeof trackId === 'number' ? ` · ID ${trackId}` : ''}
      </span>
    </div>
  )
}
