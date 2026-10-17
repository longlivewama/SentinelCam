import { edgeSide, isVisibleKeypoint, visibleEdges } from '../lib/poseOverlay'

// Draws tracked people - boxes, pose skeletons, keypoints, track ids and
// confidence - over a video or the live MJPEG view.
//
// Alignment without measuring anything: the SVG's viewBox is the source
// frame in its own pixels, and `preserveAspectRatio="xMidYMid meet"` scales
// and letterboxes it exactly the way `object-fit: contain` (an <img> with
// object-contain, or a <video>, which contains by default) letterboxes the
// media underneath. Laid over the same box as the media element, the two
// stay aligned at any size, through resizes, with no JavaScript layout
// reads. Strokes use `non-scaling-stroke` so lines keep their on-screen
// width however large the frame is.
//
// Rendering only: what to draw comes from the backend's inference, unedited.
// Tracks in the `lost` state (not observed on this frame, kept alive only so
// a brief miss does not cost the person their id) are not drawn: their last
// box is a stale position, and drawing it put a ghost on top of whoever is
// standing there now.

const COLORS = {
  box: '#22d3ee',
  tentative: '#94a3b8',
  fall: '#ef4444',
  left: '#38bdf8',
  right: '#f472b6',
  center: '#e2e8f0',
}

export default function PoseOverlay({
  persons,
  frameWidth,
  frameHeight,
  skeleton,
  keypointNames,
  keypointMinConfidence = 0.3,
  options,
  highlightTrackIds = [],
  className = 'pointer-events-none absolute inset-0 h-full w-full',
}) {
  if (!frameWidth || !frameHeight || !Array.isArray(persons)) return null

  const unit = Math.max(frameWidth, frameHeight)
  const fontSize = unit * 0.022
  const pointRadius = unit * 0.0045
  const highlighted = new Set(highlightTrackIds)
  const visible = persons.filter((p) => p.state !== 'lost')
  const ids = visible.map((p) => p.track_id).filter((id) => id !== null && id !== undefined)

  return (
    <svg
      className={className}
      viewBox={`0 0 ${frameWidth} ${frameHeight}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label={
        visible.length === 0
          ? 'No people tracked in this frame'
          : `${visible.length} tracked ${visible.length === 1 ? 'person' : 'people'}${ids.length ? `: IDs ${ids.join(', ')}` : ''}`
      }
      data-testid="pose-overlay"
    >
      {visible.map((person) => {
        const [x1, y1, x2, y2] = person.bbox
        const tentative = person.state === 'tentative'
        const fall = highlighted.has(person.track_id)
        const boxColor = fall ? COLORS.fall : tentative ? COLORS.tentative : COLORS.box
        const edges = options.skeleton ? visibleEdges(person.keypoints, skeleton, keypointMinConfidence) : []
        const labelParts = []
        if (options.trackIds) labelParts.push(`ID ${person.track_id}`)
        if (options.confidence && Number.isFinite(person.confidence)) {
          labelParts.push(`${Math.round(person.confidence * 100)}%`)
        }
        const label = labelParts.join(' · ')

        return (
          <g
            key={person.track_id}
            data-testid={`tracked-person-${person.track_id}`}
            data-state={person.state}
          >
            {options.boxes && (
              <rect
                data-testid="person-box"
                x={x1}
                y={y1}
                width={Math.max(x2 - x1, 0)}
                height={Math.max(y2 - y1, 0)}
                fill="none"
                stroke={boxColor}
                strokeWidth={fall ? 3 : 2}
                strokeDasharray={tentative ? '6 4' : undefined}
                vectorEffect="non-scaling-stroke"
              />
            )}
            {edges.map(([a, b]) => (
              <line
                key={`${a}-${b}`}
                data-testid="skeleton-edge"
                x1={person.keypoints[a][0]}
                y1={person.keypoints[a][1]}
                x2={person.keypoints[b][0]}
                y2={person.keypoints[b][1]}
                stroke={COLORS[edgeSide(keypointNames, a, b)]}
                strokeWidth={2.5}
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
            ))}
            {options.skeleton &&
              (person.keypoints || []).map(([x, y, c], index) =>
                isVisibleKeypoint([x, y, c], keypointMinConfidence) ? (
                  <circle
                    key={index}
                    data-testid="keypoint"
                    cx={x}
                    cy={y}
                    r={pointRadius}
                    fill="#0b0f14"
                    stroke={COLORS[edgeSide(keypointNames, index, index)]}
                    strokeWidth={1.5}
                    // Fainter points are less certain ones.
                    opacity={Math.min(1, 0.4 + 0.6 * c)}
                    vectorEffect="non-scaling-stroke"
                  />
                ) : null,
              )}
            {label && (
              <g data-testid="track-label">
                <rect
                  x={x1}
                  y={Math.max(y1 - fontSize * 1.45, 0)}
                  width={fontSize * 0.62 * label.length + fontSize * 0.6}
                  height={fontSize * 1.35}
                  rx={fontSize * 0.2}
                  fill={boxColor}
                />
                <text
                  x={x1 + fontSize * 0.3}
                  y={Math.max(y1 - fontSize * 1.45, 0) + fontSize * 1.0}
                  fontSize={fontSize}
                  fontWeight="700"
                  fontFamily="ui-sans-serif, system-ui, sans-serif"
                  fill="#05070a"
                >
                  {label}
                </text>
              </g>
            )}
          </g>
        )
      })}
    </svg>
  )
}
