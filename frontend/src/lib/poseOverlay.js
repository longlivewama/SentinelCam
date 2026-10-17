import { useCallback, useState } from 'react'

// Helpers behind the pose/tracking overlay, shared by the live view and the
// uploaded-video results. Everything except `useOverlayOptions` (at the end)
// is a pure function, unit-testable on its own.
//
// Coordinates are always the source frame's own pixels, exactly as the
// backend produced them. Nothing here rescales: the overlay is an SVG whose
// viewBox IS the frame, letterboxed by the browser the same way the video
// is (see PoseOverlay.jsx).

export const DEFAULT_OVERLAY_OPTIONS = Object.freeze({
  boxes: true,
  skeleton: true,
  trackIds: true,
  confidence: false,
})

export const OVERLAY_OPTION_LABELS = {
  boxes: 'Person boxes',
  skeleton: 'Pose skeletons',
  trackIds: 'Tracking IDs',
  confidence: 'Confidence',
}

const STORAGE_KEY = 'sentinelcam.overlayOptions'

/** Saved overlay options, falling back to the defaults on any problem
 * (no storage, private mode, corrupt value). */
export function loadOverlayOptions() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return { ...DEFAULT_OVERLAY_OPTIONS }
    const parsed = JSON.parse(raw)
    const options = { ...DEFAULT_OVERLAY_OPTIONS }
    for (const key of Object.keys(DEFAULT_OVERLAY_OPTIONS)) {
      if (typeof parsed?.[key] === 'boolean') options[key] = parsed[key]
    }
    return options
  } catch {
    return { ...DEFAULT_OVERLAY_OPTIONS }
  }
}

export function saveOverlayOptions(options) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(options))
  } catch {
    // A display preference; losing it is harmless.
  }
}

/**
 * The tracking-record frame to show at `currentTime` (seconds), or null.
 *
 * Inference ran on every `stride`-th frame, so between two processed frames
 * the most recent one is shown. Past `maxGapSeconds` after the last
 * processed frame nothing is shown - a stale overlay drawn on moving video
 * would point at the wrong place. `frames` must be sorted by `t`.
 */
export function frameAtTime(frames, currentTime, maxGapSeconds) {
  if (!Array.isArray(frames) || frames.length === 0 || !Number.isFinite(currentTime)) return null
  let lo = 0
  let hi = frames.length - 1
  if (currentTime < frames[0].t) return null
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2)
    if (frames[mid].t <= currentTime) lo = mid
    else hi = mid - 1
  }
  const frame = frames[lo]
  return currentTime - frame.t <= maxGapSeconds ? frame : null
}

/** How long one processed frame stays on screen in recorded results. */
export function maxGapFor(video) {
  if (!video?.fps || !video?.stride) return 0.5
  return Math.max((2 * video.stride) / video.fps, 0.25)
}

// Body sides, from the COCO keypoint names, so limbs can be told apart.
export function sideOf(name) {
  if (name?.startsWith('left_')) return 'left'
  if (name?.startsWith('right_')) return 'right'
  return 'center'
}

export function edgeSide(keypointNames, a, b) {
  const sideA = sideOf(keypointNames?.[a])
  const sideB = sideOf(keypointNames?.[b])
  return sideA === sideB ? sideA : 'center'
}

/** Whether a keypoint should be drawn. A coordinate at or below 0 is where
 * the model puts a point it could not place (the backend already zeroes
 * their confidence - this keeps older records honest too). */
export function isVisibleKeypoint(point, minConfidence) {
  return Array.isArray(point) && point[0] > 0 && point[1] > 0 && point[2] >= minConfidence
}

/** Skeleton edges whose two endpoints are both visible. */
export function visibleEdges(keypoints, skeleton, minConfidence) {
  if (!Array.isArray(keypoints) || !Array.isArray(skeleton)) return []
  return skeleton.filter(
    ([a, b]) => isVisibleKeypoint(keypoints[a], minConfidence) && isVisibleKeypoint(keypoints[b], minConfidence),
  )
}

// A person is drawn as "falling" only around their own attributed fall: from
// shortly before the alert fired (the detector's sustain window, 0.6 s by
// default, plus margin) to the end of the 3 s post-event clip - not for the
// whole video, which painted someone red at moments they were simply walking.
export const FALL_HIGHLIGHT_BEFORE_SECONDS = 1
export const FALL_HIGHLIGHT_AFTER_SECONDS = 3

/** Track ids whose attributed fall is in progress at `currentTime`. */
export function fallHighlightIds(fallEvents, currentTime) {
  if (!Array.isArray(fallEvents) || !Number.isFinite(currentTime)) return []
  const ids = fallEvents
    .filter(
      (fall) =>
        fall.person_track_id !== null &&
        fall.person_track_id !== undefined &&
        currentTime >= fall.video_time_seconds - FALL_HIGHLIGHT_BEFORE_SECONDS &&
        currentTime <= fall.video_time_seconds + FALL_HIGHLIGHT_AFTER_SECONDS,
    )
    .map((fall) => fall.person_track_id)
  return [...new Set(ids)]
}

/** One-line description of the pose evidence recorded with a fall event.
 * An event the detector fired on an upright person is held by pose
 * verification and only alerts once that changes, so its `confirmation`
 * - what the person did next - is what explains it, not the fire-time
 * verdict alone. */
export function describePoseEvidence(pose) {
  if (!pose) return null
  const angle = Number.isFinite(pose.torso_angle_deg) ? ` (${Math.round(pose.torso_angle_deg)}°)` : ''
  const confirmation = pose.confirmation
  if (pose.verdict === 'upright' && confirmation) {
    const after = Number.isFinite(confirmation.waited_seconds) ? ` ${confirmation.waited_seconds.toFixed(1)}s later` : ' later'
    const postAngle = Number.isFinite(confirmation.post_event_torso_angle_deg)
      ? ` (${Math.round(confirmation.post_event_torso_angle_deg)}°)`
      : ''
    switch (confirmation.outcome) {
      case 'released_not_upright':
        return `upright when detected, down${after}${postAngle}`
      case 'released_track_ended':
        return `upright when detected, then lost from view${after}`
      case 'released_insufficient_evidence':
      case 'released_source_ended':
        return 'upright when detected, posture after unconfirmed'
      default:
        break
    }
  }
  switch (pose.verdict) {
    case 'upright':
      return `torso upright at alert${angle}`
    case 'not_upright':
      return `torso not upright at alert${angle}`
    case 'no_pose_evidence':
      return 'no torso pose at alert'
    default:
      return null
  }
}

export function formatSeconds(value) {
  if (!Number.isFinite(value)) return '—'
  const minutes = Math.floor(value / 60)
  const seconds = value - minutes * 60
  return `${minutes}:${seconds.toFixed(1).padStart(4, '0')}`
}

/** Overlay display options, remembered per browser. */
export function useOverlayOptions() {
  const [options, setOptions] = useState(loadOverlayOptions)
  const toggle = useCallback((key) => {
    setOptions((current) => {
      const next = { ...current, [key]: !current[key] }
      saveOverlayOptions(next)
      return next
    })
  }, [])
  return [options, toggle]
}
