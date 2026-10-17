import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  DEFAULT_OVERLAY_OPTIONS,
  describePoseEvidence,
  fallHighlightIds,
  edgeSide,
  formatSeconds,
  frameAtTime,
  loadOverlayOptions,
  maxGapFor,
  saveOverlayOptions,
  visibleEdges,
} from './poseOverlay'

const FRAMES = [{ t: 0 }, { t: 0.5 }, { t: 1.0 }, { t: 3.0 }]

describe('frameAtTime', () => {
  it('shows the latest processed frame at or before the playhead', () => {
    expect(frameAtTime(FRAMES, 0.7, 1)).toBe(FRAMES[1])
    expect(frameAtTime(FRAMES, 1.0, 1)).toBe(FRAMES[2])
    expect(frameAtTime(FRAMES, 3.2, 1)).toBe(FRAMES[3])
  })

  it('shows nothing before the first processed frame or across a gap', () => {
    expect(frameAtTime([{ t: 0.4 }], 0.1, 1)).toBeNull()
    expect(frameAtTime(FRAMES, 2.4, 0.5)).toBeNull() // 1.4 s after t=1.0
  })

  it('handles empty or invalid input', () => {
    expect(frameAtTime([], 1, 1)).toBeNull()
    expect(frameAtTime(undefined, 1, 1)).toBeNull()
    expect(frameAtTime(FRAMES, Number.NaN, 1)).toBeNull()
  })
})

describe('maxGapFor', () => {
  it('covers two processed frames, never less than 0.25 s', () => {
    expect(maxGapFor({ fps: 30, stride: 5 })).toBeCloseTo(1 / 3)
    expect(maxGapFor({ fps: 60, stride: 1 })).toBe(0.25)
    expect(maxGapFor(null)).toBe(0.5)
  })
})

describe('skeleton helpers', () => {
  const names = ['nose', 'left_eye', 'right_eye', 'left_shoulder', 'right_shoulder']

  it('colours limbs by body side', () => {
    expect(edgeSide(names, 1, 3)).toBe('left')
    expect(edgeSide(names, 2, 4)).toBe('right')
    expect(edgeSide(names, 0, 1)).toBe('center')
    expect(edgeSide(names, 3, 4)).toBe('center')
  })

  it('drops edges with an invisible endpoint', () => {
    const keypoints = [[5, 5, 0.9], [1, 1, 0.9], [2, 2, 0.1], [0, 0, 0.9]]
    expect(visibleEdges(keypoints, [[0, 1], [1, 2], [1, 3]], 0.3)).toEqual([[0, 1]])
    expect(visibleEdges([], [[0, 1]], 0.3)).toEqual([])
    expect(visibleEdges(keypoints, undefined, 0.3)).toEqual([])
  })
})

describe('overlay options persistence', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it('round-trips and ignores unknown or malformed values', () => {
    saveOverlayOptions({ ...DEFAULT_OVERLAY_OPTIONS, skeleton: false })
    expect(loadOverlayOptions().skeleton).toBe(false)
    window.localStorage.setItem('sentinelcam.overlayOptions', '{"boxes":"yes","extra":true}')
    expect(loadOverlayOptions()).toEqual(DEFAULT_OVERLAY_OPTIONS)
    window.localStorage.setItem('sentinelcam.overlayOptions', 'not json')
    expect(loadOverlayOptions()).toEqual(DEFAULT_OVERLAY_OPTIONS)
  })

  it('survives storage that throws', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    expect(loadOverlayOptions()).toEqual(DEFAULT_OVERLAY_OPTIONS)
    expect(() => saveOverlayOptions(DEFAULT_OVERLAY_OPTIONS)).not.toThrow()
  })
})

it('formats video positions', () => {
  expect(formatSeconds(83.25)).toBe('1:23.3')
  expect(formatSeconds(4)).toBe('0:04.0')
  expect(formatSeconds(undefined)).toBe('—')
})

describe('fallHighlightIds', () => {
  const falls = [
    { video_time_seconds: 20, person_track_id: 1 },
    { video_time_seconds: 1.67, person_track_id: null },
    { video_time_seconds: 21, person_track_id: 7 },
  ]

  it('highlights a person only around their own fall', () => {
    // Regression: person 1 was drawn red for the whole video, including at
    // 1.7 s when they were walking and an unattributed fall fired elsewhere.
    expect(fallHighlightIds(falls, 1.7)).toEqual([])
    expect(fallHighlightIds(falls, 18.9)).toEqual([])
    expect(fallHighlightIds(falls, 19.5)).toEqual([1])
    expect(fallHighlightIds(falls, 21.5)).toEqual([1, 7])
    expect(fallHighlightIds(falls, 23.5)).toEqual([7])
    expect(fallHighlightIds(falls, 24.5)).toEqual([])
  })

  it('handles missing input', () => {
    expect(fallHighlightIds(undefined, 3)).toEqual([])
    expect(fallHighlightIds(falls, Number.NaN)).toEqual([])
  })
})

describe('describePoseEvidence', () => {
  it('words each verdict, with the angle when there is one', () => {
    expect(describePoseEvidence({ verdict: 'upright', torso_angle_deg: 2.6 })).toBe('torso upright at alert (3°)')
    expect(describePoseEvidence({ verdict: 'not_upright', torso_angle_deg: 81 })).toBe('torso not upright at alert (81°)')
    expect(describePoseEvidence({ verdict: 'no_pose_evidence', torso_angle_deg: null })).toBe('no torso pose at alert')
    expect(describePoseEvidence({ verdict: 'no_person_association' })).toBeNull()
    expect(describePoseEvidence(null)).toBeNull()
  })

  it('explains a held event by what the person did after detection', () => {
    const held = (outcome, extra = {}) => ({
      verdict: 'upright',
      torso_angle_deg: 2.6,
      confirmation: { outcome, waited_seconds: 0.4, post_event_torso_angle_deg: 72.4, ...extra },
    })
    expect(describePoseEvidence(held('released_not_upright'))).toBe('upright when detected, down 0.4s later (72°)')
    expect(describePoseEvidence(held('released_track_ended', { waited_seconds: 1.2 }))).toBe(
      'upright when detected, then lost from view 1.2s later',
    )
    expect(describePoseEvidence(held('released_insufficient_evidence'))).toBe('upright when detected, posture after unconfirmed')
    expect(describePoseEvidence(held('released_source_ended'))).toBe('upright when detected, posture after unconfirmed')
    // Never presents an unheld or unknown outcome as anything but the fire-time reading.
    expect(describePoseEvidence(held('something_new'))).toBe('torso upright at alert (3°)')
  })
})
