import { act, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { createRef } from 'react'
import { describe, expect, it, vi } from 'vitest'
import OverlayControls from './OverlayControls'
import PoseOverlay from './PoseOverlay'
import TrackTimeline from './TrackTimeline'
import TrackingVideoOverlay from './TrackingVideoOverlay'
import { DEFAULT_OVERLAY_OPTIONS } from '../lib/poseOverlay'

// COCO names/skeleton subset sufficient for drawing assertions.
const NAMES = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]
const SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10]]

function keypoints(x, confidence = 0.9) {
  return NAMES.map((_, i) => [x + i, 20 + i * 5, confidence])
}

function person(id, x, overrides = {}) {
  return {
    track_id: id,
    state: 'confirmed',
    bbox: [x, 10, x + 40, 120],
    confidence: 0.87,
    keypoints: keypoints(x),
    ...overrides,
  }
}

function draw(persons, options = DEFAULT_OVERLAY_OPTIONS, extra = {}) {
  return render(
    <div style={{ position: 'relative' }}>
      <PoseOverlay
        persons={persons}
        frameWidth={640}
        frameHeight={360}
        skeleton={SKELETON}
        keypointNames={NAMES}
        keypointMinConfidence={0.3}
        options={options}
        {...extra}
      />
    </div>,
  )
}

describe('PoseOverlay', () => {
  it('draws in the frame’s own pixel space, letterboxed like object-fit: contain', () => {
    draw([person(1, 10)])
    const svg = screen.getByTestId('pose-overlay')
    // The alignment contract: viewBox = source frame, scaled with "meet"
    // (contain) over the same box as the media, so it follows any resize.
    expect(svg.getAttribute('viewBox')).toBe('0 0 640 360')
    expect(svg.getAttribute('preserveAspectRatio')).toBe('xMidYMid meet')
    expect(svg.getAttribute('class')).toMatch(/absolute inset-0 h-full w-full/)
  })

  it('draws boxes, skeletons, keypoints and track ids for every person', () => {
    draw([person(1, 10), person(4, 300)])
    expect(screen.getAllByTestId('person-box')).toHaveLength(2)
    expect(screen.getAllByTestId('skeleton-edge')).toHaveLength(2 * SKELETON.length)
    expect(screen.getAllByTestId('keypoint')).toHaveLength(2 * 17)
    expect(screen.getByText('ID 1')).toBeInTheDocument()
    expect(screen.getByText('ID 4')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: '2 tracked people: IDs 1, 4' })).toBeInTheDocument()
  })

  it('each toggle hides only its own layer', () => {
    const { rerender } = draw([person(1, 10)], { ...DEFAULT_OVERLAY_OPTIONS, boxes: false })
    expect(screen.queryByTestId('person-box')).toBeNull()
    expect(screen.getAllByTestId('skeleton-edge').length).toBeGreaterThan(0)

    rerender(
      <PoseOverlay persons={[person(1, 10)]} frameWidth={640} frameHeight={360} skeleton={SKELETON}
        keypointNames={NAMES} options={{ ...DEFAULT_OVERLAY_OPTIONS, skeleton: false }} />,
    )
    expect(screen.queryByTestId('skeleton-edge')).toBeNull()
    expect(screen.queryByTestId('keypoint')).toBeNull()
    expect(screen.getByTestId('person-box')).toBeInTheDocument()

    rerender(
      <PoseOverlay persons={[person(1, 10)]} frameWidth={640} frameHeight={360} skeleton={SKELETON}
        keypointNames={NAMES} options={{ ...DEFAULT_OVERLAY_OPTIONS, trackIds: false }} />,
    )
    expect(screen.queryByText('ID 1')).toBeNull()

    rerender(
      <PoseOverlay persons={[person(1, 10)]} frameWidth={640} frameHeight={360} skeleton={SKELETON}
        keypointNames={NAMES} options={{ ...DEFAULT_OVERLAY_OPTIONS, confidence: true }} />,
    )
    expect(screen.getByText('ID 1 · 87%')).toBeInTheDocument()
  })

  it('does not draw keypoints or limbs the model was not confident about', () => {
    const partial = keypoints(10)
    partial[9] = [0, 0, 0.05] // left wrist
    draw([person(1, 10, { keypoints: partial })])
    expect(screen.getAllByTestId('keypoint')).toHaveLength(16)
    expect(screen.getAllByTestId('skeleton-edge')).toHaveLength(SKELETON.length - 1)
  })

  it('a person with no usable pose still gets a box', () => {
    draw([person(2, 10, { keypoints: [] })])
    expect(screen.getByTestId('person-box')).toBeInTheDocument()
    expect(screen.queryByTestId('skeleton-edge')).toBeNull()
  })

  it('marks tentative tracks, highlights a fall’s person, and never draws a lost track’s stale box', () => {
    draw(
      [person(1, 10, { state: 'tentative' }), person(2, 200, { state: 'lost' }), person(3, 400)],
      DEFAULT_OVERLAY_OPTIONS,
      { highlightTrackIds: [3] },
    )
    const boxes = screen.getAllByTestId('person-box')
    expect(boxes).toHaveLength(2)
    expect(boxes[0].getAttribute('stroke-dasharray')).toBe('6 4')
    expect(screen.queryByTestId('tracked-person-2')).toBeNull()
    expect(boxes[1].getAttribute('stroke')).toBe('#ef4444')
    expect(screen.getByRole('img', { name: '2 tracked people: IDs 1, 3' })).toBeInTheDocument()
  })

  it('never draws a limb to a keypoint at the frame’s corner', () => {
    // Regression from real footage: a person cut off by the frame edge had a
    // keypoint at (0, 0) with confidence 0.62 - drawn as a line to the corner.
    const cornered = keypoints(300)
    cornered[9] = [0, 0, 0.62]
    draw([person(1, 300, { keypoints: cornered })])
    const edges = screen.getAllByTestId('skeleton-edge')
    expect(edges.every((e) => e.getAttribute('x2') !== '0' && e.getAttribute('x1') !== '0')).toBe(true)
    expect(edges).toHaveLength(SKELETON.length - 1)
    expect(screen.getAllByTestId('keypoint')).toHaveLength(16)
  })

  it('renders nothing without frame dimensions, and a clear label for an empty frame', () => {
    const { container } = render(
      <PoseOverlay persons={[]} frameWidth={0} frameHeight={0} options={DEFAULT_OVERLAY_OPTIONS} />,
    )
    expect(container.querySelector('svg')).toBeNull()
    draw([])
    expect(screen.getByRole('img', { name: 'No people tracked in this frame' })).toBeInTheDocument()
  })
})

describe('OverlayControls', () => {
  it('exposes one labelled checkbox per layer and reports toggles', async () => {
    const onToggle = vi.fn()
    render(<OverlayControls options={DEFAULT_OVERLAY_OPTIONS} onToggle={onToggle} />)
    expect(screen.getByRole('group', { name: 'Overlay' })).toBeInTheDocument()
    expect(screen.getByLabelText('Person boxes')).toBeChecked()
    expect(screen.getByLabelText('Confidence')).not.toBeChecked()
    await userEvent.click(screen.getByLabelText('Pose skeletons'))
    expect(onToggle).toHaveBeenCalledWith('skeleton')
  })

  it('can be disabled as a whole', () => {
    render(<OverlayControls options={DEFAULT_OVERLAY_OPTIONS} onToggle={() => {}} disabled />)
    expect(screen.getByLabelText('Tracking IDs')).toBeDisabled()
  })
})

const TRACKING = {
  video: { frame_width: 640, frame_height: 360, fps: 10, stride: 2 },
  pose_format: { keypoint_names: NAMES, skeleton: SKELETON },
  keypoint_min_confidence: 0.3,
  truncated: false,
  tracks: [
    { track_id: 1, first_seen: 0, last_seen: 8, duration_seconds: 8, frames_observed: 40, max_confidence: 0.9 },
    { track_id: 2, first_seen: 4, last_seen: 6, duration_seconds: 2, frames_observed: 10, max_confidence: 0.8 },
  ],
  fall_events: [
    { event_id: 101, video_time_seconds: 5.2, person_track_id: 2 },
    { event_id: 102, video_time_seconds: 7.0, person_track_id: null },
  ],
  suppressed_events: [{ video_time_seconds: 3.1, person_track_id: 1, reason: 'pose_upright' }],
  frames: [
    { i: 0, t: 0, persons: [person(1, 10)] },
    { i: 50, t: 5, persons: [person(1, 12), person(2, 300)] },
  ],
}

describe('TrackingVideoOverlay', () => {
  it('follows the video’s playhead, including seeks', () => {
    const video = document.createElement('video')
    document.body.appendChild(video)
    const ref = createRef()
    ref.current = video

    render(<TrackingVideoOverlay videoRef={ref} tracking={TRACKING} options={DEFAULT_OVERLAY_OPTIONS} />)
    expect(screen.getAllByTestId('person-box')).toHaveLength(1)

    act(() => {
      Object.defineProperty(video, 'currentTime', { value: 5.1, configurable: true })
      fireEvent(video, new Event('seeked'))
    })
    expect(screen.getAllByTestId('person-box')).toHaveLength(2)

    act(() => {
      Object.defineProperty(video, 'currentTime', { value: 2.0, configurable: true })
      fireEvent(video, new Event('seeked'))
    })
    // 2 s after the last processed frame: nothing rather than a stale box.
    expect(screen.queryByTestId('person-box')).toBeNull()
    video.remove()
  })
})

describe('TrackTimeline', () => {
  it('lists every track with its lifetime and attributed falls, and seeks on click', async () => {
    const onSeek = vi.fn()
    render(<TrackTimeline tracking={TRACKING} duration={10} onSeek={onSeek} />)

    expect(screen.getByText(/2 tracks · up to 2 at once/)).toBeInTheDocument()
    expect(screen.getByText('8.0s · 40 frames')).toBeInTheDocument()
    expect(screen.getByText('2.0s · 10 frames · 1 fall')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Fall attributed to track 2 at 0:05.2' }))
    expect(onSeek).toHaveBeenLastCalledWith(5.2)
    await userEvent.click(screen.getByRole('button', { name: 'Track 1: seen from 0:00.0 to 0:08.0' }))
    expect(onSeek).toHaveBeenLastCalledWith(0)

    expect(screen.getByText(/1 fall event was not attributed/)).toBeInTheDocument()
    expect(screen.getByText(/withheld: person stayed upright after detection/)).toBeInTheDocument()
  })

  it('counts a single frame in the singular', () => {
    const one = { ...TRACKING, tracks: [{ ...TRACKING.tracks[0], frames_observed: 1, duration_seconds: 0 }] }
    render(<TrackTimeline tracking={one} duration={10} />)
    expect(screen.getByText('0.0s · 1 frame')).toBeInTheDocument()
  })

  it('says so when nobody was tracked', () => {
    render(<TrackTimeline tracking={{ ...TRACKING, tracks: [], frames: [] }} duration={10} />)
    expect(screen.getByText('No people were tracked in this video.')).toBeInTheDocument()
  })
})
