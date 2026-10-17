import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import CameraDetail from './CameraDetail'
import VideoUpload from './VideoUpload'
import apiClient from '../api/client'
import { onRealtimeEvent } from '../lib/realtime'
import { useAuthStore } from '../store/authStore'
import { useToastStore } from '../store/toastStore'
import { pageOf } from '../test/apiFixtures'

vi.mock('../api/client', () => ({
  default: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
  API_URL: 'http://localhost:8000',
}))

vi.mock('../lib/realtime', () => ({ onRealtimeEvent: vi.fn(() => () => {}) }))

vi.mock('../components/CameraStream', () => ({
  default: ({ cameraId }) => <img data-testid="camera-stream" alt={`stream ${cameraId}`} className="h-full w-full object-contain" />,
}))

const NAMES = [
  'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
  'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
  'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]
const POSE_FORMAT = { keypoint_names: NAMES, skeleton: [[5, 6], [5, 11], [6, 12], [11, 12]] }
const kps = (x) => NAMES.map((_, i) => [x + i, 30 + i * 4, 0.9])

beforeEach(() => {
  vi.clearAllMocks()
  window.localStorage.clear()
  useAuthStore.setState({ token: 'tok', user: { id: 1, role: 'operator' } })
  useToastStore.setState({ toasts: [] })
})

// --- live camera ----------------------------------------------------------

const CAMERA = {
  id: 5, name: 'Lobby', url: '0', camera_type: 'usb', location: 'Lobby', status: 'active',
  is_active: true, ai_detection_enabled: true, crowd_threshold: 20, abandoned_object_seconds: 30,
  created_at: '2026-01-01T00:00:00Z',
}

function renderCamera(camera) {
  apiClient.get.mockImplementation((url) => {
    if (url === '/api/cameras/5') return Promise.resolve({ data: camera })
    if (url === '/api/recordings') return Promise.resolve({ data: pageOf([]) })
    return Promise.reject(new Error(`unexpected url ${url}`))
  })
  const listeners = new Set()
  onRealtimeEvent.mockImplementation((cb) => {
    listeners.add(cb)
    return () => listeners.delete(cb)
  })
  render(
    <MemoryRouter initialEntries={['/cameras/5']}>
      <Routes>
        <Route path="/cameras/:id" element={<CameraDetail />} />
      </Routes>
    </MemoryRouter>,
  )
  return (message) => act(() => listeners.forEach((cb) => cb(message)))
}

function tracksMessage(persons, extra = {}) {
  return {
    type: 'tracks.update',
    data: {
      camera_id: 5, active: true, frame_width: 640, frame_height: 480, keypoint_min_confidence: 0.3,
      pose_format: POSE_FORMAT, persons, ...extra,
    },
  }
}

const LIVE_PERSON = { track_id: 3, state: 'confirmed', bbox: [10, 20, 110, 300], confidence: 0.9, keypoints: kps(20), duration_seconds: 4.2 }

describe('live camera pose and tracking', () => {
  afterEach(() => vi.useRealTimers())

  it('draws tracked people over the stream and lists them', async () => {
    const emit = renderCamera(CAMERA)
    await screen.findByTestId('camera-stream')
    expect(screen.getByTestId('live-tracking-status')).toHaveTextContent('Waiting for detection results')

    emit(tracksMessage([LIVE_PERSON]))
    expect(screen.getByTestId('pose-overlay').getAttribute('viewBox')).toBe('0 0 640 480')
    expect(screen.getByText('ID 3')).toBeInTheDocument()
    expect(screen.getAllByTestId('skeleton-edge')).toHaveLength(4)
    expect(screen.getByTestId('live-tracking-status')).toHaveTextContent('Tracking 1 in view · IDs 3 (4s)')

    // A track held open after a missed frame is neither drawn nor listed.
    emit(tracksMessage([LIVE_PERSON, { ...LIVE_PERSON, track_id: 8, state: 'lost' }]))
    expect(screen.queryByTestId('tracked-person-8')).toBeNull()
    expect(screen.getByTestId('live-tracking-status')).toHaveTextContent('Tracking 1 in view · IDs 3 (4s)')
  })

  it('ignores other cameras and toggles layers', async () => {
    const user = userEvent.setup()
    const emit = renderCamera(CAMERA)
    await screen.findByTestId('camera-stream')

    emit({ ...tracksMessage([LIVE_PERSON]), data: { ...tracksMessage([LIVE_PERSON]).data, camera_id: 99 } })
    expect(screen.queryByTestId('pose-overlay')).toBeNull()

    // Fresh frames between steps, as a live loop sends ~5/s: the overlay is
    // designed to drop itself after 1.5 s without one, and two user events
    // on a loaded test runner can take that long.
    emit(tracksMessage([LIVE_PERSON]))
    await user.click(screen.getByLabelText('Pose skeletons'))
    emit(tracksMessage([LIVE_PERSON]))
    expect(screen.getByTestId('person-box')).toBeInTheDocument()
    expect(screen.queryByTestId('skeleton-edge')).toBeNull()
    await user.click(screen.getByLabelText('Tracking IDs'))
    emit(tracksMessage([LIVE_PERSON]))
    expect(screen.queryByText('ID 3')).toBeNull()
    expect(screen.getByTestId('person-box')).toBeInTheDocument()
  })

  it('clears the overlay when the detection loop stops', async () => {
    const emit = renderCamera(CAMERA)
    await screen.findByTestId('camera-stream')
    emit(tracksMessage([LIVE_PERSON]))
    expect(screen.getByTestId('pose-overlay')).toBeInTheDocument()
    emit({ type: 'tracks.update', data: { camera_id: 5, active: false, persons: [] } })
    expect(screen.queryByTestId('pose-overlay')).toBeNull()
  })

  it('drops a frozen overlay when updates stop arriving', async () => {
    // Installed before render so the staleness interval runs on fake time.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const emit = renderCamera(CAMERA)
    await screen.findByTestId('camera-stream')
    emit(tracksMessage([LIVE_PERSON]))
    expect(screen.getByTestId('pose-overlay')).toBeInTheDocument()
    act(() => vi.advanceTimersByTime(2100))
    expect(screen.queryByTestId('pose-overlay')).toBeNull()
  })

  it('explains that overlays need AI detection, and disables the controls', async () => {
    renderCamera({ ...CAMERA, ai_detection_enabled: false })
    await screen.findByTestId('camera-stream')
    expect(screen.getByTestId('live-tracking-status')).toHaveTextContent('need AI detection turned on')
    expect(screen.getByLabelText('Person boxes')).toBeDisabled()
  })
})

// --- uploaded video -------------------------------------------------------

const UPLOAD = {
  id: 7, original_filename: 'ward.mp4', status: 'completed', progress_percent: 100,
  fall_events_count: 1, persons_detected: 2, duration_seconds: 10, file_size_bytes: 2048,
  created_at: '2026-01-01T00:00:00Z',
}
const EVENT = {
  id: 101, video_upload_id: 7, recording_id: 11, event_type: 'fall', confidence_score: 0.8,
  timestamp: '2026-01-01T10:00:00Z', video_timestamp_seconds: 5.2, detector: 'model',
}
const TRACKING = {
  version: 1,
  video: { frame_width: 640, frame_height: 360, fps: 10, stride: 2 },
  pose_format: POSE_FORMAT,
  keypoint_min_confidence: 0.3,
  settings: {},
  truncated: false,
  tracks: [{ track_id: 2, first_seen: 1, last_seen: 9, duration_seconds: 8, frames_observed: 40, max_confidence: 0.9 }],
  fall_events: [{
    event_id: 101, recording_id: 11, video_time_seconds: 5.2, person_track_id: 2,
    pose: { verdict: 'upright', torso_angle_deg: 2.6, samples: 6 },
  }],
  suppressed_events: [],
  frames: [
    { i: 0, t: 0, persons: [{ track_id: 2, state: 'confirmed', bbox: [5, 5, 50, 200], confidence: 0.9, keypoints: kps(10) }] },
    { i: 52, t: 5.2, persons: [{ track_id: 2, state: 'confirmed', bbox: [5, 90, 200, 160], confidence: 0.9, keypoints: kps(10) }] },
  ],
}

async function openResults(trackingResponse) {
  apiClient.get.mockImplementation((url) => {
    if (url === '/api/video-uploads') return Promise.resolve({ data: pageOf([UPLOAD]) })
    if (url === '/api/recordings') return Promise.resolve({ data: pageOf([{ id: 11, video_upload_id: 7, filename: 'a.mp4' }]) })
    if (url === '/api/alerts') return Promise.resolve({ data: pageOf([EVENT]) })
    if (url === '/api/video-uploads/7/tracking') return trackingResponse()
    return Promise.reject(new Error(`unexpected request: ${url}`))
  })
  apiClient.post.mockResolvedValue({ data: { token: 'media', expires_in: 300 } })
  const user = userEvent.setup()
  render(<VideoUpload />)
  await screen.findByText('ward.mp4')
  await user.click(screen.getByRole('button', { name: /view/i }))
  return user
}

describe('uploaded video pose and tracking', () => {
  it('overlays the source video, offers the controls, and ties the fall to its person', async () => {
    await openResults(() => Promise.resolve({ data: TRACKING }))

    expect(await screen.findByRole('group', { name: 'Overlay' })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByTestId('pose-overlay')).toBeInTheDocument())
    expect(screen.getByTestId('pose-overlay').getAttribute('viewBox')).toBe('0 0 640 360')
    // The overlay lives in the same positioned box as the source <video>.
    const video = document.querySelector('video')
    expect(video.parentElement).toBe(screen.getByTestId('pose-overlay').parentElement)

    expect(screen.getByText('People tracked')).toBeInTheDocument()
    expect(screen.getByText('person ID 2')).toBeInTheDocument()
    expect(screen.getByText('torso upright at alert (3°)')).toBeInTheDocument()

    // The fall's person is highlighted only around the fall (5.2 s), not at 0 s.
    expect(screen.getByTestId('person-box').getAttribute('stroke')).not.toBe('#ef4444')
    act(() => {
      Object.defineProperty(video, 'currentTime', { value: 5.3, configurable: true })
      video.dispatchEvent(new Event('seeked'))
    })
    expect(screen.getByTestId('person-box').getAttribute('stroke')).toBe('#ef4444')
  })

  it('shows a loading state, then an honest empty state for older analyses', async () => {
    let resolve
    await openResults(() => new Promise((_, reject) => {
      resolve = () => reject(Object.assign(new Error('nf'), { response: { status: 404 } }))
    }))
    expect(await screen.findByText(/Loading pose and tracking results/)).toBeInTheDocument()
    await act(async () => resolve())
    expect(await screen.findByText(/No pose or tracking results for this analysis/)).toBeInTheDocument()
    expect(screen.queryByRole('group', { name: 'Overlay' })).toBeNull()
  })

  it('reports a failed load rather than claiming there is no data', async () => {
    await openResults(() => Promise.reject(Object.assign(new Error('boom'), { response: { status: 500 } })))
    expect(await screen.findByText(/Could not load the pose and tracking results/)).toBeInTheDocument()
  })
})
