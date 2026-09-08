import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import VideoUpload from './VideoUpload'
import apiClient from '../api/client'
import { useAuthStore } from '../store/authStore'
import { useToastStore } from '../store/toastStore'

vi.mock('../api/client', () => ({
  default: { get: vi.fn(), post: vi.fn(), delete: vi.fn() },
  API_URL: 'http://localhost:8000',
}))

vi.mock('../lib/realtime', () => ({
  onRealtimeEvent: vi.fn(() => () => {}),
}))

const UPLOAD = {
  id: 7,
  original_filename: 'ward-3-morning.mp4',
  status: 'completed',
  progress_percent: 100,
  fall_events_count: 2,
  persons_detected: 1,
  duration_seconds: 300,
  file_size_bytes: 2048,
  created_at: '2026-01-01T00:00:00Z',
}

const RECORDINGS = [
  { id: 11, video_upload_id: 7, filename: 'a.mp4', duration_seconds: 6, trigger_action: 'fall' },
  { id: 12, video_upload_id: 7, filename: 'b.mp4', duration_seconds: 6, trigger_action: 'fall' },
]

// Newest-first, as the API returns them - and deliberately in the OPPOSITE
// order to their positions in the footage, so the ordering assertion below
// actually proves the component sorts by in-video time.
const EVENTS = [
  {
    id: 102,
    video_upload_id: 7,
    recording_id: 12,
    event_type: 'fall',
    confidence_score: 0.71,
    timestamp: '2026-01-01T10:05:00Z',
    video_timestamp_seconds: 12.5,
    detector: 'heuristic',
  },
  {
    id: 101,
    video_upload_id: 7,
    recording_id: 11,
    event_type: 'fall',
    confidence_score: 0.93,
    timestamp: '2026-01-01T10:00:00Z',
    video_timestamp_seconds: 245.0,
    detector: 'model',
  },
]

function mockDetailFetch({ events = EVENTS, recordings = RECORDINGS } = {}) {
  apiClient.get.mockImplementation((url) => {
    if (url === '/api/video-uploads') return Promise.resolve({ data: [UPLOAD] })
    if (url === '/api/recordings') return Promise.resolve({ data: recordings })
    if (url === '/api/alerts') return Promise.resolve({ data: events })
    return Promise.reject(new Error(`unexpected request: ${url}`))
  })
}

async function openDetail(user) {
  render(<VideoUpload />)
  await screen.findByText('ward-3-morning.mp4')
  await user.click(screen.getByRole('button', { name: /view/i }))
}

describe('VideoUpload results screen', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAuthStore.setState({ token: 'tok', user: { id: 1, email: 'a@example.com', role: 'viewer' } })
    useToastStore.setState({ toasts: [] })
  })

  it('shows where in the footage each fall happened, not just when it was analysed', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    await openDetail(user)

    // 245s -> 4:05, 12.5s -> 0:12
    expect(await screen.findByText(/at 4:05 in video/i)).toBeInTheDocument()
    expect(screen.getByText(/at 0:12 in video/i)).toBeInTheDocument()
  })

  it('orders falls by their position in the video, not by detection order', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    await openDetail(user)

    await screen.findByText(/at 0:12 in video/i)
    const stamps = screen.getAllByText(/in video/i).map((el) => el.textContent)
    expect(stamps).toEqual(['at 0:12 in video', 'at 4:05 in video'])
  })

  it('seeks the source video when a fall timestamp is clicked', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    // jsdom's HTMLMediaElement has no play(); provide one so the handler runs.
    window.HTMLMediaElement.prototype.play = vi.fn().mockResolvedValue(undefined)
    window.HTMLMediaElement.prototype.scrollIntoView = vi.fn()
    await openDetail(user)

    await user.click(await screen.findByText(/at 4:05 in video/i))

    const sourceVideo = document.querySelector('video')
    expect(sourceVideo.currentTime).toBe(245)
  })

  it('labels which detector produced each fall', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    await openDetail(user)

    expect(await screen.findByText('AI model')).toBeInTheDocument()
    expect(screen.getByText('Pose heuristic')).toBeInTheDocument()
  })

  it('distinguishes the analysis time from the in-video position', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    await openDetail(user)

    expect(await screen.findAllByText(/^Analysed /)).toHaveLength(2)
  })

  it('says the position is unavailable rather than showing a wrong one', async () => {
    const user = userEvent.setup()
    mockDetailFetch({
      events: [{ ...EVENTS[0], video_timestamp_seconds: null, detector: null }],
      recordings: [RECORDINGS[1]],
    })
    await openDetail(user)

    expect(await screen.findByText(/position not recorded/i)).toBeInTheDocument()
    expect(screen.queryByText(/in video/i)).not.toBeInTheDocument()
  })

  it('reports a results-loading failure instead of claiming no falls were found', async () => {
    const user = userEvent.setup()
    apiClient.get.mockImplementation((url) => {
      if (url === '/api/video-uploads') return Promise.resolve({ data: [UPLOAD] })
      return Promise.reject(new Error('boom'))
    })
    await openDetail(user)

    await waitFor(() => {
      expect(screen.getByText(/could not load the fall events/i)).toBeInTheDocument()
    })
    expect(screen.queryByText(/no fall events detected/i)).not.toBeInTheDocument()
  })
})
