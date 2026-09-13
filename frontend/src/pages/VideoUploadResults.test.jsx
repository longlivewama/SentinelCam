import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import VideoUpload from './VideoUpload'
import apiClient from '../api/client'
import { pageOf } from '../test/apiFixtures'
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

// Media URLs no longer carry the session JWT. Each player asks the API
// for a short-lived token scoped to its own clip, so the mock has to mint
// one - and it returns a DIFFERENT token every time, which is what lets
// the retry assertions below tell a refreshed player from a stale one.
let mintCount = 0

function mockDetailFetch({ events = EVENTS, recordings = RECORDINGS } = {}) {
  mintCount = 0
  apiClient.get.mockImplementation((url) => {
    if (url === '/api/video-uploads') return Promise.resolve({ data: pageOf([UPLOAD]) })
    if (url === '/api/recordings') return Promise.resolve({ data: pageOf(recordings) })
    if (url === '/api/alerts') return Promise.resolve({ data: pageOf(events) })
    return Promise.reject(new Error(`unexpected request: ${url}`))
  })
  apiClient.post.mockImplementation((url) => {
    if (url.endsWith('/media-token')) {
      mintCount += 1
      return Promise.resolve({ data: { token: `media-${mintCount}`, expires_in: 300 } })
    }
    return Promise.reject(new Error(`unexpected request: ${url}`))
  })
}

/** Fails a player twice - the first failure is spent on a token refresh. */
async function failPlayer(index) {
  const at = () => document.querySelectorAll('video')[index]
  const originalSrc = at().getAttribute('src')

  fireEvent.error(at())
  await waitFor(() => {
    expect(at()?.getAttribute('src')).not.toBe(originalSrc)
  })

  fireEvent.error(at())
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
      if (url === '/api/video-uploads') return Promise.resolve({ data: pageOf([UPLOAD]) })
      return Promise.reject(new Error('boom'))
    })
    await openDetail(user)

    await waitFor(() => {
      expect(screen.getByText(/could not load the fall events/i)).toBeInTheDocument()
    })
    expect(screen.queryByText(/no fall events detected/i)).not.toBeInTheDocument()
  })
  // --- playback failure surfacing -------------------------------------
  //
  // These cover a real incident: fall clips were encoded as MPEG-4 Part 2
  // ("mp4v"), which no current browser decodes, while the H.264 source
  // played fine. Every backend layer was healthy - correct rows, files on
  // disk, 206 range responses - so the only visible symptom was a silent
  // black player, which reads as "the results didn't load". The encoder is
  // fixed, but the player must never fail silently again.

  it('reports a clip that the browser cannot play instead of showing a blank player', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    await openDetail(user)

    await screen.findByText(/at 0:12 in video/i)
    await failPlayer(1)

    expect(await screen.findByText(/this video could not be played/i)).toBeInTheDocument()
    expect(screen.getByText(/cannot decode/i)).toBeInTheDocument()
  })

  it('reports a source video that fails to load', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    await openDetail(user)

    await screen.findByText(/at 0:12 in video/i)
    await failPlayer(0)

    expect(await screen.findByText(/this video could not be played/i)).toBeInTheDocument()
  })

  it('keeps the working clips playable when one of them fails', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    await openDetail(user)

    await screen.findByText(/at 0:12 in video/i)
    const before = document.querySelectorAll('video').length
    await failPlayer(1)

    await screen.findByText(/this video could not be played/i)
    // Exactly one player was replaced by the message; the rest still render.
    expect(document.querySelectorAll('video')).toHaveLength(before - 1)
  })

  it('requests metadata up front so a broken clip fails fast rather than hanging', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    await openDetail(user)

    await screen.findByText(/at 0:12 in video/i)
    for (const video of document.querySelectorAll('video')) {
      expect(video.getAttribute('preload')).toBe('metadata')
    }
  })

  // --- results are actually fetched for a completed upload --------------

  it('fetches both recordings and alerts for the upload it was opened for', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    await openDetail(user)

    await screen.findByText(/at 0:12 in video/i)
    const calls = apiClient.get.mock.calls
    const recordingsCall = calls.find(([url]) => url === '/api/recordings')
    const alertsCall = calls.find(([url]) => url === '/api/alerts')

    // One generous page rather than paging: these are one upload's own
    // results, and a video with 100+ detected falls is a footage problem,
    // not a browsing one.
    expect(recordingsCall[1].params).toEqual({ video_upload_id: 7, page_size: 100 })
    expect(alertsCall[1].params).toEqual({ video_upload_id: 7, page_size: 100, event_type: 'fall' })
  })

  it('points each player at the right clip', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    await openDetail(user)

    await screen.findByText(/at 0:12 in video/i)
    await waitFor(() => expect(document.querySelectorAll('video')).toHaveLength(3))
    const sources = [...document.querySelectorAll('video')].map((v) => v.getAttribute('src'))

    expect(sources[0]).toMatch(/^http:\/\/localhost:8000\/api\/video-uploads\/7\/video\?token=/)
    // Falls render in footage order: recording 12 (0:12) before 11 (4:05).
    expect(sources[1]).toMatch(/^http:\/\/localhost:8000\/api\/recordings\/12\/video\?token=/)
    expect(sources[2]).toMatch(/^http:\/\/localhost:8000\/api\/recordings\/11\/video\?token=/)
  })

  it('never puts the session token in a media URL', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    await openDetail(user)

    await screen.findByText(/at 0:12 in video/i)
    await waitFor(() => expect(document.querySelectorAll('video')).toHaveLength(3))
    const sources = [...document.querySelectorAll('video')].map((v) => v.getAttribute('src'))

    // 'tok' is the session JWT this component used to publish into every
    // player's src, and from there into the access log of every proxy on
    // the path. Each player now carries its own scoped, minutes-long token.
    for (const src of sources) {
      expect(src).not.toContain('token=tok')
    }
    expect(new Set(sources.map((s) => new URL(s).searchParams.get('token'))).size).toBe(3)
  })

  it('asks for a token scoped to each resource, not one token for everything', async () => {
    const user = userEvent.setup()
    mockDetailFetch()
    await openDetail(user)

    await screen.findByText(/at 0:12 in video/i)
    await waitFor(() => expect(document.querySelectorAll('video')).toHaveLength(3))

    const minted = apiClient.post.mock.calls.map(([url]) => url)
    expect(minted).toContain('/api/video-uploads/7/media-token')
    expect(minted).toContain('/api/recordings/11/media-token')
    expect(minted).toContain('/api/recordings/12/media-token')
  })

  it('says no falls were detected when a completed upload genuinely has none', async () => {
    const user = userEvent.setup()
    mockDetailFetch({ events: [], recordings: [] })
    await openDetail(user)

    expect(await screen.findByText(/no fall events detected/i)).toBeInTheDocument()
  })
})
