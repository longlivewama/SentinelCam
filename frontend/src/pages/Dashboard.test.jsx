import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Dashboard from './Dashboard'
import apiClient from '../api/client'
import { pageOf } from '../test/apiFixtures'

vi.mock('../api/client', () => ({
  default: { get: vi.fn() },
  API_URL: 'http://localhost:8000',
}))

vi.mock('../lib/realtime', () => ({
  onRealtimeEvent: vi.fn(() => () => {}),
}))

const SUMMARY = {
  cameras: { total: 3, active: 2 },
  alerts: { total: 10, unacknowledged: 4, by_type_recent: { fall: 3 } },
  falls: {
    over_time: [{ date: '2026-01-01', count: 1 }, { date: '2026-01-02', count: 2 }],
    by_camera: [{ camera_id: 1, camera_name: 'Lobby', count: 3 }],
  },
  confidence: { avg_by_type_recent: { fall: 0.82 } },
  recordings: { total: 5, total_storage_bytes: 123456 },
  video_uploads: { total: 1, by_status: { completed: 1 } },
  window_days: 14,
}

describe('Dashboard page', () => {
  beforeEach(() => vi.clearAllMocks())

  it('shows a loading state, then renders summary stats', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/api/analytics/summary') return Promise.resolve({ data: SUMMARY })
      if (url === '/api/alerts') return Promise.resolve({ data: pageOf([]) })
      if (url === '/api/video-uploads') return Promise.resolve({ data: pageOf([]) })
      return Promise.reject(new Error('unexpected url'))
    })

    render(<MemoryRouter><Dashboard /></MemoryRouter>)
    expect(screen.getByText(/loading dashboard/i)).toBeInTheDocument()

    expect(await screen.findByText('2/3')).toBeInTheDocument() // active/total cameras
    expect(screen.getByText('4')).toBeInTheDocument() // unacknowledged
    expect(screen.getByText('No alerts yet.')).toBeInTheDocument()
    expect(screen.getByText(/no videos analyzed yet/i)).toBeInTheDocument()
  })

  it('lists recent analyses, flags the latest, and explains a failed one', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/api/analytics/summary') return Promise.resolve({ data: SUMMARY })
      if (url === '/api/alerts') return Promise.resolve({ data: pageOf([]) })
      if (url === '/api/video-uploads') {
        return Promise.resolve({
          data: pageOf([
            {
              id: 2,
              original_filename: 'newest.mp4',
              status: 'completed',
              progress_percent: 100,
              fall_events_count: 1,
              persons_detected: 2,
              created_at: '2026-01-02T00:00:00Z',
            },
            {
              id: 1,
              original_filename: 'older.mp4',
              status: 'failed',
              progress_percent: 0,
              error_message: 'Could not open uploaded video file',
              fall_events_count: 0,
              persons_detected: 0,
              created_at: '2026-01-01T00:00:00Z',
            },
          ]),
        })
      }
      return Promise.reject(new Error('unexpected url'))
    })

    render(<MemoryRouter><Dashboard /></MemoryRouter>)

    expect(await screen.findByText('newest.mp4')).toBeInTheDocument()
    expect(screen.getByText('Latest')).toBeInTheDocument()
    expect(screen.getByText(/1 fall/)).toBeInTheDocument()
    expect(screen.getByText(/could not open uploaded video file/i)).toBeInTheDocument()
  })

  it('keeps the rest of the dashboard usable when the analyses request fails', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/api/analytics/summary') return Promise.resolve({ data: SUMMARY })
      if (url === '/api/alerts') return Promise.resolve({ data: pageOf([]) })
      return Promise.reject(new Error('uploads unavailable'))
    })

    render(<MemoryRouter><Dashboard /></MemoryRouter>)

    expect(await screen.findByText('2/3')).toBeInTheDocument() // summary still rendered
    expect(screen.getByText(/could not load recent analyses/i)).toBeInTheDocument()
  })

  it('shows an error state when the summary request fails', async () => {
    apiClient.get.mockRejectedValue(new Error('network error'))
    render(<MemoryRouter><Dashboard /></MemoryRouter>)
    expect(await screen.findByText(/failed to load dashboard data/i)).toBeInTheDocument()
  })
})
