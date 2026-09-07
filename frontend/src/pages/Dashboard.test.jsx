import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Dashboard from './Dashboard'
import apiClient from '../api/client'

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
      if (url === '/api/alerts') return Promise.resolve({ data: [] })
      return Promise.reject(new Error('unexpected url'))
    })

    render(<MemoryRouter><Dashboard /></MemoryRouter>)
    expect(screen.getByText(/loading dashboard/i)).toBeInTheDocument()

    expect(await screen.findByText('2/3')).toBeInTheDocument() // active/total cameras
    expect(screen.getByText('4')).toBeInTheDocument() // unacknowledged
    expect(screen.getByText('No alerts yet.')).toBeInTheDocument()
  })

  it('shows an error state when the summary request fails', async () => {
    apiClient.get.mockRejectedValue(new Error('network error'))
    render(<MemoryRouter><Dashboard /></MemoryRouter>)
    expect(await screen.findByText(/failed to load dashboard data/i)).toBeInTheDocument()
  })
})
