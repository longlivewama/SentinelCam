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

describe('VideoUpload page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAuthStore.setState({ token: 'tok', user: { id: 1, email: 'a@example.com', role: 'viewer' } })
    useToastStore.setState({ toasts: [] })
  })

  it('shows an empty state when there are no uploads', async () => {
    apiClient.get.mockResolvedValueOnce({ data: [] })
    render(<VideoUpload />)
    expect(await screen.findByText(/no videos uploaded yet/i)).toBeInTheDocument()
  })

  it('shows a loading state before uploads resolve', () => {
    apiClient.get.mockReturnValueOnce(new Promise(() => {})) // never resolves
    render(<VideoUpload />)
    expect(screen.getByText(/loading uploads/i)).toBeInTheDocument()
  })

  it('shows an error state when fetching uploads fails', async () => {
    apiClient.get.mockRejectedValueOnce(new Error('network error'))
    render(<VideoUpload />)
    expect(await screen.findByText(/failed to load video uploads/i)).toBeInTheDocument()
  })

  it('lists existing uploads with their status and stats', async () => {
    apiClient.get.mockResolvedValueOnce({
      data: [
        {
          id: 1,
          original_filename: 'lobby-fall.mp4',
          status: 'completed',
          progress_percent: 100,
          fall_events_count: 2,
          persons_detected: 1,
          file_size_bytes: 1024 * 1024,
          created_at: '2026-01-01T00:00:00Z',
        },
      ],
    })
    render(<VideoUpload />)
    expect(await screen.findByText('lobby-fall.mp4')).toBeInTheDocument()
    expect(screen.getByText(/2 falls/i)).toBeInTheDocument()
    expect(screen.getByText('completed')).toBeInTheDocument()
  })

  it('uploads a selected video file and adds it to the list', async () => {
    const user = userEvent.setup()
    apiClient.get.mockResolvedValueOnce({ data: [] })
    apiClient.post.mockResolvedValueOnce({
      data: {
        id: 42,
        original_filename: 'test-clip.mp4',
        status: 'pending',
        progress_percent: 0,
        fall_events_count: 0,
        persons_detected: 0,
        file_size_bytes: 500,
        created_at: '2026-01-01T00:00:00Z',
      },
    })

    render(<VideoUpload />)
    await screen.findByText(/no videos uploaded yet/i)

    const file = new File(['fake video content'], 'test-clip.mp4', { type: 'video/mp4' })
    const input = document.querySelector('input[type="file"]')
    await user.upload(input, file)

    await waitFor(() => expect(apiClient.post).toHaveBeenCalledTimes(1))
    const [url, formData] = apiClient.post.mock.calls[0]
    expect(url).toBe('/api/video-uploads')
    expect(formData.get('file')).toBe(file)

    expect(await screen.findByText('test-clip.mp4')).toBeInTheDocument()
    expect(useToastStore.getState().toasts.some((t) => /analysis started/i.test(t.message))).toBe(true)
  })

  it('shows an error toast when upload fails', async () => {
    const user = userEvent.setup()
    apiClient.get.mockResolvedValueOnce({ data: [] })
    apiClient.post.mockRejectedValueOnce({ response: { data: { detail: 'Unsupported file type' } } })

    render(<VideoUpload />)
    await screen.findByText(/no videos uploaded yet/i)

    const file = new File(['not a video'], 'notes.txt', { type: 'video/mp4' })
    const input = document.querySelector('input[type="file"]')
    await user.upload(input, file)

    await waitFor(() => {
      expect(useToastStore.getState().toasts.some((t) => /unsupported file type/i.test(t.message))).toBe(true)
    })
  })

  it('deletes an upload after confirmation', async () => {
    const user = userEvent.setup()
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    apiClient.get.mockResolvedValueOnce({
      data: [
        {
          id: 7,
          original_filename: 'to-delete.mp4',
          status: 'completed',
          progress_percent: 100,
          fall_events_count: 0,
          persons_detected: 0,
          file_size_bytes: 100,
          created_at: '2026-01-01T00:00:00Z',
        },
      ],
    })
    apiClient.delete.mockResolvedValueOnce({})

    render(<VideoUpload />)
    await screen.findByText('to-delete.mp4')

    await user.click(screen.getByRole('button', { name: /delete/i }))

    await waitFor(() => expect(apiClient.delete).toHaveBeenCalledWith('/api/video-uploads/7'))
    await waitFor(() => expect(screen.queryByText('to-delete.mp4')).not.toBeInTheDocument())
  })
})
