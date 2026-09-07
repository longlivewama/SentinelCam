import { fireEvent, render, screen, waitFor } from '@testing-library/react'
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

  it('surfaces the server\'s rejection reason when an upload fails', async () => {
    const user = userEvent.setup()
    apiClient.get.mockResolvedValueOnce({ data: [] })
    apiClient.post.mockRejectedValueOnce({ response: { data: { detail: 'Unsupported file type' } } })

    render(<VideoUpload />)
    await screen.findByText(/no videos uploaded yet/i)

    // A file the client-side checks accept, so the request actually
    // reaches the (mocked) server and its detail message is what surfaces.
    const file = new File(['not really a video'], 'clip.mp4', { type: 'video/mp4' })
    const input = document.querySelector('input[type="file"]')
    await user.upload(input, file)

    await waitFor(() => {
      expect(useToastStore.getState().toasts.some((t) => /unsupported file type/i.test(t.message))).toBe(true)
    })
  })

  it('explains a network failure that never reached the server', async () => {
    const user = userEvent.setup()
    apiClient.get.mockResolvedValueOnce({ data: [] })
    apiClient.post.mockRejectedValueOnce(new Error('Network Error')) // no `response`

    render(<VideoUpload />)
    await screen.findByText(/no videos uploaded yet/i)

    const file = new File(['data'], 'clip.mp4', { type: 'video/mp4' })
    await user.upload(document.querySelector('input[type="file"]'), file)

    await waitFor(() => {
      expect(useToastStore.getState().toasts.some((t) => /could not reach the server/i.test(t.message))).toBe(true)
    })
  })

  it('rejects an unsupported file type dropped onto the drop zone', async () => {
    apiClient.get.mockResolvedValueOnce({ data: [] })

    render(<VideoUpload />)
    await screen.findByText(/no videos uploaded yet/i)

    // The file picker's `accept` attribute filters this out for us, but
    // drag-and-drop bypasses `accept` entirely - so this is the path that
    // actually needs the client-side check.
    const file = new File(['not a video'], 'notes.txt', { type: 'text/plain' })
    const dropZone = screen.getByText(/drag and drop a video file here/i).closest('div')
    fireEvent.drop(dropZone, { dataTransfer: { files: [file] } })

    await waitFor(() => {
      expect(useToastStore.getState().toasts.some((t) => /supported video formats/i.test(t.message))).toBe(true)
    })
    expect(apiClient.post).not.toHaveBeenCalled()
  })

  it('rejects a file over the size limit without uploading it', async () => {
    const user = userEvent.setup()
    apiClient.get.mockResolvedValueOnce({ data: [] })

    render(<VideoUpload />)
    await screen.findByText(/no videos uploaded yet/i)

    const file = new File(['x'], 'huge.mp4', { type: 'video/mp4' })
    Object.defineProperty(file, 'size', { value: 501 * 1024 * 1024 })
    await user.upload(document.querySelector('input[type="file"]'), file)

    await waitFor(() => {
      expect(useToastStore.getState().toasts.some((t) => /over the 500MB limit/i.test(t.message))).toBe(true)
    })
    expect(apiClient.post).not.toHaveBeenCalled()
  })

  it('blocks a second submission while an upload is still in flight', async () => {
    const user = userEvent.setup()
    apiClient.get.mockResolvedValueOnce({ data: [] })
    apiClient.post.mockReturnValueOnce(new Promise(() => {})) // never settles

    render(<VideoUpload />)
    await screen.findByText(/no videos uploaded yet/i)

    const input = document.querySelector('input[type="file"]')
    await user.upload(input, new File(['a'], 'first.mp4', { type: 'video/mp4' }))

    await waitFor(() => expect(apiClient.post).toHaveBeenCalledTimes(1))

    // The picker button and its input are both disabled for the duration,
    // so a second selection can't race the first request.
    expect(screen.getByRole('button', { name: /uploading/i })).toBeDisabled()
    expect(input).toBeDisabled()
    expect(apiClient.post).toHaveBeenCalledTimes(1)
  })

  it('shows why an analysis failed', async () => {
    apiClient.get.mockResolvedValueOnce({
      data: [
        {
          id: 9,
          original_filename: 'broken.mp4',
          status: 'failed',
          progress_percent: 0,
          error_message: 'Could not open uploaded video file - it may be corrupt or in an unsupported format',
          fall_events_count: 0,
          persons_detected: 0,
          file_size_bytes: 2048,
          created_at: '2026-01-01T00:00:00Z',
        },
      ],
    })

    render(<VideoUpload />)
    expect(await screen.findByText('broken.mp4')).toBeInTheDocument()
    expect(screen.getByText(/could not open uploaded video file/i)).toBeInTheDocument()
  })

  describe('results modal', () => {
    const COMPLETED_UPLOAD = {
      id: 5,
      original_filename: 'hallway.mp4',
      status: 'completed',
      progress_percent: 100,
      fall_events_count: 1,
      persons_detected: 3,
      duration_seconds: 12.5,
      file_size_bytes: 4096,
      created_at: '2026-01-01T00:00:00Z',
    }

    it('shows each fall with the confidence the backend actually reported', async () => {
      const user = userEvent.setup()
      apiClient.get.mockImplementation((url) => {
        if (url === '/api/video-uploads') return Promise.resolve({ data: [COMPLETED_UPLOAD] })
        if (url === '/api/recordings') {
          return Promise.resolve({ data: [{ id: 71, video_upload_id: 5, filename: 'fall.mp4' }] })
        }
        if (url === '/api/alerts') {
          return Promise.resolve({
            data: [
              {
                id: 900,
                video_upload_id: 5,
                recording_id: 71,
                event_type: 'fall',
                confidence_score: 0.87,
                timestamp: '2026-01-01T00:00:05Z',
              },
            ],
          })
        }
        return Promise.reject(new Error(`unexpected url ${url}`))
      })

      render(<VideoUpload />)
      await user.click(await screen.findByRole('button', { name: 'View' }))

      expect(await screen.findByText('Fall 1')).toBeInTheDocument()
      // 0.87 from the API, rendered as a percentage - not invented.
      expect(screen.getByText('87% confidence')).toBeInTheDocument()
      expect(screen.getByText('Persons detected')).toBeInTheDocument()
      expect(screen.queryByText('No fall events detected.')).not.toBeInTheDocument()
    })

    it('reports a failed results fetch instead of claiming no falls were found', async () => {
      const user = userEvent.setup()
      apiClient.get.mockImplementation((url) => {
        if (url === '/api/video-uploads') return Promise.resolve({ data: [COMPLETED_UPLOAD] })
        return Promise.reject(new Error('boom'))
      })

      render(<VideoUpload />)
      await user.click(await screen.findByRole('button', { name: 'View' }))

      expect(await screen.findByText(/could not load the fall events/i)).toBeInTheDocument()
      // The distinction matters: "no falls" is a detection result, and a
      // failed request must never be presented as one.
      expect(screen.queryByText('No fall events detected.')).not.toBeInTheDocument()
    })

    it('reports a genuine zero-fall result as such', async () => {
      const user = userEvent.setup()
      apiClient.get.mockImplementation((url) => {
        if (url === '/api/video-uploads') {
          return Promise.resolve({ data: [{ ...COMPLETED_UPLOAD, fall_events_count: 0 }] })
        }
        if (url === '/api/recordings' || url === '/api/alerts') return Promise.resolve({ data: [] })
        return Promise.reject(new Error(`unexpected url ${url}`))
      })

      render(<VideoUpload />)
      await user.click(await screen.findByRole('button', { name: 'View' }))

      expect(await screen.findByText('No fall events detected.')).toBeInTheDocument()
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
