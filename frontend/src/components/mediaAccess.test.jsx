import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import CameraStream from './CameraStream'
import MediaDownloadLink from './MediaDownloadLink'
import MediaVideo from './MediaVideo'
import apiClient from '../api/client'
import { useAuthStore } from '../store/authStore'
import { useToastStore } from '../store/toastStore'

vi.mock('../api/client', () => ({
  default: { get: vi.fn(), post: vi.fn(), delete: vi.fn() },
  API_URL: 'http://localhost:8000',
}))

// These components exist to keep the session JWT out of URLs. A browser
// cannot put a header on `<video src>`, `<img src>` or a download link, so
// something has to travel in the query string - and what travels there is
// a token scoped to one resource and valid for minutes, not the day-long
// credential that can delete the account.

function mintsTokens() {
  let n = 0
  apiClient.post.mockImplementation((url) => {
    if (url.endsWith('/media-token')) {
      n += 1
      return Promise.resolve({ data: { token: `scoped-${n}`, expires_in: 300 } })
    }
    return Promise.reject(new Error(`unexpected request: ${url}`))
  })
}

describe('media access', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAuthStore.setState({ token: 'session-jwt', user: { id: 1, role: 'viewer' } })
    useToastStore.setState({ toasts: [] })
  })

  describe('MediaVideo', () => {
    it('mints a clip-scoped token and never puts the session token in the URL', async () => {
      mintsTokens()
      render(<MediaVideo kind="recording" id={42} />)

      await waitFor(() => expect(document.querySelector('video')).not.toBeNull())
      const src = document.querySelector('video').getAttribute('src')

      expect(apiClient.post).toHaveBeenCalledWith('/api/recordings/42/media-token')
      expect(src).toBe('http://localhost:8000/api/recordings/42/video?token=scoped-1')
      expect(src).not.toContain('session-jwt')
    })

    it('retries once with a fresh token before reporting a failure', async () => {
      // A media token expires in minutes, so a page left open and then
      // played from will legitimately fail once. That is indistinguishable
      // from an unplayable file at the DOM level, so the first failure buys
      // a new token rather than an error message.
      mintsTokens()
      render(<MediaVideo kind="recording" id={42} />)
      await waitFor(() => expect(document.querySelector('video')).not.toBeNull())

      fireEvent.error(document.querySelector('video'))

      await waitFor(() => {
        expect(document.querySelector('video').getAttribute('src')).toContain('token=scoped-2')
      })
      expect(screen.queryByText(/could not be played/i)).not.toBeInTheDocument()
    })

    it('reports the failure when a fresh token does not help', async () => {
      mintsTokens()
      render(<MediaVideo kind="recording" id={42} />)
      await waitFor(() => expect(document.querySelector('video')).not.toBeNull())

      fireEvent.error(document.querySelector('video'))
      await waitFor(() => {
        expect(document.querySelector('video').getAttribute('src')).toContain('token=scoped-2')
      })
      fireEvent.error(document.querySelector('video'))

      expect(await screen.findByText(/could not be played/i)).toBeInTheDocument()
    })

    it('says so rather than rendering a dead player when no token can be obtained', async () => {
      apiClient.post.mockRejectedValue(new Error('403'))
      render(<MediaVideo kind="recording" id={42} />)

      expect(await screen.findByText(/could not be played/i)).toBeInTheDocument()
      expect(document.querySelector('video')).toBeNull()
    })
  })

  describe('MediaDownloadLink', () => {
    it('mints nothing until the link is actually clicked', async () => {
      mintsTokens()
      render(<MediaDownloadLink kind="recording" id={7} />)

      // A table of fifty recordings must not fire fifty mint requests for
      // rows nobody downloads - and a token minted at render time would be
      // minutes stale by the time anyone clicked it.
      expect(apiClient.post).not.toHaveBeenCalled()
      expect(screen.getByRole('link', { name: /download/i }).getAttribute('href')).not.toContain('token')
    })

    it('mints a scoped token on click and navigates to the download', async () => {
      const user = userEvent.setup()
      mintsTokens()
      // jsdom refuses real navigation; capture the assignment instead.
      const assigned = []
      delete window.location
      window.location = { get href() { return '' }, set href(v) { assigned.push(v) } }

      render(<MediaDownloadLink kind="recording" id={7} />)
      await user.click(screen.getByRole('link', { name: /download/i }))

      await waitFor(() => expect(assigned).toHaveLength(1))
      expect(apiClient.post).toHaveBeenCalledWith('/api/recordings/7/media-token')
      expect(assigned[0]).toBe('http://localhost:8000/api/recordings/7/download?token=scoped-1')
      expect(assigned[0]).not.toContain('session-jwt')
    })

    it('tells the user when the download cannot be started', async () => {
      const user = userEvent.setup()
      apiClient.post.mockRejectedValue(new Error('boom'))
      render(<MediaDownloadLink kind="recording" id={7} />)

      await user.click(screen.getByRole('link', { name: /download/i }))

      await waitFor(() => {
        expect(useToastStore.getState().toasts.some((t) => /could not start the download/i.test(t.message))).toBe(true)
      })
    })
  })

  describe('CameraStream', () => {
    it('mints a camera-scoped token for the MJPEG stream', async () => {
      mintsTokens()
      render(<CameraStream cameraId={3} alt="Lobby live stream" className="x" />)

      const img = await screen.findByAltText('Lobby live stream')

      expect(apiClient.post).toHaveBeenCalledWith('/api/cameras/3/media-token')
      expect(img.getAttribute('src')).toBe('http://localhost:8000/api/cameras/3/stream?token=scoped-1')
      expect(img.getAttribute('src')).not.toContain('session-jwt')
    })

    it('hands a persistent failure to the caller, which owns the fallback UI', async () => {
      mintsTokens()
      const onError = vi.fn()
      render(<CameraStream cameraId={3} alt="Lobby live stream" onError={onError} />)
      const img = await screen.findByAltText('Lobby live stream')

      // First drop: retried with a fresh token, caller not told.
      fireEvent.error(img)
      await waitFor(() => {
        expect(screen.getByAltText('Lobby live stream').getAttribute('src')).toContain('token=scoped-2')
      })
      expect(onError).not.toHaveBeenCalled()

      fireEvent.error(screen.getByAltText('Lobby live stream'))
      await waitFor(() => expect(onError).toHaveBeenCalled())
    })
  })
})
