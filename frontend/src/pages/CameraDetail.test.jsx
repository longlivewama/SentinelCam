import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import CameraDetail from './CameraDetail'
import apiClient from '../api/client'
import { useAuthStore } from '../store/authStore'
import { onRealtimeEvent } from '../lib/realtime'
import { pageOf } from '../test/apiFixtures'

vi.mock('../api/client', () => ({
  default: { get: vi.fn(), put: vi.fn(), delete: vi.fn() },
  API_URL: 'http://localhost:8000',
}))

vi.mock('../lib/realtime', () => ({
  onRealtimeEvent: vi.fn(() => () => {}),
}))

// The stream itself talks to a media-token endpoint and renders an <img>
// hitting a real MJPEG URL - irrelevant to this page's on/off and overlay
// logic, so it's stubbed to something inspectable instead.
vi.mock('../components/CameraStream', () => ({
  default: ({ cameraId }) => <div data-testid="camera-stream">stream:{cameraId}</div>,
}))

const BASE_CAMERA = {
  id: 5,
  name: 'Lobby',
  url: '0',
  camera_type: 'usb',
  location: 'Lobby',
  status: 'active',
  is_active: true,
  ai_detection_enabled: true,
  crowd_threshold: 20,
  abandoned_object_seconds: 30,
  created_at: '2026-01-01T00:00:00Z',
}

function renderDetail() {
  return render(
    <MemoryRouter initialEntries={['/cameras/5']}>
      <Routes>
        <Route path="/cameras/:id" element={<CameraDetail />} />
      </Routes>
    </MemoryRouter>,
  )
}

function mockGet(camera) {
  apiClient.get.mockImplementation((url) => {
    if (url === '/api/cameras/5') return Promise.resolve({ data: camera })
    if (url === '/api/recordings') return Promise.resolve({ data: pageOf([]) })
    return Promise.reject(new Error(`unexpected url ${url}`))
  })
}

describe('CameraDetail page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAuthStore.setState({ token: 'tok', user: { id: 1, role: 'operator' } })
  })

  it('shows the live stream and RUNNING fall detection when the camera is on', async () => {
    mockGet(BASE_CAMERA)
    renderDetail()

    expect(await screen.findByTestId('camera-stream')).toBeInTheDocument()
    expect(screen.getByText('RUNNING')).toBeInTheDocument()
    expect(screen.getAllByText('ON').length).toBeGreaterThan(0)
  })

  it('shows a clear "Camera Off" state instead of a broken stream when is_active is false', async () => {
    mockGet({ ...BASE_CAMERA, is_active: false })
    renderDetail()

    expect(await screen.findByText('Camera Off')).toBeInTheDocument()
    expect(screen.queryByTestId('camera-stream')).not.toBeInTheDocument()
    expect(screen.getByText('STOPPED')).toBeInTheDocument() // fall detection follows camera state
  })

  it('lets an operator switch the camera off via the dedicated toggle, reusing PUT /api/cameras/{id}', async () => {
    mockGet(BASE_CAMERA)
    apiClient.put.mockResolvedValue({ data: { ...BASE_CAMERA, is_active: false } })
    renderDetail()

    await screen.findByTestId('camera-stream')
    const cameraToggle = screen.getAllByRole('button', { name: 'ON' })[0]
    await userEvent.click(cameraToggle)

    await waitFor(() => {
      expect(apiClient.put).toHaveBeenCalledWith('/api/cameras/5', { is_active: false })
    })
    expect(await screen.findByText('Camera Off')).toBeInTheDocument()
  })

  it('draws the live fall box exactly where the backend reports it, only for this camera', async () => {
    mockGet(BASE_CAMERA)
    let deliver
    onRealtimeEvent.mockImplementation((cb) => {
      deliver = cb
      return () => {}
    })
    renderDetail()
    await screen.findByTestId('camera-stream')

    // A fall on a different camera must not show up here.
    act(() => {
      deliver({ type: 'fall.bbox', data: { camera_id: 99, bbox: [0, 0, 1, 1], frame_width: 10, frame_height: 10, confidence: 0.5 } })
    })
    expect(screen.queryByText(/FALL/)).not.toBeInTheDocument()

    act(() => {
      deliver({
        type: 'fall.bbox',
        data: { camera_id: 5, bbox: [10, 20, 110, 220], frame_width: 640, frame_height: 480, confidence: 0.87, track_id: 3 },
      })
    })
    expect(await screen.findByText(/FALL 87%/)).toBeInTheDocument()
    expect(screen.getByText(/ID 3/)).toBeInTheDocument()
  })
})
