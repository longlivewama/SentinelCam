import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import SystemStatus from './SystemStatus'
import apiClient from '../api/client'

vi.mock('../api/client', () => ({
  default: { get: vi.fn() },
  API_URL: 'http://localhost:8000',
}))

function statusPayload(overrides = {}) {
  return {
    environment: 'production',
    model_device: 'cpu',
    models: {
      pose_model: 'yolov8n-pose.pt',
      object_model: 'yolov8n.pt',
      pose_object_models_loaded: true,
      violence_model_configured: false,
      fall_classifier_configured: false,
      fall_classifier_loaded: false,
      fall_detector: {
        configured_path: '/app/ml/exported/fall_detector_v1.pt',
        loaded: true,
        class_names: { 0: 'Fall' },
        min_confidence: 0.4,
        error: null,
      },
      ...(overrides.models || {}),
    },
    detection: {
      active_camera_detection_loops: 0,
      detection_frame_stride: 3,
      video_analysis_frame_stride: 5,
      fall_detection_mode_configured: 'auto',
      fall_detection_mode_active: 'model',
      max_concurrent_video_analyses: 2,
      ...(overrides.detection || {}),
    },
    notifications: { channels_enabled: ['email'], alert_recipients_configured: true },
    storage: { recordings_dir_writable: true, uploads_dir_writable: true },
  }
}

describe('SystemStatus page', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('reports the trained fall detector as the active strategy', async () => {
    apiClient.get.mockResolvedValue({ data: statusPayload() })
    render(<SystemStatus />)

    expect(await screen.findByText(/trained fall detector/i)).toBeInTheDocument()
    expect(screen.getByText(/trained model/i)).toBeInTheDocument()
    expect(screen.getByText('Fall')).toBeInTheDocument()
  })

  it('warns when an auto deployment has silently fallen back to the heuristic', async () => {
    // The failure this exists to surface: configured to prefer the model,
    // running without it, and otherwise indistinguishable from healthy.
    apiClient.get.mockResolvedValue({
      data: statusPayload({
        models: {
          fall_detector: {
            configured_path: '/app/ml/exported/fall_detector_v1.pt',
            loaded: false,
            class_names: null,
            min_confidence: 0.4,
            error: 'model file not found: /app/ml/exported/fall_detector_v1.pt',
          },
        },
        detection: { fall_detection_mode_active: 'heuristic', fall_detection_mode_configured: 'auto' },
      }),
    })
    render(<SystemStatus />)

    expect(await screen.findByText(/running on the pose heuristic, not the trained model/i)).toBeInTheDocument()
    expect(screen.getByText(/model file not found/i)).toBeInTheDocument()
  })

  it('does not warn when the heuristic was deliberately configured', async () => {
    apiClient.get.mockResolvedValue({
      data: statusPayload({
        models: { fall_detector: { configured_path: null, loaded: false, class_names: null, min_confidence: 0.4, error: null } },
        detection: { fall_detection_mode_active: 'heuristic', fall_detection_mode_configured: 'heuristic' },
      }),
    })
    render(<SystemStatus />)

    await screen.findByText(/trained fall detector/i)
    expect(screen.queryByText(/running on the pose heuristic, not the trained model/i)).not.toBeInTheDocument()
  })

  it('shows an error state when the status request fails', async () => {
    apiClient.get.mockRejectedValue(new Error('boom'))
    render(<SystemStatus />)
    expect(await screen.findByText(/failed to load system status/i)).toBeInTheDocument()
  })
})
