import { useEffect, useState } from 'react'
import { onRealtimeEvent } from './realtime'

// A camera's tracked people, as pushed by its detection loop over the
// realtime socket (`tracks.update`, see backend detection/engine.py).
//
// The feed exists only while AI detection runs for that camera - the
// backend adds no inference for the overlay, it publishes what detection
// already computed. Updates arrive once per processed frame; if they stop
// (camera off, detection disabled, socket down) the last frame is dropped
// after STALE_MS rather than left frozen over a live picture.
export const STALE_MS = 1500

export function useLiveTracks(cameraId, enabled) {
  const [scene, setScene] = useState(null)

  useEffect(() => {
    if (!enabled) {
      setScene(null)
      return undefined
    }
    const unsubscribe = onRealtimeEvent((message) => {
      if (message?.type !== 'tracks.update' || message?.data?.camera_id !== Number(cameraId)) return
      if (!message.data.active) {
        setScene(null)
        return
      }
      setScene({ ...message.data, receivedAt: Date.now() })
    })
    const timer = setInterval(() => {
      setScene((current) => (current && Date.now() - current.receivedAt > STALE_MS ? null : current))
    }, 500)
    return () => {
      unsubscribe()
      clearInterval(timer)
      setScene(null)
    }
  }, [cameraId, enabled])

  return scene
}
