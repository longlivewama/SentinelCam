import { useEffect, useRef } from 'react'
import { buildMediaUrl, useMediaToken } from '../lib/mediaToken'

// The MJPEG live view. `<img src>` cannot send an Authorization header,
// so this fetches a camera-scoped media token first - see lib/mediaToken.
//
// Expiry matters less here than for a clip: the stream is one long-lived
// response, authorized when it opens, so a token that expires mid-view
// does not interrupt anything. It matters when the connection drops and
// the browser retries with a URL whose token has since died, which is
// what the single refresh-then-retry below covers. A second failure is
// passed to the caller, which owns the "stream unavailable" presentation
// (it differs between the grid tile and the detail page).
export default function CameraStream({ cameraId, alt, className, onError }) {
  const { token, error: tokenError, refresh } = useMediaToken('camera', cameraId)
  const retried = useRef(false)

  useEffect(() => {
    retried.current = false
  }, [cameraId])

  const handleError = async (event) => {
    if (retried.current) {
      onError?.(event)
      return
    }
    retried.current = true
    // React pools nothing in v17+, but the element is needed after the
    // await, so hold it rather than the event.
    const element = event.currentTarget
    const fresh = await refresh()
    if (!fresh) onError?.({ currentTarget: element })
  }

  if (tokenError || !token) {
    // No token yet (or none obtainable): render nothing rather than an
    // <img> with a broken src, which would fire its own error event and
    // race the retry above.
    return null
  }

  return (
    <img
      src={buildMediaUrl('camera', cameraId, 'stream', token)}
      alt={alt}
      className={className}
      onError={handleError}
    />
  )
}
