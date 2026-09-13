import { useEffect, useRef, useState } from 'react'
import { buildMediaUrl, useMediaToken } from '../lib/mediaToken'

// A <video> that fetches its own short-lived media token and reports
// failure instead of rendering an empty black box.
//
// The failure UI exists because of a real incident: fall clips were being
// encoded as MPEG-4 Part 2 ("mp4v"), which no current browser can decode,
// while the H.264 source video played fine. The backend was healthy - the
// rows were right, the clips were on disk, and the range endpoint
// returned 206 - so the only symptom anyone could see was a silent blank
// player, which reads as "the results didn't load" rather than "this file
// can't be played". The encoder side is fixed (see recording_engine's
// codec ladder), but a player that fails silently would make the NEXT
// codec or storage problem just as hard to recognise.
//
// The retry is the other half. A media token expires in minutes, so a
// page left open and then played from will legitimately 401 once. That is
// indistinguishable from a broken file at the DOM level - both surface as
// an `error` event - so the first failure buys one fresh token and one
// more attempt, and only a second failure is reported to the user.
export default function MediaVideo({
  kind,
  id,
  action = 'video',
  videoRef,
  className = 'w-full rounded-lg bg-black',
  autoPlay = false,
}) {
  const { token, error: tokenError, refresh } = useMediaToken(kind, id)
  const [failed, setFailed] = useState(false)
  const retried = useRef(false)

  // A new source deserves a fresh attempt - otherwise one bad clip would
  // leave the player permanently marked as broken.
  useEffect(() => {
    setFailed(false)
    retried.current = false
  }, [kind, id])

  const handleError = async () => {
    if (retried.current) {
      setFailed(true)
      return
    }
    retried.current = true
    const fresh = await refresh()
    if (!fresh) setFailed(true)
  }

  if (failed || tokenError) {
    return (
      <div className="rounded-lg border border-status-error/30 bg-status-error/10 px-4 py-3 text-sm text-red-300">
        <p className="font-medium">This video could not be played.</p>
        <p className="mt-1 text-red-300/80">
          The file may be missing, or encoded in a format this browser cannot decode.
        </p>
      </div>
    )
  }

  if (!token) {
    return <div className={`${className} flex items-center justify-center text-sm text-slate-500`}>Loading video…</div>
  }

  return (
    <video
      ref={videoRef}
      controls
      autoPlay={autoPlay}
      preload="metadata"
      className={className}
      src={buildMediaUrl(kind, id, action, token)}
      onError={handleError}
    >
      Your browser does not support the video tag.
    </video>
  )
}
