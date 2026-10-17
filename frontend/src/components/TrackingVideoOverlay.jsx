import { useEffect, useState } from 'react'
import { fallHighlightIds, frameAtTime, maxGapFor } from '../lib/poseOverlay'
import PoseOverlay from './PoseOverlay'

// Replays an uploaded video's tracking record over its own <video>, in step
// with playback: the frame shown is the processed frame at (or just before)
// the video's current time. Follows play, pause, seek and scrubbing. A person
// is highlighted only while their own attributed fall is in progress.
export default function TrackingVideoOverlay({ videoRef, tracking, options }) {
  const [currentTime, setCurrentTime] = useState(0)

  useEffect(() => {
    const video = videoRef?.current
    if (!video) return undefined
    let raf = null

    const sync = () => setCurrentTime(video.currentTime)
    const loop = () => {
      sync()
      raf = window.requestAnimationFrame(loop)
    }
    const start = () => {
      if (raf === null) raf = window.requestAnimationFrame(loop)
    }
    const stop = () => {
      if (raf !== null) window.cancelAnimationFrame(raf)
      raf = null
      sync()
    }

    video.addEventListener('play', start)
    video.addEventListener('pause', stop)
    video.addEventListener('ended', stop)
    video.addEventListener('seeked', sync)
    video.addEventListener('timeupdate', sync)
    sync()
    if (!video.paused) start()
    return () => {
      video.removeEventListener('play', start)
      video.removeEventListener('pause', stop)
      video.removeEventListener('ended', stop)
      video.removeEventListener('seeked', sync)
      video.removeEventListener('timeupdate', sync)
      if (raf !== null) window.cancelAnimationFrame(raf)
    }
  }, [videoRef, tracking])

  if (!tracking) return null
  const frame = frameAtTime(tracking.frames, currentTime, maxGapFor(tracking.video))

  return (
    <PoseOverlay
      persons={frame ? frame.persons : []}
      frameWidth={tracking.video.frame_width}
      frameHeight={tracking.video.frame_height}
      skeleton={tracking.pose_format.skeleton}
      keypointNames={tracking.pose_format.keypoint_names}
      keypointMinConfidence={tracking.keypoint_min_confidence}
      options={options}
      highlightTrackIds={fallHighlightIds(tracking.fall_events, currentTime)}
    />
  )
}
