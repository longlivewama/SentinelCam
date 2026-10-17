import { formatSeconds } from '../lib/poseOverlay'

const REASONS = {
  pose_upright: 'withheld: person stayed upright after detection',
  duplicate_for_track: 'withheld: repeat for the same person',
}

// Who was tracked in an uploaded video, when, and which falls were
// attributed to them - all read from the analysis' own tracking record.
// Clicking a bar or a fall marker seeks the video above to that moment.
export default function TrackTimeline({ tracking, duration, onSeek }) {
  const tracks = [...(tracking?.tracks || [])].sort((a, b) => a.first_seen - b.first_seen)
  const span = Math.max(Number(duration) || 0, ...tracks.map((t) => t.last_seen), 0.001)
  const pct = (seconds) => `${Math.min(100, Math.max(0, (seconds / span) * 100))}%`

  const peak = (tracking?.frames || []).reduce(
    (max, frame) => Math.max(max, frame.persons.filter((p) => p.state !== 'lost').length),
    0,
  )
  const attributed = new Map()
  const unattributed = []
  for (const fall of tracking?.fall_events || []) {
    if (fall.person_track_id === null || fall.person_track_id === undefined) unattributed.push(fall)
    else attributed.set(fall.person_track_id, [...(attributed.get(fall.person_track_id) || []), fall])
  }
  const withheld = tracking?.suppressed_events || []

  if (tracks.length === 0) {
    return <p className="text-sm text-slate-500">No people were tracked in this video.</p>
  }

  return (
    <div>
      <p className="mb-3 text-xs text-slate-400">
        {tracks.length} track{tracks.length === 1 ? '' : 's'} · up to {peak} at once
        {tracking.truncated ? ' · record truncated for this long video' : ''}. Track IDs are temporary
        labels for this analysis, not identities - one person can get a new ID after leaving view.
      </p>
      <ul className="flex flex-col gap-2" aria-label="Tracked people">
        {tracks.map((track) => {
          const falls = attributed.get(track.track_id) || []
          return (
            <li key={track.track_id} className="grid grid-cols-[4.5rem_1fr] items-center gap-3 sm:grid-cols-[4.5rem_1fr_11rem]">
              <span className={`sc-badge justify-center border ${falls.length ? 'border-status-error/40 bg-status-error/10 text-status-error' : 'border-accent-cyan/30 bg-accent-cyan/10 text-accent-cyan'}`}>
                ID {track.track_id}
              </span>
              <div className="relative h-3 rounded bg-surface-800" data-testid={`track-bar-${track.track_id}`}>
                <button
                  type="button"
                  onClick={() => onSeek?.(track.first_seen)}
                  title={`Seen ${formatSeconds(track.first_seen)}–${formatSeconds(track.last_seen)}`}
                  aria-label={`Track ${track.track_id}: seen from ${formatSeconds(track.first_seen)} to ${formatSeconds(track.last_seen)}`}
                  className="absolute top-0 h-3 rounded bg-accent-cyan/50 hover:bg-accent-cyan/80"
                  style={{ left: pct(track.first_seen), width: `max(${pct(track.last_seen - track.first_seen)}, 3px)` }}
                />
                {falls.map((fall) => (
                  <button
                    key={`${fall.event_id}-${fall.video_time_seconds}`}
                    type="button"
                    onClick={() => onSeek?.(fall.video_time_seconds)}
                    aria-label={`Fall attributed to track ${track.track_id} at ${formatSeconds(fall.video_time_seconds)}`}
                    className="absolute -top-1 h-5 w-1.5 rounded bg-status-error"
                    style={{ left: pct(fall.video_time_seconds) }}
                  />
                ))}
              </div>
              <span className="col-span-2 text-xs text-slate-400 sm:col-span-1">
                {track.duration_seconds.toFixed(1)}s · {track.frames_observed} frame{track.frames_observed === 1 ? '' : 's'}
                {falls.length ? ` · ${falls.length} fall${falls.length === 1 ? '' : 's'}` : ''}
              </span>
            </li>
          )
        })}
      </ul>
      {unattributed.length > 0 && (
        <p className="mt-3 text-xs text-slate-400">
          {unattributed.length} fall event{unattributed.length === 1 ? ' was' : 's were'} not attributed to a
          tracked person (the detector&apos;s box did not clearly belong to anyone):{' '}
          {unattributed.map((fall, index) => (
            <button
              key={index}
              type="button"
              onClick={() => onSeek?.(fall.video_time_seconds)}
              className="mr-1 text-accent-cyan hover:underline"
            >
              {formatSeconds(fall.video_time_seconds)}
            </button>
          ))}
        </p>
      )}
      {withheld.length > 0 && (
        <p className="mt-2 text-xs text-amber-300/90">
          {withheld.length} detector event{withheld.length === 1 ? ' was' : 's were'} withheld by
          pose/tracking assistance:{' '}
          {withheld.map((event, index) => (
            <button
              key={index}
              type="button"
              onClick={() => onSeek?.(event.video_time_seconds)}
              className="mr-2 hover:underline"
              title={REASONS[event.reason] || event.reason}
            >
              {formatSeconds(event.video_time_seconds)} ({REASONS[event.reason] || event.reason})
            </button>
          ))}
        </p>
      )}
    </div>
  )
}
