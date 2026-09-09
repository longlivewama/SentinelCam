import { useCallback, useEffect, useRef, useState } from 'react'
import apiClient, { API_URL } from '../api/client'

// Media URLs are the one place this app cannot use an Authorization
// header: `<video src>`, `<img src>` and a plain download link all issue
// their request without one. Something therefore has to travel in the
// URL, and a URL is written down everywhere - the access log of every
// proxy on the path, browser history, any `Referer` sent onward.
//
// So what travels there is not the session token. It is a media token:
// minted per resource, expiring in minutes, carrying no role, and refused
// by every endpoint except the single clip, upload or camera it names
// (see backend app/core/deps.py). Recovering one from a log buys an
// attacker the ability to re-watch the clip whose URL they already had.
//
// The cost is one extra request before playback, which is what this
// module hides.

const MEDIA_PATHS = {
  recording: 'recordings',
  upload: 'video-uploads',
  camera: 'cameras',
}

export function mediaTokenPath(kind, id) {
  return `/api/${MEDIA_PATHS[kind]}/${id}/media-token`
}

export function buildMediaUrl(kind, id, action, token) {
  return `${API_URL}/api/${MEDIA_PATHS[kind]}/${id}/${action}?token=${encodeURIComponent(token)}`
}

/** Mints a media token for one resource. Throws on failure. */
export async function mintMediaToken(kind, id) {
  const { data } = await apiClient.post(mediaTokenPath(kind, id))
  return data.token
}

/**
 * A media token for `kind`/`id`, minted on mount and re-mintable.
 *
 * `refresh` exists because tokens expire while a page stays open: a user
 * who leaves the results page up for ten minutes and then presses play
 * would otherwise get a silent failure. Media elements report that as an
 * `error` event, so the players below call `refresh` once on error and
 * try again before showing anything to the user - a fresh token fixes an
 * expiry, and a second failure means the file really is unplayable.
 */
export function useMediaToken(kind, id) {
  const [token, setToken] = useState(null)
  const [error, setError] = useState(false)
  // Guards against a response for a previous id landing after a newer
  // one and pointing the player at the wrong resource.
  const requestSeq = useRef(0)

  const refresh = useCallback(async () => {
    if (id === null || id === undefined) return null
    const seq = (requestSeq.current += 1)
    try {
      const fresh = await mintMediaToken(kind, id)
      if (seq !== requestSeq.current) return null
      setToken(fresh)
      setError(false)
      return fresh
    } catch {
      if (seq !== requestSeq.current) return null
      setError(true)
      return null
    }
  }, [kind, id])

  useEffect(() => {
    refresh()
  }, [refresh])

  return { token, error, refresh }
}
