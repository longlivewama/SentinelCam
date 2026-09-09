import { useState } from 'react'
import { buildMediaUrl, mintMediaToken } from '../lib/mediaToken'
import { toast } from '../store/toastStore'

// A download link that mints its media token when it is clicked, rather
// than when the row renders.
//
// Two reasons for on-click rather than on-render. A download URL is the
// one media URL that reaches browser history, so it should carry the
// freshest and shortest-lived credential available - a token minted when
// the list loaded could be minutes old by the time anyone clicks it. And
// a table of fifty recordings would otherwise fire fifty mint requests
// for rows nobody is going to download.
//
// It stays an <a> rather than becoming a button: the appearance,
// keyboard behaviour and screen-reader semantics of a download link are
// all correct already, and only the href needs deferring.
export default function MediaDownloadLink({
  kind,
  id,
  className = 'sc-btn-secondary px-3 py-1.5 text-xs',
  children = 'Download',
}) {
  const [busy, setBusy] = useState(false)

  const handleClick = async (event) => {
    event.preventDefault()
    if (busy) return
    setBusy(true)
    try {
      const token = await mintMediaToken(kind, id)
      // Content-Disposition on the response is `attachment`, so the
      // browser saves the file instead of navigating away from the SPA.
      window.location.href = buildMediaUrl(kind, id, 'download', token)
    } catch {
      toast.error('Could not start the download. Please try again.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <a href="#download" onClick={handleClick} aria-busy={busy} className={className}>
      {busy ? 'Preparing…' : children}
    </a>
  )
}
