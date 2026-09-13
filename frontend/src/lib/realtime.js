import { create } from 'zustand'
import { API_URL } from '../api/client'
import { useAuthStore } from '../store/authStore'

export const useRealtimeStatus = create(() => ({ connected: false }))

let socket = null
let intentionalClose = false
let reconnectAttempt = 0
let reconnectTimer = null
const listeners = new Set()

// The backend distinguishes an authentication failure from a network drop
// by close code, specifically so this client can stop retrying a
// credential it now knows is dead (see backend/app/api/routes/realtime.py).
//
// 4403 is the one that matters in a browser: the socket was accepted and
// then closed at the token's own `exp`, so the code arrives on a live
// connection. Without this branch the client reconnected on a backoff with
// a token it had just been told was expired - forever, for as long as the
// tab stayed open - while the UI went on looking signed in. The session is
// genuinely over at that point, so end it the way any other expiry ends:
// drop the credential and let the route guards send the user to /login.
//
// 4401 is sent when the handshake itself is rejected. A browser generally
// sees that as a failed connection (1006) rather than the code, because
// the close happens before the upgrade completes - it is handled here for
// correctness, not because it is the common path. A 1006 is deliberately
// left to reconnect: it is indistinguishable from a network drop.
const WS_UNAUTHORIZED = 4401
const WS_TOKEN_EXPIRED = 4403

function wsUrl(token) {
  const url = new URL(API_URL)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  url.pathname = '/api/ws/events'
  url.search = `?token=${encodeURIComponent(token)}`
  return url.toString()
}

/** Subscribe to realtime events ({type, data}). Returns an unsubscribe function. */
export function onRealtimeEvent(callback) {
  listeners.add(callback)
  return () => listeners.delete(callback)
}

export function connectRealtime() {
  const token = useAuthStore.getState().token
  if (!token) return
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return

  intentionalClose = false
  socket = new WebSocket(wsUrl(token))

  socket.onopen = () => {
    reconnectAttempt = 0
    useRealtimeStatus.setState({ connected: true })
  }

  socket.onmessage = (event) => {
    let parsed
    try {
      parsed = JSON.parse(event.data)
    } catch {
      return
    }
    listeners.forEach((cb) => {
      try {
        cb(parsed)
      } catch (err) {
        console.error('Realtime listener error', err)
      }
    })
  }

  socket.onclose = (event) => {
    useRealtimeStatus.setState({ connected: false })
    if (intentionalClose) return

    if (event?.code === WS_UNAUTHORIZED || event?.code === WS_TOKEN_EXPIRED) {
      socket = null
      if (event.code === WS_TOKEN_EXPIRED) {
        useAuthStore.getState().logout()
      }
      return
    }

    reconnectAttempt += 1
    const delay = Math.min(1000 * 2 ** reconnectAttempt, 30000)
    reconnectTimer = setTimeout(connectRealtime, delay)
  }

  socket.onerror = () => {
    socket?.close()
  }
}

export function disconnectRealtime() {
  intentionalClose = true
  clearTimeout(reconnectTimer)
  socket?.close()
  socket = null
  useRealtimeStatus.setState({ connected: false })
}
