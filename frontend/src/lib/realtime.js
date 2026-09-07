import { create } from 'zustand'
import { API_URL } from '../api/client'
import { useAuthStore } from '../store/authStore'

export const useRealtimeStatus = create(() => ({ connected: false }))

let socket = null
let intentionalClose = false
let reconnectAttempt = 0
let reconnectTimer = null
const listeners = new Set()

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

  socket.onclose = () => {
    useRealtimeStatus.setState({ connected: false })
    if (intentionalClose) return
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
