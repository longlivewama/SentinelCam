import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { connectRealtime, disconnectRealtime, useRealtimeStatus } from './realtime'
import { useAuthStore } from '../store/authStore'

// The backend closes the realtime socket with a distinct code when the
// problem is the credential rather than the network (see
// backend/app/api/routes/realtime.py):
//
//   4401 - the token was rejected at the handshake
//   4403 - the token expired while the socket was open
//
// It does that specifically so this client can stop. Reconnecting with
// exponential backoff on an auth failure re-presents a credential the
// server has already said is dead, forever, for as long as the tab is
// open - and on 4403 it does so while the user's session is actually over,
// leaving the UI looking signed in.
//
// 4403 is the case a browser actually observes, because the socket was
// accepted before it was closed. A rejected handshake usually reaches the
// client as 1006 instead, which is why 1006 must still reconnect: it is
// what an ordinary dropped connection looks like too.
const WS_UNAUTHORIZED = 4401
const WS_TOKEN_EXPIRED = 4403
const WS_ABNORMAL_CLOSURE = 1006 // what a dropped connection looks like

let sockets

class FakeWebSocket {
  constructor(url) {
    this.url = url
    this.readyState = FakeWebSocket.CONNECTING
    sockets.push(this)
  }

  close() {
    this.readyState = FakeWebSocket.CLOSED
  }

  // Drives the handler the module installed, the way the browser would.
  fireClose(code) {
    this.readyState = FakeWebSocket.CLOSED
    this.onclose?.({ code })
  }
}
FakeWebSocket.CONNECTING = 0
FakeWebSocket.OPEN = 1
FakeWebSocket.CLOSING = 2
FakeWebSocket.CLOSED = 3

describe('realtime socket lifecycle', () => {
  beforeEach(() => {
    sockets = []
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', FakeWebSocket)
    localStorage.clear()
    useAuthStore.setState({ token: 'a-session-token', user: { id: 1, role: 'viewer' } })
  })

  afterEach(() => {
    disconnectRealtime()
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('opens one socket carrying the session token', () => {
    connectRealtime()

    expect(sockets).toHaveLength(1)
    expect(sockets[0].url).toContain('/api/ws/events?token=a-session-token')
  })

  it('reconnects after an ordinary connection drop', () => {
    connectRealtime()
    sockets[0].fireClose(WS_ABNORMAL_CLOSURE)

    vi.advanceTimersByTime(60_000)

    expect(sockets.length).toBeGreaterThan(1)
  })

  it('stops reconnecting when the handshake was rejected', () => {
    connectRealtime()
    sockets[0].fireClose(WS_UNAUTHORIZED)

    vi.advanceTimersByTime(120_000)

    expect(sockets).toHaveLength(1)
    expect(useRealtimeStatus.getState().connected).toBe(false)
  })

  it('stops reconnecting when the token expired mid-connection', () => {
    connectRealtime()
    sockets[0].fireClose(WS_TOKEN_EXPIRED)

    vi.advanceTimersByTime(120_000)

    expect(sockets).toHaveLength(1)
  })

  it('ends the session when the token expired mid-connection', () => {
    connectRealtime()
    sockets[0].fireClose(WS_TOKEN_EXPIRED)

    // The socket closing at the token's own `exp` means every REST call
    // would 401 from here on; the UI should not keep looking signed in
    // until the user happens to make one.
    expect(useAuthStore.getState().token).toBeNull()
  })

  it('leaves the session alone when a handshake is merely rejected', () => {
    connectRealtime()
    sockets[0].fireClose(WS_UNAUTHORIZED)

    expect(useAuthStore.getState().token).toBe('a-session-token')
  })

  it('does not open a socket with no token', () => {
    useAuthStore.setState({ token: null, user: null })

    connectRealtime()

    expect(sockets).toHaveLength(0)
  })

  it('does not reconnect after an intentional disconnect', () => {
    connectRealtime()
    disconnectRealtime()
    vi.advanceTimersByTime(120_000)

    expect(sockets).toHaveLength(1)
  })
})
