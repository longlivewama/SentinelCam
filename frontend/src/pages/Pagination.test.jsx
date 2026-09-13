import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import Pagination from '../components/Pagination'
import Recordings from './Recordings'
import apiClient from '../api/client'
import { pageOf } from '../test/apiFixtures'
import { useAuthStore } from '../store/authStore'

vi.mock('../api/client', () => ({
  default: { get: vi.fn(), post: vi.fn(), delete: vi.fn() },
  API_URL: 'http://localhost:8000',
}))

const TOTAL = 45
const PAGE_SIZE = 20

function rowsFor(page) {
  const start = (page - 1) * PAGE_SIZE
  const count = Math.min(PAGE_SIZE, TOTAL - start)
  return Array.from({ length: Math.max(count, 0) }, (_, i) => ({
    id: start + i + 1,
    camera_id: 1,
    filename: `clip-${start + i + 1}.mp4`,
    duration_seconds: 6,
    trigger_action: 'fall',
    file_size_bytes: 1024,
    event_timestamp: '2026-01-01T00:00:00Z',
    created_at: '2026-01-01T00:00:00Z',
  }))
}

/** Serves real pages, so "next" has to actually request page 2 to see them. */
function serveRecordings({ onRequest } = {}) {
  apiClient.get.mockImplementation((url, config) => {
    if (url === '/api/cameras') {
      return Promise.resolve({ data: [{ id: 1, name: 'Lobby' }] })
    }
    if (url === '/api/recordings') {
      const params = config?.params ?? {}
      onRequest?.(params)
      const page = params.page ?? 1
      const filtered = params.trigger_action && params.trigger_action !== 'fall'
      const items = filtered ? [] : rowsFor(page)
      return Promise.resolve({
        data: pageOf(items, { page, total: filtered ? 0 : TOTAL, page_size: PAGE_SIZE }),
      })
    }
    return Promise.reject(new Error(`unexpected request: ${url}`))
  })
  apiClient.post.mockResolvedValue({ data: { token: 'scoped', expires_in: 300 } })
}

describe('Pagination control', () => {
  it('stays out of the way when everything fits on one page', () => {
    const { container } = render(
      <Pagination page={1} pages={1} total={7} pageSize={20} onChange={() => {}} />,
    )

    // Short lists - which is most of them - should look exactly as they did
    // before paging existed.
    expect(container).toBeEmptyDOMElement()
  })

  it('reports where the user is and how much there is', () => {
    render(<Pagination page={2} pages={3} total={45} pageSize={20} onChange={() => {}} />)

    expect(screen.getByText('Page 2 of 3')).toBeInTheDocument()
    expect(screen.getByText(/showing 21–40 of 45/i)).toBeInTheDocument()
  })

  it('counts the final partial page correctly', () => {
    render(<Pagination page={3} pages={3} total={45} pageSize={20} onChange={() => {}} />)

    expect(screen.getByText(/showing 41–45 of 45/i)).toBeInTheDocument()
  })

  it('disables previous on the first page and next on the last', () => {
    const { rerender } = render(
      <Pagination page={1} pages={3} total={45} pageSize={20} onChange={() => {}} />,
    )
    expect(screen.getByRole('button', { name: /previous/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /next/i })).toBeEnabled()

    rerender(<Pagination page={3} pages={3} total={45} pageSize={20} onChange={() => {}} />)
    expect(screen.getByRole('button', { name: /previous/i })).toBeEnabled()
    expect(screen.getByRole('button', { name: /next/i })).toBeDisabled()
  })

  it('disables both while a page is in flight', () => {
    render(<Pagination page={2} pages={3} total={45} pageSize={20} onChange={() => {}} busy />)

    expect(screen.getByRole('button', { name: /previous/i })).toBeDisabled()
    expect(screen.getByRole('button', { name: /next/i })).toBeDisabled()
  })
})

describe('Recordings page paging', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAuthStore.setState({ token: 'session-jwt', user: { id: 1, role: 'viewer' } })
  })

  it('shows a loading state, then the first page and the true total', async () => {
    serveRecordings()
    render(<Recordings />)

    expect(screen.getByText(/loading recordings/i)).toBeInTheDocument()

    expect(await screen.findByText('45 recordings')).toBeInTheDocument()
    expect(document.querySelectorAll('tbody tr')).toHaveLength(PAGE_SIZE)
    expect(screen.getByText('Page 1 of 3')).toBeInTheDocument()
  })

  it('fetches the next page from the server rather than slicing what it has', async () => {
    const user = userEvent.setup()
    const requests = []
    serveRecordings({ onRequest: (params) => requests.push(params) })
    render(<Recordings />)
    await screen.findByText('Page 1 of 3')

    await user.click(screen.getByRole('button', { name: /next/i }))

    await waitFor(() => expect(screen.getByText('Page 2 of 3')).toBeInTheDocument())
    expect(requests.at(-1).page).toBe(2)
    // Row 21 exists only on page 2, so its presence proves the second
    // request's rows were rendered.
    expect(await screen.findByText(/showing 21–40 of 45/i)).toBeInTheDocument()
  })

  it('goes back to the previous page', async () => {
    const user = userEvent.setup()
    serveRecordings()
    render(<Recordings />)
    await screen.findByText('Page 1 of 3')

    await user.click(screen.getByRole('button', { name: /next/i }))
    await waitFor(() => expect(screen.getByText('Page 2 of 3')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: /previous/i }))

    await waitFor(() => expect(screen.getByText('Page 1 of 3')).toBeInTheDocument())
  })

  it('keeps the filter applied when paging, and sends it to the server', async () => {
    const user = userEvent.setup()
    const requests = []
    serveRecordings({ onRequest: (params) => requests.push(params) })
    render(<Recordings />)
    await screen.findByText('Page 1 of 3')

    await user.selectOptions(screen.getByLabelText(/camera/i), '1')
    await waitFor(() => expect(requests.at(-1).camera_id).toBe('1'))
    await user.click(screen.getByRole('button', { name: /next/i }))

    await waitFor(() => expect(requests.at(-1).page).toBe(2))
    // The filter must travel with the page - a "next" that dropped it would
    // silently show unfiltered rows.
    expect(requests.at(-1).camera_id).toBe('1')
  })

  it('returns to page 1 when a filter changes', async () => {
    const user = userEvent.setup()
    const requests = []
    serveRecordings({ onRequest: (params) => requests.push(params) })
    render(<Recordings />)
    await screen.findByText('Page 1 of 3')

    await user.click(screen.getByRole('button', { name: /next/i }))
    await waitFor(() => expect(screen.getByText('Page 2 of 3')).toBeInTheDocument())
    await user.selectOptions(screen.getByLabelText(/camera/i), '1')

    // Page 4 of the unfiltered list is very unlikely to exist in the
    // filtered one, so staying put would show an empty table.
    await waitFor(() => expect(requests.at(-1).page).toBe(1))
  })

  it('filters on the server, not over the rows that happen to be on screen', async () => {
    const user = userEvent.setup()
    const requests = []
    serveRecordings({ onRequest: (params) => requests.push(params) })
    render(<Recordings />)
    await screen.findByText('Page 1 of 3')

    await user.selectOptions(screen.getByLabelText(/event type/i), 'crowd')

    await waitFor(() => expect(requests.at(-1).trigger_action).toBe('crowd'))
    expect(await screen.findByText(/no recordings found/i)).toBeInTheDocument()
  })

  it('shows an empty state and no pager for an empty collection', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/api/cameras') return Promise.resolve({ data: [] })
      return Promise.resolve({ data: pageOf([]) })
    })
    render(<Recordings />)

    expect(await screen.findByText(/no recordings found/i)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /next/i })).not.toBeInTheDocument()
    expect(screen.getByText('0 recordings')).toBeInTheDocument()
  })

  it('reports an error instead of silently showing an empty list', async () => {
    apiClient.get.mockImplementation((url) => {
      if (url === '/api/cameras') return Promise.resolve({ data: [] })
      return Promise.reject(new Error('boom'))
    })
    render(<Recordings />)

    expect(await screen.findByText(/failed to load recordings/i)).toBeInTheDocument()
  })

  it('lays the pager out so it wraps rather than overflowing a phone', async () => {
    serveRecordings()
    render(<Recordings />)
    await screen.findByText('Page 1 of 3')

    const nav = screen.getByRole('navigation', { name: /recordings pagination/i })
    expect(within(nav).getByText('Page 1 of 3')).toBeInTheDocument()
    expect(nav.className).toContain('flex-wrap')
  })
})
