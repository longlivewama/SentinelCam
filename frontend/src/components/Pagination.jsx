// Previous / next controls for the paged list endpoints.
//
// Deliberately previous/next rather than numbered page links: the API
// reports `pages`, so numbered links are possible, but a surveillance
// deployment reaches hundreds of pages quickly and a row of three-digit
// page numbers is neither usable nor mobile-safe. "Page 3 of 47" plus two
// buttons says the same thing in a fixed amount of space.
//
// Renders nothing at all when there is at most one page, so short lists -
// which is most of them - look exactly as they did before paging existed.
export default function Pagination({ page, pages, total, pageSize, onChange, busy = false, noun = 'item' }) {
  if (pages <= 1) return null

  const first = (page - 1) * pageSize + 1
  const last = Math.min(page * pageSize, total)

  return (
    <nav
      aria-label={`${noun} pagination`}
      className="mt-4 flex flex-wrap items-center justify-between gap-3"
    >
      <p className="text-xs text-slate-500">
        Showing {first}–{last} of {total}
      </p>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => onChange(page - 1)}
          disabled={busy || page <= 1}
          className="sc-btn-secondary px-3 py-1.5 text-xs disabled:cursor-not-allowed disabled:opacity-40"
        >
          Previous
        </button>
        <span className="whitespace-nowrap text-xs text-slate-400" aria-live="polite">
          Page {page} of {pages}
        </span>
        <button
          type="button"
          onClick={() => onChange(page + 1)}
          disabled={busy || page >= pages}
          className="sc-btn-secondary px-3 py-1.5 text-xs disabled:cursor-not-allowed disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </nav>
  )
}
