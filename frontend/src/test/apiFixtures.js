/**
 * Wraps rows in the paginated list envelope the API returns:
 *
 *   { items, page, page_size, total, pages }
 *
 * Tests care about the rows; the envelope is boilerplate that would
 * otherwise be re-typed at every mock and drift the moment the contract
 * changes. Overrides let a test that IS about paging set page/pages
 * explicitly.
 */
export function pageOf(items, overrides = {}) {
  const pageSize = overrides.page_size ?? 20
  const total = overrides.total ?? items.length
  return {
    items,
    page: 1,
    page_size: pageSize,
    total,
    pages: total ? Math.ceil(total / pageSize) : 0,
    ...overrides,
  }
}
