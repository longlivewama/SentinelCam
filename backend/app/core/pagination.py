"""
Offset pagination for the collections whose history grows without bound.

WHY THIS REPLACES `limit`
-------------------------
The list endpoints previously took a `limit` (defaulting to 200-500) and
returned the newest N rows. That bounds the response size, which is the
half of the problem that can hurt the server - but it is not pagination:
there was no way to ask for row 501. A deployment recording falls for six
months had older events that no API call could reach, and the note in
recordings.py said as much ("real pagination is on the roadmap").

WHAT THE CONTRACT IS
--------------------
    {"items": [...], "page": 1, "page_size": 20, "total": 137, "pages": 7}

`total` and `pages` are what let a UI say "page 3 of 7" and disable
"next" on the last page, which is the difference between pagination a
user can navigate and a cursor they can only follow forwards. They cost
one extra COUNT per request; on an indexed column that is cheap, and the
alternative - a UI that cannot tell the user how much there is - is
worse.

`pages` is 0 for an empty collection, not 1. An empty collection has no
pages, and reporting 1 makes "page 1 of 1" appear over an empty table.

ORDERING MUST BE TOTAL
----------------------
Offset pagination is only correct if the sort is deterministic. Ordering
by a timestamp alone is not: two rows sharing a timestamp may come back
in either order, and Postgres is free to choose differently between the
query for page 1 and the query for page 2 - so a row can appear on both
pages while another appears on neither. Fall events are exactly the case
where this bites, because a clip and its alert are written in the same
transaction and several can land inside the same second.

So every caller passes the primary key as a final tiebreaker:

    ORDER BY event_timestamp DESC, id DESC

`paginate` cannot enforce that on the caller's behalf - it does not know
which model it is ordering - but every call site here does it, and
test_pagination.py pins the behaviour with rows that deliberately share a
timestamp.

WHAT IS NOT PAGINATED
---------------------
Cameras (bounded by how many are physically installed) and the analytics
aggregates (which summarise rather than list). Adding paging there would
be churn for collections that cannot grow unbounded.
"""
from __future__ import annotations

from math import ceil
from typing import Generic, List, Sequence, TypeVar

from fastapi import Query
from pydantic import BaseModel, ConfigDict

T = TypeVar("T")

# 20 fits a screen without scrolling on a laptop and keeps a page of
# recordings well under 100 KB.
DEFAULT_PAGE_SIZE = 20

# The ceiling exists so one request cannot ask for the whole table and
# undo the point of paginating. A client wanting everything pages through
# it; 100 rows keeps that to a reasonable number of requests without
# letting a single response get large.
MAX_PAGE_SIZE = 100


class PageParams:
    """`?page=&page_size=`, validated by FastAPI.

    `ge=1` on both means page 0, a negative page and an oversized
    page_size are all rejected as 422 by the framework, before any query
    runs - rather than being silently clamped, which would answer a
    different question from the one asked."""

    def __init__(
        self,
        page: int = Query(default=1, ge=1, description="1-based page number"),
        page_size: int = Query(
            default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE,
            description=f"Rows per page (max {MAX_PAGE_SIZE})",
        ),
    ):
        self.page = page
        self.page_size = page_size

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class Page(BaseModel, Generic[T]):
    model_config = ConfigDict(from_attributes=True)

    items: List[T]
    page: int
    page_size: int
    total: int
    pages: int


def paginate(query, params: PageParams, *order_by) -> dict:
    """Runs `query` for one page, at the database.

    Both halves are pushed to Postgres: a COUNT for the total, and
    OFFSET/LIMIT for the rows. Nothing fetches the full result set and
    slices it in Python - that would make every request cost as much as
    the whole table, which is the failure mode paginating is meant to
    prevent.

    `query` must already carry its ownership filter. Applying the scope
    before this call (rather than passing a user in here) keeps
    authorization visible at the route, and means a page's `total`
    counts only rows the caller may see.

    A page past the end returns no items rather than a 404: "there is
    nothing on page 9" is a valid answer to a valid question, and a
    client that just deleted the last row on page 9 should see an empty
    page, not an error.
    """
    # order_by(None) drops any ordering the caller applied before
    # counting - sorting rows only to count them is wasted work, and some
    # planners will not elide it on their own.
    total = query.order_by(None).count()

    items: Sequence = (
        query.order_by(*order_by).offset(params.offset).limit(params.page_size).all()
    )

    return {
        "items": items,
        "page": params.page,
        "page_size": params.page_size,
        "total": total,
        "pages": ceil(total / params.page_size) if total else 0,
    }
