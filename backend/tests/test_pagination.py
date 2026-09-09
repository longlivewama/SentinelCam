"""
Pagination across the collections whose history grows without bound.

The endpoints used to take a `limit` and return the newest N rows. That
bounded the response, but there was no way to ask for row N+1 - a
deployment recording falls for months had older events no API call could
reach. These tests pin the replacement.

Two things get more attention than the arithmetic, because they are the
two ways offset pagination goes quietly wrong:

  * TOTAL ORDERING. Ordering by a timestamp alone is not deterministic
    when rows share one, and Postgres may order them differently between
    the query for page 1 and the query for page 2 - so a row appears
    twice and another never appears at all. Fall events are exactly this
    case: an Event and its Recording are written in one transaction, and
    several land inside the same microsecond. `test_equal_timestamps_*`
    creates rows with identical timestamps on purpose.

  * SCOPE. `total` is computed from the same query the rows come from, so
    an ownership filter that stopped applying would show up as a count of
    other people's rows even when the page itself looked empty.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.core.pagination import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from app.models.camera import Camera
from app.models.event import Event
from app.models.recording import Recording
from app.models.video_upload import VideoUpload

TOTAL_ROWS = 25  # more than one default page, less than two


@pytest.fixture()
def camera(db):
    row = Camera(name="Cam A", url="0", camera_type="usb")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@pytest.fixture()
def alerts(db, camera):
    """25 camera alerts, one second apart, newest last in creation order."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        Event(
            camera_id=camera.id, event_type="fall", confidence_score=0.9,
            timestamp=base + timedelta(seconds=i), triggered_recording=True,
        )
        for i in range(TOTAL_ROWS)
    ]
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows


@pytest.fixture()
def recordings(db, camera, tmp_path):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(TOTAL_ROWS):
        path = tmp_path / f"clip{i}.mp4"
        path.write_bytes(b"x" * 16)
        rows.append(Recording(
            camera_id=camera.id, filename=path.name, file_path=str(path),
            duration_seconds=2.0, trigger_action="fall", file_size_bytes=16,
            event_timestamp=base + timedelta(seconds=i),
        ))
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows


def page(client, path, headers, **params):
    response = client.get(path, params=params, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


# --- the contract ---------------------------------------------------------

@pytest.mark.parametrize("path", ["/api/alerts", "/api/recordings", "/api/video-uploads"])
def test_the_envelope_shape_is_the_same_everywhere(client, viewer_headers, path):
    """One contract across every paginated collection - a client that can
    render one list can render them all."""
    body = page(client, path, viewer_headers)

    assert set(body) == {"items", "page", "page_size", "total", "pages"}
    assert isinstance(body["items"], list)
    assert body["page"] == 1
    assert body["page_size"] == DEFAULT_PAGE_SIZE


def test_defaults_return_the_first_page(client, viewer_headers, alerts):
    body = page(client, "/api/alerts", viewer_headers)

    assert body["page"] == 1
    assert body["page_size"] == DEFAULT_PAGE_SIZE
    assert body["total"] == TOTAL_ROWS
    assert body["pages"] == 2
    assert len(body["items"]) == DEFAULT_PAGE_SIZE


def test_page_two_returns_the_remainder(client, viewer_headers, alerts):
    body = page(client, "/api/alerts", viewer_headers, page=2)

    assert body["page"] == 2
    assert len(body["items"]) == TOTAL_ROWS - DEFAULT_PAGE_SIZE
    assert body["total"] == TOTAL_ROWS


def test_the_last_page_is_the_last(client, viewer_headers, alerts):
    body = page(client, "/api/alerts", viewer_headers, page=3, page_size=10)

    assert body["pages"] == 3
    assert body["page"] == 3
    assert len(body["items"]) == 5


def test_a_page_past_the_end_is_empty_rather_than_an_error(client, viewer_headers, alerts):
    """A client that deleted the last row on page 9 should see an empty
    page, not a 404 - and `total` still tells it where to go instead."""
    body = page(client, "/api/alerts", viewer_headers, page=99)

    assert body["items"] == []
    assert body["total"] == TOTAL_ROWS
    assert body["pages"] == 2


def test_an_empty_collection_reports_no_pages(client, viewer_headers):
    """`pages: 0`, not 1. Reporting 1 makes a UI render "page 1 of 1" over
    an empty table."""
    body = page(client, "/api/alerts", viewer_headers)

    assert body == {"items": [], "page": 1, "page_size": DEFAULT_PAGE_SIZE, "total": 0, "pages": 0}


# --- rejected input -------------------------------------------------------

@pytest.mark.parametrize("path", ["/api/alerts", "/api/recordings", "/api/video-uploads"])
@pytest.mark.parametrize("params", [
    {"page": 0},
    {"page": -1},
    {"page": "abc"},
    {"page_size": 0},
    {"page_size": -5},
    {"page_size": MAX_PAGE_SIZE + 1},
    {"page_size": "lots"},
])
def test_invalid_paging_is_rejected_not_clamped(client, viewer_headers, path, params):
    """422 rather than a silent clamp: answering a different question from
    the one asked is how a client ends up paging through a list it thinks
    is ordered differently than it is."""
    assert client.get(path, params=params, headers=viewer_headers).status_code == 422


def test_the_maximum_page_size_is_accepted(client, viewer_headers, alerts):
    body = page(client, "/api/alerts", viewer_headers, page_size=MAX_PAGE_SIZE)

    assert body["page_size"] == MAX_PAGE_SIZE
    assert len(body["items"]) == TOTAL_ROWS
    assert body["pages"] == 1


# --- ordering -------------------------------------------------------------

def test_alerts_are_newest_first(client, viewer_headers, alerts):
    body = page(client, "/api/alerts", viewer_headers, page_size=5)

    stamps = [item["timestamp"] for item in body["items"]]
    assert stamps == sorted(stamps, reverse=True)


def test_recordings_are_newest_first(client, viewer_headers, recordings):
    body = page(client, "/api/recordings", viewer_headers, page_size=5)

    stamps = [item["event_timestamp"] for item in body["items"]]
    assert stamps == sorted(stamps, reverse=True)


def test_paging_covers_every_row_exactly_once(client, viewer_headers, alerts):
    """The property that actually matters. Walking the pages must yield
    each row once - no duplicates across a page boundary, nothing skipped."""
    seen = []
    for number in range(1, 6):
        seen.extend(item["id"] for item in page(
            client, "/api/alerts", viewer_headers, page=number, page_size=5,
        )["items"])

    assert len(seen) == TOTAL_ROWS
    assert len(set(seen)) == TOTAL_ROWS
    assert set(seen) == {a.id for a in alerts}


def test_equal_timestamps_still_page_without_duplicates(client, viewer_headers, db, camera):
    """The case a timestamp-only sort gets wrong. Every row here shares one
    timestamp to the microsecond - which is what a fall's Event and
    Recording, written in the same transaction, actually look like. Without
    the id tiebreaker Postgres is free to return them in a different order
    for each page's query, so a row lands on two pages and another on none."""
    same_moment = datetime(2026, 2, 2, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        Event(
            camera_id=camera.id, event_type="fall", confidence_score=0.5,
            timestamp=same_moment, triggered_recording=True,
        )
        for _ in range(TOTAL_ROWS)
    ]
    db.add_all(rows)
    db.commit()

    seen = []
    for number in range(1, 6):
        seen.extend(item["id"] for item in page(
            client, "/api/alerts", viewer_headers, page=number, page_size=5,
        )["items"])

    assert len(seen) == TOTAL_ROWS, "a row was dropped between pages"
    assert len(set(seen)) == TOTAL_ROWS, "a row appeared on two pages"


def test_equal_timestamps_are_ordered_by_id_descending(client, viewer_headers, db, camera):
    """And the tiebreak is not merely stable, it is the documented one:
    newest id first, so ties read in the same direction as the timestamp."""
    same_moment = datetime(2026, 2, 2, 12, 0, 0, tzinfo=timezone.utc)
    rows = [
        Event(
            camera_id=camera.id, event_type="fall", confidence_score=0.5,
            timestamp=same_moment, triggered_recording=True,
        )
        for _ in range(5)
    ]
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)

    body = page(client, "/api/alerts", viewer_headers)

    assert [i["id"] for i in body["items"]] == sorted((r.id for r in rows), reverse=True)


def test_repeated_requests_for_the_same_page_are_identical(client, viewer_headers, db, camera):
    """A user clicking "next" then "previous" must see the same rows they
    saw before, not a reshuffle."""
    same_moment = datetime(2026, 3, 3, tzinfo=timezone.utc)
    db.add_all([
        Event(camera_id=camera.id, event_type="fall", confidence_score=0.5,
              timestamp=same_moment, triggered_recording=True)
        for _ in range(TOTAL_ROWS)
    ])
    db.commit()

    first = [i["id"] for i in page(client, "/api/alerts", viewer_headers, page=2, page_size=5)["items"]]
    second = [i["id"] for i in page(client, "/api/alerts", viewer_headers, page=2, page_size=5)["items"]]

    assert first == second


# --- ownership and IDOR ---------------------------------------------------

@pytest.fixture()
def other_users_rows(db, make_user_factory, tmp_path):
    """An upload owned by someone else, with a recording and an alert
    hanging off it - the shape the cross-user isolation tests use."""
    owner = make_user_factory("pagination-owner@example.com")
    source = tmp_path / "theirs.mp4"
    source.write_bytes(b"x" * 32)
    upload = VideoUpload(
        user_id=owner.id, original_filename="theirs.mp4", stored_path=str(source),
        status="completed", progress_percent=100, persons_detected=1, fall_events_count=1,
        file_size_bytes=32,
    )
    db.add(upload)
    db.commit()
    db.refresh(upload)

    clips, events = [], []
    for i in range(TOTAL_ROWS):
        path = tmp_path / f"theirs{i}.mp4"
        path.write_bytes(b"x" * 16)
        clip = Recording(
            camera_id=None, video_upload_id=upload.id, filename=path.name, file_path=str(path),
            duration_seconds=2.0, trigger_action="fall", file_size_bytes=16,
            event_timestamp=datetime.now(timezone.utc),
        )
        db.add(clip)
        db.flush()
        events.append(Event(
            camera_id=None, video_upload_id=upload.id, recording_id=clip.id,
            event_type="fall", confidence_score=0.8,
            timestamp=datetime.now(timezone.utc), triggered_recording=True,
        ))
        clips.append(clip)
    db.add_all(events)
    db.commit()
    return {"owner": owner, "upload": upload}


@pytest.mark.parametrize("path", ["/api/alerts", "/api/recordings", "/api/video-uploads"])
def test_a_page_never_contains_another_users_rows(client, viewer_headers, other_users_rows, path):
    body = page(client, path, viewer_headers)

    assert body["items"] == []


@pytest.mark.parametrize("path", ["/api/alerts", "/api/recordings", "/api/video-uploads"])
def test_the_total_is_scoped_too(client, viewer_headers, other_users_rows, path):
    """The count is the part it would be easy to get wrong - filtering the
    rows but counting the table. A `total` of 25 over an empty page tells
    the caller exactly how many rows they are not allowed to see."""
    body = page(client, path, viewer_headers)

    assert body["total"] == 0
    assert body["pages"] == 0


def test_paging_cannot_be_used_to_walk_into_another_users_rows(client, viewer_headers, other_users_rows):
    """The IDOR shape specific to pagination: if the scope were applied
    after the offset rather than in SQL, a high enough page would slide
    past the caller's own rows and into someone else's."""
    for number in (1, 2, 3, 50):
        body = page(client, "/api/alerts", viewer_headers, page=number, page_size=5)
        assert body["items"] == [], f"page {number} leaked rows"


def test_the_upload_filter_cannot_escape_the_scope(client, viewer_headers, other_users_rows):
    upload_id = other_users_rows["upload"].id

    body = page(client, "/api/alerts", viewer_headers, video_upload_id=upload_id, page_size=100)

    assert body["items"] == []
    assert body["total"] == 0


def test_the_owner_still_sees_their_own_rows_paged(client, auth_headers_for, other_users_rows):
    headers = auth_headers_for(other_users_rows["owner"])

    body = page(client, "/api/alerts", headers, page_size=10)

    assert body["total"] == TOTAL_ROWS
    assert len(body["items"]) == 10


# --- filters compose with paging ------------------------------------------

def test_a_filter_narrows_the_total_not_just_the_page(client, viewer_headers, db, camera):
    """Otherwise a UI computes the wrong page count for a filtered view and
    offers pages that are always empty."""
    db.add_all(
        [Event(camera_id=camera.id, event_type="fall", confidence_score=0.9,
               timestamp=datetime.now(timezone.utc), triggered_recording=True) for _ in range(7)]
        + [Event(camera_id=camera.id, event_type="crowd", confidence_score=0.9,
                 timestamp=datetime.now(timezone.utc), triggered_recording=True) for _ in range(13)]
    )
    db.commit()

    falls = page(client, "/api/alerts", viewer_headers, event_type="fall", page_size=5)

    assert falls["total"] == 7
    assert falls["pages"] == 2
    assert all(item["event_type"] == "fall" for item in falls["items"])


# --- admin users ----------------------------------------------------------

def test_admin_users_are_paged_oldest_first(client, admin_headers, make_user_factory):
    for i in range(5):
        make_user_factory(f"paged-user-{i}@example.com")

    body = page(client, "/api/admin/users", admin_headers, page_size=3)

    assert body["total"] == 6  # the five above plus the admin itself
    assert body["pages"] == 2
    created = [item["created_at"] for item in body["items"]]
    assert created == sorted(created)


def test_admin_user_paging_still_requires_admin(client, viewer_headers, operator_headers):
    for headers in (viewer_headers, operator_headers):
        assert client.get("/api/admin/users", params={"page": 1}, headers=headers).status_code == 403


# --- authentication is unchanged ------------------------------------------

@pytest.mark.parametrize("path", ["/api/alerts", "/api/recordings", "/api/video-uploads", "/api/admin/users"])
def test_paging_parameters_do_not_bypass_authentication(client, path):
    assert client.get(path, params={"page": 1, "page_size": 10}).status_code == 401


def test_recordings_can_be_filtered_by_trigger_action(client, viewer_headers, db, camera, tmp_path):
    """Added with paging, because client-side filtering became wrong the
    moment the client stopped receiving every row."""
    for i, action in enumerate(["fall"] * 4 + ["crowd"] * 6):
        path = tmp_path / f"filtered{i}.mp4"
        path.write_bytes(b"x" * 16)
        db.add(Recording(
            camera_id=camera.id, filename=path.name, file_path=str(path),
            duration_seconds=2.0, trigger_action=action, file_size_bytes=16,
            event_timestamp=datetime.now(timezone.utc),
        ))
    db.commit()

    falls = page(client, "/api/recordings", viewer_headers, trigger_action="fall", page_size=2)

    assert falls["total"] == 4
    assert falls["pages"] == 2
    assert all(item["trigger_action"] == "fall" for item in falls["items"])
