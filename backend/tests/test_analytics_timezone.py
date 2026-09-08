"""
Regression tests for analytics day-bucketing across database session
timezones.

THE BUG
-------
`Event.timestamp` is `DateTime(timezone=True)` - `timestamptz` in
Postgres. psycopg2 returns a timestamptz as a timezone-aware datetime
rendered **in the database session's timezone**, not in UTC. The same
stored instant therefore arrives as different wall-clock values
depending only on how the server happens to be configured:

    2026-09-08 23:30:00+00 read under session TZ=UTC
        -> datetime(2026, 9, 8, 23, 30, tzinfo=utc)
    ...the very same row read under session TZ=Africa/Cairo
        -> datetime(2026, 9, 9,  2, 30, tzinfo=+03:00)

`get_summary` bucketed falls with `ts.strftime("%Y-%m-%d")` on that
value, while building the bucket *keys* it looks those up in from
`datetime.now(timezone.utc)`. Under a non-UTC session the two disagree,
and every event in the disputed window silently fell into a bucket that
the key list did not contain - so `falls.over_time` reported 0.

The window is exactly the offset: for Cairo (UTC+3) any event between
21:00 and 24:00 UTC is already "tomorrow" locally. That is why this only
ever showed up on a workstation whose Postgres inherited a non-UTC
system timezone; the `postgres:16` image used by CI and docker-compose
defaults to UTC, so CI could never catch it.

THE CONVENTION
--------------
UTC. It is what the application already uses everywhere else (27 call
sites of `datetime.now(timezone.utc)`, all `DateTime(timezone=True)`
columns), and no timezone configuration or policy exists anywhere in the
repository. The fix normalizes the value read back from the database to
UTC before taking its date, rather than trusting whatever timezone the
session happened to render it in.

WHY THE TESTS LOOK LIKE THIS
----------------------------
The session timezone must be controlled explicitly - reading the
developer's local timezone is what made the original bug invisible to
CI, and a test that depended on it would be exactly as unreliable. Each
test below pins the session timezone by overriding the `get_db`
dependency, so it asserts the same thing on every machine.
"""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from app.database import SessionLocal, get_db
from app.main import app
from app.models.camera import Camera
from app.models.event import Event

# UTC+3 year-round since 2014 (Egypt's DST reintroduction in 2023 runs
# late April - late October, so early September is +03 either way; these
# tests pin explicit instants rather than relying on that).
CAIRO = "Africa/Cairo"
UTC = "UTC"


@pytest.fixture()
def session_timezone(request):
    """Runs the API under a pinned database session timezone.

    Overrides `get_db` rather than touching the server or the database's
    own setting: the point is to prove the application produces the same
    answer whatever timezone a deployment's Postgres happens to use, and
    an override is scoped to the test instead of leaking into the rest of
    the session.
    """
    def _apply(tz_name: str):
        def _get_db_with_timezone():
            db = SessionLocal()
            try:
                db.execute(text(f"SET TIME ZONE '{tz_name}'"))
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = _get_db_with_timezone
        return tz_name

    yield _apply
    app.dependency_overrides.pop(get_db, None)


def _camera(db):
    camera = Camera(name="Cam A", url="0", camera_type="usb", status="active")
    db.add(camera)
    db.commit()
    db.refresh(camera)
    return camera


def _fall_at(db, camera, when: datetime):
    db.add(Event(
        camera_id=camera.id, event_type="fall", confidence_score=0.8,
        timestamp=when, triggered_recording=True,
    ))
    db.commit()


def _buckets(client, headers, days=14):
    resp = client.get(f"/api/analytics/summary?days={days}", headers=headers)
    assert resp.status_code == 200
    return {row["date"]: row["count"] for row in resp.json()["falls"]["over_time"]}


# --------------------------------------------------------------------------
# The exact failure: an event in the offset window, under a non-UTC session
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tz_name", [UTC, CAIRO])
def test_a_late_evening_utc_event_lands_on_its_utc_day_under_any_session_timezone(
    client, viewer_headers, db, session_timezone, tz_name,
):
    """The regression. 23:30 UTC is 02:30 the NEXT day in Cairo, so before
    the fix this event was bucketed under tomorrow's date while the bucket
    keys were built from UTC - and the count came back 0."""
    session_timezone(tz_name)
    camera = _camera(db)

    # Yesterday at 23:30 UTC: safely inside the 14-day window, and inside
    # the 21:00-24:00 UTC band where Cairo has already rolled over.
    instant = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
        hour=23, minute=30, second=0, microsecond=0,
    )
    _fall_at(db, camera, instant)

    buckets = _buckets(client, viewer_headers)
    utc_day = instant.strftime("%Y-%m-%d")

    assert buckets.get(utc_day) == 1, (
        f"under session TZ={tz_name}, a fall at {instant.isoformat()} was not counted "
        f"on its UTC day {utc_day}; buckets={ {k: v for k, v in buckets.items() if v} }"
    )
    assert sum(buckets.values()) == 1


def test_utc_and_cairo_sessions_produce_identical_buckets(
    client, viewer_headers, db, session_timezone,
):
    """The invariant stated directly: the same stored rows must aggregate
    to the same buckets regardless of the session timezone. This is the
    assertion that makes the result independent of deployment config."""
    camera = _camera(db)
    base = (datetime.now(timezone.utc) - timedelta(days=2)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    # Spread across the whole day, including both boundary bands.
    for hour in (0, 5, 12, 20, 21, 22, 23):
        _fall_at(db, camera, base.replace(hour=hour))

    session_timezone(UTC)
    utc_buckets = _buckets(client, viewer_headers)

    session_timezone(CAIRO)
    cairo_buckets = _buckets(client, viewer_headers)

    assert utc_buckets == cairo_buckets
    assert sum(utc_buckets.values()) == 7


# --------------------------------------------------------------------------
# Day boundaries, both directions
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tz_name", [UTC, CAIRO])
@pytest.mark.parametrize("hour", [21, 22, 23])
def test_every_hour_of_the_disputed_utc_band_is_counted(
    client, viewer_headers, db, session_timezone, tz_name, hour,
):
    """21:00-23:59 UTC is precisely the band Cairo (+03) has already
    carried into the next calendar day."""
    session_timezone(tz_name)
    camera = _camera(db)
    instant = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
        hour=hour, minute=30, second=0, microsecond=0,
    )
    _fall_at(db, camera, instant)

    buckets = _buckets(client, viewer_headers)
    assert buckets.get(instant.strftime("%Y-%m-%d")) == 1


@pytest.mark.parametrize("tz_name", [UTC, CAIRO])
def test_events_either_side_of_a_utc_midnight_fall_in_different_buckets(
    client, viewer_headers, db, session_timezone, tz_name,
):
    """Bucketing must still *separate* days - a fix that lumped
    everything together would satisfy the tests above but be just as
    wrong."""
    session_timezone(tz_name)
    camera = _camera(db)

    midnight = (datetime.now(timezone.utc) - timedelta(days=3)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    before = midnight - timedelta(minutes=30)   # previous UTC day, 23:30
    after = midnight + timedelta(minutes=30)    # this UTC day, 00:30
    _fall_at(db, camera, before)
    _fall_at(db, camera, after)

    buckets = _buckets(client, viewer_headers)
    assert buckets.get(before.strftime("%Y-%m-%d")) == 1
    assert buckets.get(after.strftime("%Y-%m-%d")) == 1
    assert before.strftime("%Y-%m-%d") != after.strftime("%Y-%m-%d")


@pytest.mark.parametrize("tz_name", [UTC, CAIRO])
def test_an_event_at_cairo_midnight_is_bucketed_by_its_utc_day(
    client, viewer_headers, db, session_timezone, tz_name,
):
    """The mirror case: 21:00 UTC is exactly 00:00 in Cairo. Under the
    documented UTC convention it belongs to the UTC day, not the Cairo
    one - pinned explicitly so a future change of convention has to be
    deliberate rather than accidental."""
    session_timezone(tz_name)
    camera = _camera(db)

    instant = (datetime.now(timezone.utc) - timedelta(days=2)).replace(
        hour=21, minute=0, second=0, microsecond=0,
    )
    _fall_at(db, camera, instant)

    buckets = _buckets(client, viewer_headers)
    assert buckets.get(instant.strftime("%Y-%m-%d")) == 1
    # NOT the following (Cairo) day.
    next_day = (instant + timedelta(days=1)).strftime("%Y-%m-%d")
    assert buckets.get(next_day, 0) == 0


# --------------------------------------------------------------------------
# Aggregation shape: multiple events, empty days, window structure
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tz_name", [UTC, CAIRO])
def test_several_events_on_one_day_accumulate_into_a_single_bucket(
    client, viewer_headers, db, session_timezone, tz_name,
):
    session_timezone(tz_name)
    camera = _camera(db)
    day = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0,
    )
    for hour in (1, 9, 22, 23):
        _fall_at(db, camera, day.replace(hour=hour))

    buckets = _buckets(client, viewer_headers)
    assert buckets.get(day.strftime("%Y-%m-%d")) == 4
    assert sum(buckets.values()) == 4


@pytest.mark.parametrize("tz_name", [UTC, CAIRO])
def test_days_with_no_events_are_present_and_zero(
    client, viewer_headers, db, session_timezone, tz_name,
):
    """The series must stay dense - the chart plots every bucket, so a
    missing day would shift the axis rather than show a gap."""
    session_timezone(tz_name)
    _camera(db)

    resp = client.get("/api/analytics/summary?days=7", headers=viewer_headers)
    rows = resp.json()["falls"]["over_time"]

    assert len(rows) == 7
    assert all(row["count"] == 0 for row in rows)
    dates = [row["date"] for row in rows]
    assert dates == sorted(dates), "buckets must run oldest-first"
    assert dates[-1] == datetime.now(timezone.utc).strftime("%Y-%m-%d"), (
        "the final bucket must be today in UTC"
    )


@pytest.mark.parametrize("tz_name", [UTC, CAIRO])
def test_consecutive_days_each_get_their_own_bucket(
    client, viewer_headers, db, session_timezone, tz_name,
):
    session_timezone(tz_name)
    camera = _camera(db)

    expected = {}
    for days_ago, count in ((1, 2), (2, 1), (4, 3)):
        day = (datetime.now(timezone.utc) - timedelta(days=days_ago)).replace(
            hour=22, minute=0, second=0, microsecond=0,  # inside the disputed band
        )
        for _ in range(count):
            _fall_at(db, camera, day)
        expected[day.strftime("%Y-%m-%d")] = count

    buckets = _buckets(client, viewer_headers)
    for date, count in expected.items():
        assert buckets.get(date) == count, f"{date} under TZ={tz_name}"
    assert sum(buckets.values()) == 6


# --------------------------------------------------------------------------
# The rest of the payload must not shift with the session timezone either
# --------------------------------------------------------------------------

def test_the_window_filter_and_totals_are_timezone_independent(
    client, viewer_headers, db, session_timezone,
):
    """`since` is an absolute instant compared in SQL, so it was never
    part of the bug - but it shares the endpoint, and a fix that
    accidentally shifted the window would show up here."""
    camera = _camera(db)
    inside = datetime.now(timezone.utc) - timedelta(days=3)
    outside = datetime.now(timezone.utc) - timedelta(days=40)
    _fall_at(db, camera, inside)
    _fall_at(db, camera, outside)

    session_timezone(UTC)
    utc_body = client.get("/api/analytics/summary?days=14", headers=viewer_headers).json()
    session_timezone(CAIRO)
    cairo_body = client.get("/api/analytics/summary?days=14", headers=viewer_headers).json()

    # Both events are stored, but only the recent one is in the window.
    assert utc_body["alerts"]["total"] == 2
    assert utc_body["alerts"]["by_type_recent"]["fall"] == 1
    assert sum(d["count"] for d in utc_body["falls"]["over_time"]) == 1
    # ...and the session timezone changes none of it.
    assert utc_body["alerts"] == cairo_body["alerts"]
    assert utc_body["falls"] == cairo_body["falls"]
    assert utc_body["confidence"] == cairo_body["confidence"]


def test_the_response_contract_is_unchanged(client, viewer_headers, db, session_timezone):
    """The frontend reads `date` as an opaque label (`d.date.slice(5)`)
    and sums `count` - this pins both, so the fix stays invisible to it."""
    session_timezone(CAIRO)
    camera = _camera(db)
    _fall_at(db, camera, datetime.now(timezone.utc) - timedelta(days=1, hours=1))

    body = client.get("/api/analytics/summary?days=14", headers=viewer_headers).json()
    rows = body["falls"]["over_time"]

    assert isinstance(rows, list) and len(rows) == 14
    assert set(rows[0]) == {"date", "count"}
    assert all(isinstance(r["date"], str) and len(r["date"]) == 10 for r in rows)
    assert all(isinstance(r["count"], int) for r in rows)
    # by_camera keeps its shape too.
    assert set(body["falls"]["by_camera"][0]) == {"camera_id", "camera_name", "count"}
