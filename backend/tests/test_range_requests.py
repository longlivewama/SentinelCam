"""
Regression tests for HTTP Range handling on the two byte-serving
endpoints. Both previously parsed the header inline with
`int(value.partition("-")[0])`, which:

  * raised ValueError -> HTTP 500 on a non-numeric header,
  * read `bytes=-500` ("the last 500 bytes") as bytes 0-500, and
  * answered a start past EOF with 206 and a nonsensical Content-Range.
"""
from datetime import datetime, timezone

import pytest

from app.core.ranges import RangeNotSatisfiable, parse_range_header
from app.models.recording import Recording

CONTENT = bytes(range(256)) * 4  # 1024 deterministic bytes


@pytest.fixture()
def recording(db, tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(CONTENT)
    row = Recording(
        camera_id=None,
        video_upload_id=None,
        filename="clip.mp4",
        file_path=str(clip),
        duration_seconds=4.0,
        trigger_action="fall",
        file_size_bytes=len(CONTENT),
        event_timestamp=datetime.now(timezone.utc),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# --- parser unit tests ----------------------------------------------------

def test_absent_or_malformed_range_serves_the_whole_file():
    for header in (None, "", "bytes=abc-def", "bytes=", "items=0-10", "bytes=0-99,200-299", "garbage"):
        assert parse_range_header(header, 1000) is None, header


def test_open_ended_range():
    assert parse_range_header("bytes=100-", 1000) == (100, 999)


def test_closed_range_is_clamped_to_the_file():
    assert parse_range_header("bytes=100-5000", 1000) == (100, 999)


def test_suffix_range_means_the_last_n_bytes():
    # The old inline parser read this as bytes 0-500.
    assert parse_range_header("bytes=-500", 1000) == (500, 999)


def test_suffix_longer_than_the_file_starts_at_zero():
    assert parse_range_header("bytes=-5000", 1000) == (0, 999)


def test_start_past_end_of_file_is_unsatisfiable():
    with pytest.raises(RangeNotSatisfiable):
        parse_range_header("bytes=2000-3000", 1000)


def test_inverted_range_is_unsatisfiable():
    with pytest.raises(RangeNotSatisfiable):
        parse_range_header("bytes=900-100", 1000)


# --- endpoint behaviour ---------------------------------------------------

def test_full_response_without_a_range_header(client, viewer_headers, recording):
    response = client.get(f"/api/recordings/{recording.id}/video", headers=viewer_headers)
    assert response.status_code == 200
    assert response.content == CONTENT
    assert response.headers["accept-ranges"] == "bytes"


def test_partial_response_returns_exactly_the_requested_bytes(client, viewer_headers, recording):
    response = client.get(
        f"/api/recordings/{recording.id}/video",
        headers={**viewer_headers, "Range": "bytes=10-19"},
    )
    assert response.status_code == 206
    assert response.content == CONTENT[10:20]
    assert response.headers["content-range"] == f"bytes 10-19/{len(CONTENT)}"


def test_suffix_range_returns_the_tail_of_the_file(client, viewer_headers, recording):
    response = client.get(
        f"/api/recordings/{recording.id}/video",
        headers={**viewer_headers, "Range": "bytes=-16"},
    )
    assert response.status_code == 206
    assert response.content == CONTENT[-16:]


def test_garbage_range_header_does_not_500(client, viewer_headers, recording):
    """The whole point of the fix: a scanner sending nonsense must not
    produce a server error."""
    response = client.get(
        f"/api/recordings/{recording.id}/video",
        headers={**viewer_headers, "Range": "bytes=not-a-number"},
    )
    assert response.status_code == 200
    assert response.content == CONTENT


def test_unsatisfiable_range_returns_416(client, viewer_headers, recording):
    response = client.get(
        f"/api/recordings/{recording.id}/video",
        headers={**viewer_headers, "Range": "bytes=99999-"},
    )
    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{len(CONTENT)}"


# --- list endpoints are bounded -------------------------------------------

def test_recordings_listing_is_bounded(client, viewer_headers, recording):
    """An unbounded list grows with every clip a running camera produces,
    until the response is megabytes and the browser renders thousands of
    rows. The cap is generous; what matters is that one exists."""
    from app.api.routes.recordings import DEFAULT_LIMIT, MAX_LIMIT

    assert DEFAULT_LIMIT <= MAX_LIMIT

    ok = client.get("/api/recordings", params={"limit": 1}, headers=viewer_headers)
    assert ok.status_code == 200
    assert len(ok.json()) <= 1

    over = client.get("/api/recordings", params={"limit": MAX_LIMIT + 1}, headers=viewer_headers)
    assert over.status_code == 422

    under = client.get("/api/recordings", params={"limit": 0}, headers=viewer_headers)
    assert under.status_code == 422


def test_alerts_listing_rejects_an_out_of_range_limit(client, viewer_headers):
    from app.api.routes.alerts import MAX_LIMIT

    assert client.get("/api/alerts", params={"limit": MAX_LIMIT + 1}, headers=viewer_headers).status_code == 422
    assert client.get("/api/alerts", params={"limit": 0}, headers=viewer_headers).status_code == 422
    assert client.get("/api/alerts", params={"limit": 5}, headers=viewer_headers).status_code == 200
