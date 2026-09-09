"""
Tests for the guard that keeps this suite off non-disposable databases.

THE INCIDENT
------------
`conftest` resolves DATABASE_URL with `setdefault`, so an already-exported
value wins. That is deliberate - it is how CI points the suite at its own
Postgres service - but it means the suite follows DATABASE_URL wherever it
happens to lead. Run `pytest` inside the backend container, where
docker-compose exports DATABASE_URL for the application itself, and the
session fixture's `drop_all` lands on live data. That is not hypothetical:
it is how the development database got wiped.

The guard makes the destructive fixtures refuse anything that isn't named
like a scratch database. These tests pin that behaviour, including the
exact DSNs the real setups use - if someone renames the CI database, the
suite should fail here with a clear reason rather than by being blocked at
collection time in CI.
"""
import pytest

from conftest import ALLOW_NON_TEST_DATABASE_ENV, _assert_disposable_database, _redacted

LOCAL_DEFAULT = "postgresql://sentinelcam:sentinelcam@localhost:5433/sentinelcam_test"
CI_DSN = "postgresql://sentinelcam:sentinelcam@localhost:5432/sentinelcam_test"
# What docker-compose exports inside the backend container - the DSN that
# caused the incident.
LIVE_CONTAINER_DSN = "postgresql://sentinelcam:sentinelcam@postgres:5432/sentinelcam"


@pytest.fixture(autouse=True)
def _no_inherited_override(monkeypatch):
    """The override must not leak in from the ambient environment, or the
    rejection tests below would silently stop testing anything."""
    monkeypatch.delenv(ALLOW_NON_TEST_DATABASE_ENV, raising=False)


# --- the databases the project actually uses must keep working -------------

@pytest.mark.parametrize("url", [LOCAL_DEFAULT, CI_DSN])
def test_the_real_test_databases_are_accepted(url):
    """Both supported setups must pass, or this guard breaks the suite it
    is meant to protect."""
    _assert_disposable_database(url)


@pytest.mark.parametrize(
    "database",
    ["sentinelcam_test", "test_sentinelcam", "anything_test", "test_"],
)
def test_conventionally_named_scratch_databases_are_accepted(database):
    _assert_disposable_database(f"postgresql://u:p@host:5432/{database}")


def test_query_parameters_do_not_hide_the_database_name():
    """A DSN carrying options must still be parsed down to the name, not
    checked as a whole string."""
    _assert_disposable_database(f"{LOCAL_DEFAULT}?sslmode=require")


# --- and everything else must be refused ----------------------------------

def test_the_live_container_database_is_refused():
    """The exact regression: pytest run inside the backend container."""
    with pytest.raises(pytest.UsageError) as excinfo:
        _assert_disposable_database(LIVE_CONTAINER_DSN)

    assert "sentinelcam" in str(excinfo.value)


@pytest.mark.parametrize(
    "database",
    ["sentinelcam", "postgres", "sentinelcam_prod", "testing", "sentinelcam_test_backup"],
)
def test_databases_that_are_not_obviously_disposable_are_refused(database):
    """`testing` and `..._test_backup` are the near-misses worth pinning:
    neither ends in `_test`, and a substring check would have let both
    through."""
    with pytest.raises(pytest.UsageError):
        _assert_disposable_database(f"postgresql://u:p@host:5432/{database}")


def test_a_dsn_with_no_database_at_all_is_refused():
    with pytest.raises(pytest.UsageError):
        _assert_disposable_database("postgresql://u:p@host:5432/")


def test_the_refusal_explains_how_to_proceed():
    """A guard that only says "no" gets deleted by the next person who
    hits it, so the message has to carry the way out."""
    with pytest.raises(pytest.UsageError) as excinfo:
        _assert_disposable_database(LIVE_CONTAINER_DSN)
    message = str(excinfo.value)

    assert "DROP" in message and "DELETE" in message, "must say why it is refusing"
    assert "_test" in message, "must name the convention it wants"
    assert ALLOW_NON_TEST_DATABASE_ENV in message, "must name the deliberate opt-out"


# --- the opt-out ----------------------------------------------------------

def test_the_opt_out_allows_a_non_test_database(monkeypatch):
    monkeypatch.setenv(ALLOW_NON_TEST_DATABASE_ENV, "1")
    _assert_disposable_database(LIVE_CONTAINER_DSN)


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_the_opt_out_takes_only_an_exact_1(monkeypatch, value):
    """Anything vaguer would make a stray/truthy-looking value silently
    re-arm the original footgun."""
    monkeypatch.setenv(ALLOW_NON_TEST_DATABASE_ENV, value)
    with pytest.raises(pytest.UsageError):
        _assert_disposable_database(LIVE_CONTAINER_DSN)


# --- credentials must not leak into the failure output --------------------

def test_the_password_is_redacted_from_the_quoted_dsn():
    """The message quotes DATABASE_URL, and that output lands in CI logs."""
    with pytest.raises(pytest.UsageError) as excinfo:
        _assert_disposable_database("postgresql://admin:hunter2@db.internal:5432/sentinelcam")
    message = str(excinfo.value)

    assert "hunter2" not in message
    assert "admin" in message and "db.internal" in message, "the rest must stay diagnosable"


def test_redaction_leaves_a_passwordless_dsn_untouched():
    url = "postgresql://sentinelcam@localhost:5432/sentinelcam_test"
    assert _redacted(url) == url
