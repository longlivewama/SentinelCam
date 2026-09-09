"""
Tests for the guard on scripts/seed_e2e_fixtures.py.

The seeder truncates every table before inserting its fixtures. It runs
inside the backend container, where DATABASE_URL addresses the compose
stack's own `sentinelcam` database - so the destructive run the e2e suite
wants and an accidental run over data someone still needs are the same
command against the same database. Only stated intent separates them,
which is why the wipe has to be asked for rather than inferred.

The name-based rule guarding tests/conftest.py deliberately does NOT apply
here: it would have to accept `sentinelcam`, and then it would accept the
accident too.
"""
import importlib.util
from pathlib import Path

import pytest

from app.config import settings

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "seed_e2e_fixtures.py"


def _load_seed_module():
    spec = importlib.util.spec_from_file_location("seed_e2e_fixtures", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def seed():
    return _load_seed_module()


def test_a_bare_invocation_is_refused(seed):
    """The accident: same command, no stated intent."""
    with pytest.raises(SystemExit) as excinfo:
        seed._assert_wipe_was_requested([])

    assert excinfo.value.code == 2, "must exit non-zero so a caller's execSync fails"


def test_the_confirm_flag_allows_the_wipe(seed):
    """The e2e suite's path - if this breaks, global-setup.js breaks."""
    seed._assert_wipe_was_requested([seed.CONFIRM_FLAG])


def test_the_flag_the_e2e_suite_passes_is_the_flag_this_script_expects(seed):
    """Pins the literal string on both sides of the contract; the caller
    lives in another language and cannot import this constant."""
    assert seed.CONFIRM_FLAG == "--wipe-database"

    global_setup = (Path(__file__).resolve().parents[2] / "e2e" / "global-setup.js").read_text()
    seeding_calls = [line for line in global_setup.splitlines() if "seed_e2e_fixtures.py" in line]

    assert seeding_calls, "expected global-setup.js to invoke the seeder"
    for call in seeding_calls:
        assert seed.CONFIRM_FLAG in call, f"seeding call would now be refused: {call.strip()}"


def test_unrelated_arguments_do_not_count_as_consent(seed):
    with pytest.raises(SystemExit):
        seed._assert_wipe_was_requested(["--wipe", "-y", "--force"])


def test_production_is_refused_even_with_the_flag(seed, monkeypatch):
    """A fixture seeder has no business in production, so consent does not
    apply there."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    with pytest.raises(SystemExit) as excinfo:
        seed._assert_wipe_was_requested([seed.CONFIRM_FLAG])

    assert excinfo.value.code == 2


def test_the_refusal_says_what_it_would_destroy(seed, capsys, db):
    """A bare "no" leaves the operator guessing whether they just avoided
    losing something. The inventory is the part that answers that."""
    from app.models.user import ROLE_ADMIN, User

    db.add(User(email="inventory@example.com", password_hash="x", full_name="Inv", role=ROLE_ADMIN, is_active=True))
    db.commit()

    with pytest.raises(SystemExit):
        seed._assert_wipe_was_requested([])
    message = capsys.readouterr().err

    assert "users" in message, "must inventory the tables that hold rows"
    assert seed.CONFIRM_FLAG in message, "must name the way forward"
    assert "DELETE" in message, "must say what it does"


def test_the_refusal_names_the_database_without_printing_a_dsn(seed, capsys):
    """It identifies the target by name and host only. Asserting "the
    password is absent" would be the obvious test and a false one - in the
    default local DSN the password is literally `sentinelcam`, a substring
    of the database name - so pin the real property instead: no connection
    string is ever printed, which is what keeps credentials out of logs."""
    with pytest.raises(SystemExit):
        seed._assert_wipe_was_requested([])
    message = capsys.readouterr().err

    assert str(seed.engine.url.database) in message, "must identify which database"
    assert "://" not in message, f"a DSN leaked into the message: {message}"
    assert "@" not in message, f"a credential-bearing netloc leaked into the message: {message}"


def test_row_counts_tolerate_a_database_whose_tables_do_not_exist_yet(seed, monkeypatch):
    """First run against a fresh database: `create_all` has not happened,
    and that is ordinary rather than an error."""
    monkeypatch.setattr(seed, "inspect", lambda _engine: type("I", (), {"get_table_names": lambda self: []})())

    assert seed._existing_row_counts() == {}
