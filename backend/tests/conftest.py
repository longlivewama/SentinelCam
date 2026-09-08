"""
Shared pytest fixtures.

Tests run against a REAL PostgreSQL database (sentinelcam_test) rather than
sqlite - the app is written and tuned against Postgres-specific behavior in
a few places (e.g. boolean filters, timezone-aware timestamps), so a real
Postgres instance gives much higher confidence than an in-memory sqlite
substitute. The env vars below are set before any `app.*` module is
imported, so `app.config.settings` (a module-level singleton) picks up the
test database and disabled notification channels from the very first
import anywhere in the test session.

Requires a local Postgres reachable at the DSN below with a
`sentinelcam_test` database already created (see backend/README.md's
"Running tests" section for the one-time setup command).
"""
import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql://sentinelcam:sentinelcam@localhost:5433/sentinelcam_test"
)
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("ENVIRONMENT", "test")
# No real network calls to an SMTP server during tests.
os.environ.setdefault("NOTIFICATION_CHANNELS", "")
os.environ.setdefault("ALERT_RECIPIENTS", "")

import pytest
from fastapi.testclient import TestClient

from app.core.security import create_access_token, hash_password
from app.database import Base, SessionLocal, engine
from app.main import app
from app.models.camera import Camera
from app.models.event import Event
from app.models.password_reset_token import PasswordResetToken
from app.models.recording import Recording
from app.models.user import ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER, User
from app.models.video_upload import VideoUpload

# Tables are dropped/deleted in FK-dependency order.
_TABLES_IN_DELETE_ORDER = [
    PasswordResetToken.__table__,
    Event.__table__,
    Recording.__table__,
    VideoUpload.__table__,
    Camera.__table__,
    User.__table__,
]


@pytest.fixture(scope="session", autouse=True)
def _schema():
    """Rebuilds the test schema from the models at the start of every
    session. `create_all` alone only ever ADDS missing tables - it never
    alters an existing one - so a test database left over from an older
    revision of the models would silently keep the old columns and fail
    every test that touches a newly added one. Dropping first makes the
    schema unconditionally match app/models/*.py."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture(autouse=True)
def _clean_tables():
    """Runs before every test: guarantees a blank slate regardless of what
    a previous test (or a crashed previous run) left behind."""
    with engine.begin() as conn:
        for table in _TABLES_IN_DELETE_ORDER:
            conn.execute(table.delete())
    yield


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """The auth rate limiter (app.core.rate_limit) tracks hits in a
    process-wide dict keyed by (path, client IP) - without resetting it,
    tests that repeatedly hit /api/auth/login etc. across the whole test
    session would eventually trip each other's limits, since TestClient
    requests all share the same fake client IP."""
    from app.core.rate_limit import _hits

    _hits.clear()
    yield
    _hits.clear()


@pytest.fixture()
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def _make_user(db, email: str, role: str, password: str = "password123") -> User:
    user = User(email=email, password_hash=hash_password(password), full_name=email.split("@")[0], role=role, is_active=True)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _auth_headers(user: User) -> dict:
    token = create_access_token(user.id, user.email)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def make_user_factory(db):
    """Creates additional users beyond the three role fixtures below -
    e.g. a second viewer, to test that one account cannot reach another's
    data. Exposed as a fixture rather than importing _make_user directly,
    because `tests` is not an importable package here."""
    def _factory(email: str, role: str = ROLE_VIEWER, password: str = "password123") -> User:
        return _make_user(db, email, role, password)

    return _factory


@pytest.fixture()
def auth_headers_for():
    """Bearer headers for an arbitrary User object."""
    return _auth_headers


@pytest.fixture()
def admin_user(db):
    return _make_user(db, "admin@example.com", ROLE_ADMIN)


@pytest.fixture()
def operator_user(db):
    return _make_user(db, "operator@example.com", ROLE_OPERATOR)


@pytest.fixture()
def viewer_user(db):
    return _make_user(db, "viewer@example.com", ROLE_VIEWER)


@pytest.fixture()
def admin_headers(admin_user):
    return _auth_headers(admin_user)


@pytest.fixture()
def operator_headers(operator_user):
    return _auth_headers(operator_user)


@pytest.fixture()
def viewer_headers(viewer_user):
    return _auth_headers(viewer_user)
