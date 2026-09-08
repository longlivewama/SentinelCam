"""
Regression test for a real bug found via Docker Compose e2e testing: the
rest of this suite creates its schema with Base.metadata.create_all()
(see conftest.py), which never actually exercises the Alembic migrations
themselves - so a migration that only works against an
already-existing (pre-Alembic) database, and fails on a genuinely fresh
one, was never caught here. This test runs `alembic upgrade head` for
real against a throwaway, completely empty database.
"""
import os
import subprocess
import sys
import uuid
from pathlib import Path

import psycopg2
import pytest

from app.config import settings

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _admin_dsn(dbname: str = "postgres") -> str:
    base = settings.DATABASE_URL.rsplit("/", 1)[0]
    return f"{base}/{dbname}"


@pytest.fixture()
def fresh_empty_database():
    db_name = f"sentinelcam_migration_check_{uuid.uuid4().hex[:8]}"
    admin_conn = psycopg2.connect(_admin_dsn())
    admin_conn.autocommit = True
    with admin_conn.cursor() as cur:
        cur.execute(f'CREATE DATABASE "{db_name}"')
    admin_conn.close()

    yield f"{settings.DATABASE_URL.rsplit('/', 1)[0]}/{db_name}"

    admin_conn = psycopg2.connect(_admin_dsn())
    admin_conn.autocommit = True
    with admin_conn.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
    admin_conn.close()


def test_alembic_upgrade_head_succeeds_on_a_fresh_database(fresh_empty_database):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        env={**os.environ, "DATABASE_URL": fresh_empty_database},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"alembic upgrade head failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    conn = psycopg2.connect(fresh_empty_database)
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = {row[0] for row in cur.fetchall()}
    conn.close()

    for expected in ("users", "cameras", "events", "recordings", "password_reset_tokens", "video_uploads", "alembic_version"):
        assert expected in tables, f"missing table {expected!r} after alembic upgrade head: {tables}"


def test_migrated_schema_matches_the_models(fresh_empty_database):
    """`alembic upgrade head` must produce the schema app/models/*.py
    expects. The rest of the suite builds its schema with
    `create_all`, so a model column added without a matching migration
    passes every other test and then fails in production with
    UndefinedColumn on the first write - which is exactly what happened
    when events.video_timestamp_seconds was added."""
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=BACKEND_DIR,
        env={**os.environ, "DATABASE_URL": fresh_empty_database},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"alembic upgrade head failed:\nSTDERR: {result.stderr}"

    from app.database import Base
    import app.models  # noqa: F401 - registers every model on Base.metadata

    conn = psycopg2.connect(fresh_empty_database)
    migrated = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'public'"
        )
        for table_name, column_name in cur.fetchall():
            migrated.setdefault(table_name, set()).add(column_name)
    conn.close()

    missing = []
    for table in Base.metadata.sorted_tables:
        model_columns = {c.name for c in table.columns}
        migrated_columns = migrated.get(table.name, set())
        for column in sorted(model_columns - migrated_columns):
            missing.append(f"{table.name}.{column}")

    assert not missing, (
        "these columns exist on the SQLAlchemy models but not after "
        f"`alembic upgrade head` - a migration is missing: {missing}"
    )
