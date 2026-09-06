"""
SQLAlchemy engine/session setup. All models import `Base` from here, and
FastAPI route dependencies use `get_db()` to obtain a request-scoped session.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

# `connect_args` below only applies to sqlite (used solely for local
# throwaway verification, never in the shipped/default configuration which
# targets PostgreSQL via DATABASE_URL).
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
