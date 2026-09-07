#!/usr/bin/env python
"""
Creates the first admin user for SentinelCam.

Usage:
    python seed_admin.py

Reads ADMIN_EMAIL / ADMIN_PASSWORD / ADMIN_FULL_NAME from the environment
(or a .env file, via app.config) if present; otherwise prompts
interactively. Safe to re-run - if a user with that email already exists
it is left untouched (and promoted to admin/active if it wasn't already).
"""
import getpass
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # so ADMIN_EMAIL / ADMIN_PASSWORD in .env are picked up, same as app.config does for its own settings

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.models.user import ROLE_ADMIN, User  # noqa: E402


def _prompt_email() -> str:
    email = os.environ.get("ADMIN_EMAIL")
    if email:
        return email.strip()
    return input("Admin email: ").strip()


def _prompt_password() -> str:
    password = os.environ.get("ADMIN_PASSWORD")
    if password:
        return password
    while True:
        pw1 = getpass.getpass("Admin password: ")
        pw2 = getpass.getpass("Confirm password: ")
        if pw1 != pw2:
            print("Passwords do not match, try again.")
            continue
        if len(pw1) < 8:
            print("Password must be at least 8 characters.")
            continue
        return pw1


def main():
    Base.metadata.create_all(bind=engine)

    email = _prompt_email()
    if not email:
        print("An email address is required.", file=sys.stderr)
        sys.exit(1)

    full_name = os.environ.get("ADMIN_FULL_NAME", "Administrator")
    password = _prompt_password()

    with SessionLocal() as db:
        existing = db.query(User).filter(User.email == email).first()
        if existing is not None:
            changed = False
            if existing.role != ROLE_ADMIN:
                existing.role = ROLE_ADMIN
                changed = True
            if not existing.is_active:
                existing.is_active = True
                changed = True
            if changed:
                db.commit()
                print(f"User {email} already existed; promoted to active admin.")
            else:
                print(f"User {email} already exists and is already an active admin. No changes made.")
            return

        user = User(
            email=email,
            password_hash=hash_password(password),
            full_name=full_name,
            role=ROLE_ADMIN,
            is_active=True,
        )
        db.add(user)
        db.commit()
        print(f"Admin user created: {email}")


if __name__ == "__main__":
    main()
