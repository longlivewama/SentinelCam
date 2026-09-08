import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import require_admin
from app.core.security import hash_password
from app.database import get_db
from app.models.password_reset_token import PasswordResetToken
from app.models.user import User
from app.models.video_upload import STATUS_PENDING, STATUS_PROCESSING, VideoUpload
from app.schemas.user import UserCreate, UserOut, UserUpdateRole
from app.services.cascade_delete import stage_delete_video_upload, unlink_user_from_acknowledged_alerts

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=List[UserOut])
def list_users(db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    return db.query(User).order_by(User.created_at.asc()).all()


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already in use")

    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        role=payload.role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    if user_id == admin.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete your own account")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # An upload that is still pending/processing has an analysis worker
    # thread holding its stored_path and writing to its row - deleting the
    # user (and cascading into that upload) here would race it exactly
    # like deleting the upload directly does (see video_uploads.py's
    # DELETE endpoint). That endpoint defers to the worker by flipping
    # deletion_requested, but video_uploads.user_id is NOT NULL, so a
    # deferred-but-not-yet-deleted upload would still block deleting this
    # user row on the same FK a moment later. Refuse instead: the admin
    # can retry once analysis finishes, or delete those uploads
    # individually first (which defers correctly and returns immediately).
    active_uploads = (
        db.query(VideoUpload)
        .filter(VideoUpload.user_id == user_id, VideoUpload.status.in_((STATUS_PENDING, STATUS_PROCESSING)))
        .count()
    )
    if active_uploads:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot delete this user: {active_uploads} of their video upload(s) are still "
                "being analyzed. Wait for analysis to finish (or fail), or delete those uploads "
                "individually, then retry."
            ),
        )

    # Clear/cascade everything else that holds a foreign key to this user
    # before deleting it, so this doesn't hit any of those FK constraints:
    # their uploaded videos have no independent meaning without them (so
    # cascade-delete, like a user deleting their own upload would), but an
    # alert they acknowledged is still meaningful history (so just clear
    # the attribution), and any password reset tokens are meaningless
    # without the account.
    file_paths = []
    for upload in db.query(VideoUpload).filter(VideoUpload.user_id == user_id).all():
        file_paths.extend(stage_delete_video_upload(db, upload))
    unlink_user_from_acknowledged_alerts(db, user_id)
    db.query(PasswordResetToken).filter(PasswordResetToken.user_id == user_id).delete()

    db.delete(user)
    db.commit()

    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    return None


@router.put("/users/{user_id}/toggle", response_model=UserOut)
def toggle_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    if user_id == admin.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot disable your own account")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user.is_active = not user.is_active
    db.commit()
    db.refresh(user)
    return user


@router.put("/users/{user_id}/role", response_model=UserOut)
def update_user_role(
    user_id: int,
    payload: UserUpdateRole,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    if user_id == admin.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot change your own role")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user.role = payload.role
    db.commit()
    db.refresh(user)
    return user
