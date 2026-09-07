from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.deps import get_current_user, require_operator
from app.database import get_db
from app.models.event import Event
from app.models.user import User
from app.schemas.event import EventOut

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=List[EventOut])
def list_alerts(
    camera_id: Optional[int] = None,
    video_upload_id: Optional[int] = None,
    event_type: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    limit: int = 200,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Event)
    if camera_id is not None:
        query = query.filter(Event.camera_id == camera_id)
    if video_upload_id is not None:
        query = query.filter(Event.video_upload_id == video_upload_id)
    if event_type is not None:
        query = query.filter(Event.event_type == event_type)
    if acknowledged is not None:
        query = query.filter(Event.acknowledged == acknowledged)
    return query.order_by(Event.timestamp.desc()).limit(min(limit, 1000)).all()


@router.get("/{alert_id}", response_model=EventOut)
def get_alert(alert_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    alert = db.query(Event).filter(Event.id == alert_id).first()
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    return alert


@router.put("/{alert_id}/acknowledge", response_model=EventOut)
def acknowledge_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    operator: User = Depends(require_operator),
):
    alert = db.query(Event).filter(Event.id == alert_id).first()
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")

    alert.acknowledged = True
    alert.acknowledged_by = operator.id
    alert.acknowledged_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(alert)
    return alert
