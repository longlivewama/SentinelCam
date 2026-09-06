"""
Alert email delivery via stdlib smtplib (no extra mail dependency, per
project scope). Sends a plain-text + optional JPEG snapshot attachment to
every address in settings.ALERT_RECIPIENTS whenever a detector fires.
"""
from __future__ import annotations

import logging
import smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


def send_alert_email(
    event_type: str,
    camera_name: str,
    timestamp: datetime,
    snapshot_path: Optional[str] = None,
) -> bool:
    """Send an alert email for a detected event. Returns True on success,
    False on failure (never raises - a broken mail server should not crash
    the detection pipeline)."""
    recipients = settings.alert_recipients
    if not recipients:
        logger.info("No ALERT_RECIPIENTS configured; skipping alert email for %s", event_type)
        return False

    subject = f"[SentinelCam] {event_type.replace('_', ' ').title()} detected on {camera_name}"
    body = (
        f"SentinelCam detected a '{event_type}' event.\n\n"
        f"Camera: {camera_name}\n"
        f"Time: {timestamp.isoformat()}\n"
    )

    message = MIMEMultipart()
    message["Subject"] = subject
    message["From"] = settings.SMTP_FROM
    message["To"] = ", ".join(recipients)
    message.attach(MIMEText(body, "plain"))

    if snapshot_path:
        path = Path(snapshot_path)
        if path.exists():
            with open(path, "rb") as f:
                image = MIMEImage(f.read())
                image.add_header("Content-Disposition", "attachment", filename=path.name)
                message.attach(image)

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
            server.starttls()
            if settings.SMTP_USER:
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM, recipients, message.as_string())
        logger.info("Alert email sent for %s event on %s", event_type, camera_name)
        return True
    except Exception:
        logger.exception("Failed to send alert email for %s event on %s", event_type, camera_name)
        return False
