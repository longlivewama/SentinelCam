"""
Email delivery via stdlib smtplib (no extra mail dependency, per project
scope). This is a straight refactor of the previous
`app/services/email_service.py` module into the `Notifier` interface so it
can be composed with other channels (SMS/WhatsApp) by NotificationService.
"""
from __future__ import annotations

import logging
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.services.notifications.base import Notifier

logger = logging.getLogger(__name__)


class EmailNotifier(Notifier):
    name = "email"

    def send(
        self,
        subject: str,
        body: str,
        recipients: List[str],
        attachment_path: Optional[str] = None,
    ) -> bool:
        if not recipients:
            logger.info("EmailNotifier: no recipients given; skipping send for %r", subject)
            return False

        message = MIMEMultipart()
        message["Subject"] = subject
        message["From"] = settings.SMTP_FROM
        message["To"] = ", ".join(recipients)
        message.attach(MIMEText(body, "plain"))

        if attachment_path:
            path = Path(attachment_path)
            if path.exists():
                with open(path, "rb") as f:
                    image = MIMEImage(f.read())
                    image.add_header("Content-Disposition", "attachment", filename=path.name)
                    message.attach(image)

        try:
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
                if settings.SMTP_USE_TLS:
                    server.starttls()
                if settings.SMTP_USER:
                    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                server.sendmail(settings.SMTP_FROM, recipients, message.as_string())
            logger.info("Email sent: %r to %s", subject, recipients)
            return True
        except Exception:
            logger.exception("Failed to send email %r to %s", subject, recipients)
            return False
