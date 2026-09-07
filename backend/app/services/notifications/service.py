"""
Fans a notification out to every configured channel (settings.
NOTIFICATION_CHANNELS, default "email"). Callers use the semantic methods
below (notify_alert, notify_password_reset, ...) rather than talking to a
specific channel directly, so adding SMS/WhatsApp later is a config change,
not a call-site change.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from app.config import settings
from app.services.notifications.base import Notifier
from app.services.notifications.email_notifier import EmailNotifier
from app.services.notifications.sms_notifier import SmsNotifier, WhatsAppNotifier

logger = logging.getLogger(__name__)

_CHANNEL_REGISTRY = {
    "email": EmailNotifier,
    "sms": SmsNotifier,
    "whatsapp": WhatsAppNotifier,
}


class NotificationService:
    def __init__(self):
        self._channels: List[Notifier] = self._build_channels()

    def _build_channels(self) -> List[Notifier]:
        channels = []
        for name in settings.notification_channels:
            cls = _CHANNEL_REGISTRY.get(name)
            if cls is None:
                logger.warning("Unknown notification channel %r in NOTIFICATION_CHANNELS; skipping", name)
                continue
            channels.append(cls())
        return channels

    def _dispatch(self, subject: str, body: str, recipients: List[str], attachment_path: Optional[str] = None) -> bool:
        if not recipients:
            logger.info("No recipients configured; skipping notification %r", subject)
            return False
        sent_any = False
        for channel in self._channels:
            try:
                if channel.send(subject, body, recipients, attachment_path):
                    sent_any = True
            except Exception:
                logger.exception("Notification channel %s raised unexpectedly", channel.name)
        return sent_any

    # -- semantic notification methods -----------------------------------

    def notify_alert(
        self,
        event_type: str,
        source_name: str,
        timestamp: datetime,
        snapshot_path: Optional[str] = None,
    ) -> bool:
        recipients = settings.alert_recipients
        subject = f"[SentinelCam] {event_type.replace('_', ' ').title()} detected on {source_name}"
        body = (
            f"SentinelCam detected a '{event_type}' event.\n\n"
            f"Source: {source_name}\n"
            f"Time: {timestamp.isoformat()}\n"
        )
        return self._dispatch(subject, body, recipients, snapshot_path)

    def notify_password_reset(self, email: str, reset_url: str) -> bool:
        subject = "[SentinelCam] Password reset request"
        body = (
            "We received a request to reset your SentinelCam password.\n\n"
            f"Reset your password using this link (valid for a limited time):\n{reset_url}\n\n"
            "If you did not request this, you can safely ignore this email."
        )
        return self._dispatch(subject, body, [email])

    def notify_video_analysis_complete(self, email: str, filename: str, fall_events_count: int) -> bool:
        subject = f"[SentinelCam] Video analysis complete: {filename}"
        summary = (
            f"{fall_events_count} fall event(s) detected."
            if fall_events_count
            else "No fall events detected."
        )
        body = f"Your uploaded video '{filename}' has finished processing.\n\n{summary}\n"
        return self._dispatch(subject, body, [email])


notification_service = NotificationService()
