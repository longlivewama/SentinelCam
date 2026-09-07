"""
Placeholder SMS channel. Not wired to a real provider (Twilio, SNS, etc.)
in this codebase - this stub exists purely to demonstrate the extension
point described in NotificationService. To make it real: add the
provider's SDK to requirements.txt, add its credentials to app/config.py
as env vars (never hardcoded), implement `send()` below, and add "sms" to
NOTIFICATION_CHANNELS.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.services.notifications.base import Notifier

logger = logging.getLogger(__name__)


class SmsNotifier(Notifier):
    name = "sms"

    def send(
        self,
        subject: str,
        body: str,
        recipients: List[str],
        attachment_path: Optional[str] = None,
    ) -> bool:
        logger.warning(
            "SMS notification channel is not configured with a real provider; "
            "dropping message %r intended for %s. See sms_notifier.py to wire one up.",
            subject, recipients,
        )
        return False


class WhatsAppNotifier(Notifier):
    name = "whatsapp"

    def send(
        self,
        subject: str,
        body: str,
        recipients: List[str],
        attachment_path: Optional[str] = None,
    ) -> bool:
        logger.warning(
            "WhatsApp notification channel is not configured with a real provider; "
            "dropping message %r intended for %s. See sms_notifier.py to wire one up.",
            subject, recipients,
        )
        return False
