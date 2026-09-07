"""
Notification channel abstraction. Every channel (email today; SMS/WhatsApp
later) implements this interface, so `NotificationService` can fan a single
notification out to whichever channels are configured without the caller
(recording_engine, video_analysis, auth routes) needing to know which
channels exist.

To add a new channel: implement `Notifier`, register it in
`service.py::NotificationService._build_channels`, and add its config to
`app/config.py`. Nothing else in the codebase needs to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class Notifier(ABC):
    """A single delivery channel (email, SMS, WhatsApp, ...)."""

    name: str = "notifier"

    @abstractmethod
    def send(
        self,
        subject: str,
        body: str,
        recipients: List[str],
        attachment_path: Optional[str] = None,
    ) -> bool:
        """Send a notification. Returns True on success, False on failure.
        Must never raise - a broken channel should not crash the caller
        (detection pipeline, password reset flow, etc.)."""
        raise NotImplementedError
