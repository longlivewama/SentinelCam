"""
Application configuration, loaded from environment variables / a .env file
via pydantic-settings. See .env.example for the full list of settings and
sane placeholder values.
"""
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Database ---
    DATABASE_URL: str = "postgresql://sentinelcam:sentinelcam@localhost:5432/sentinelcam"

    # --- Auth / JWT ---
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 24

    # --- SMTP / alert email ---
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = "alerts@sentinelcam.local"
    # Comma-separated list of recipient addresses, e.g. "a@b.com,c@d.com"
    # (kept as a plain string field: pydantic-settings requires strict
    # JSON syntax for env vars typed as List[...], including for an empty
    # value, which is unfriendly for a plain .env file). Use the
    # `alert_recipients` property below to get the parsed list.
    ALERT_RECIPIENTS: str = ""

    @property
    def alert_recipients(self) -> List[str]:
        return [addr.strip() for addr in self.ALERT_RECIPIENTS.split(",") if addr.strip()]

    # --- Storage ---
    RECORDINGS_DIR: str = str(BASE_DIR / "storage" / "recordings")
    SNAPSHOTS_DIR: str = str(BASE_DIR / "storage" / "snapshots")

    # --- Models ---
    YOLO_POSE_MODEL: str = "yolov8n-pose.pt"
    YOLO_MODEL: str = "yolov8n.pt"
    MODEL_DEVICE: str = "cpu"

    # Hook for swapping in a trained temporal action-recognition model
    # (CNN+LSTM / 3D-CNN trained on RWF-2000) for violence detection in
    # production. Unused by the current MVP heuristic implementation.
    VIOLENCE_MODEL_PATH: str = ""

    # --- Streaming / recording ---
    STREAM_FPS: int = 30
    ROLLING_BUFFER_FRAMES: int = 90
    POST_EVENT_SECONDS: int = 3

    # How often (in frames) to run the detection pipeline. Running every
    # frame is unnecessarily expensive on CPU; every 2nd/3rd frame keeps
    # latency low while cutting inference load significantly. Tunable.
    DETECTION_FRAME_STRIDE: int = 3


settings = Settings()
