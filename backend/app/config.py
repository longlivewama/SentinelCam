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

    # --- Environment ---
    ENVIRONMENT: str = "development"  # "development" | "production" | "test"

    # --- Database ---
    DATABASE_URL: str = "postgresql://sentinelcam:sentinelcam@localhost:5432/sentinelcam"

    # --- Auth / JWT ---
    JWT_SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 24
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = 30

    # --- Auth rate limiting ---
    # Sensitive unauthenticated endpoints are rate-limited per client IP
    # (see core/rate_limit.py). The defaults below are the intended
    # production values; they're configurable only so a test environment
    # that drives many auth requests from a single IP (the e2e suite) can
    # opt out. Do not relax these outside a test environment.
    AUTH_SIGNUP_MAX_REQUESTS: int = 5
    AUTH_SIGNUP_WINDOW_SECONDS: float = 60
    AUTH_LOGIN_MAX_REQUESTS: int = 10
    AUTH_LOGIN_WINDOW_SECONDS: float = 60
    AUTH_FORGOT_PASSWORD_MAX_REQUESTS: int = 5
    AUTH_FORGOT_PASSWORD_WINDOW_SECONDS: float = 300
    AUTH_RESET_PASSWORD_MAX_REQUESTS: int = 10
    AUTH_RESET_PASSWORD_WINDOW_SECONDS: float = 60

    # --- CORS ---
    # Comma-separated list of allowed origins for the frontend. Kept as a
    # plain string (see ALERT_RECIPIENTS below for why) rather than
    # List[str] so a plain .env file doesn't need JSON syntax.
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    @property
    def cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    # --- Frontend (used to build links in emails, e.g. password reset) ---
    FRONTEND_URL: str = "http://localhost:5173"

    # --- Notifications ---
    # Comma-separated list of enabled channels. "email" works out of the
    # box; "sms"/"whatsapp" are architectural stubs (see
    # services/notifications/sms_notifier.py) until a real provider is wired in.
    NOTIFICATION_CHANNELS: str = "email"

    @property
    def notification_channels(self) -> List[str]:
        return [c.strip() for c in self.NOTIFICATION_CHANNELS.split(",") if c.strip()]

    # --- SMTP / alert email ---
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    # Most public mail providers (Gmail, etc.) require STARTTLS; some
    # internal/local relays (e.g. a Mailpit/MailHog catcher used for local
    # dev and e2e tests) don't support it at all - set to false for those.
    SMTP_USE_TLS: bool = True
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
    UPLOADS_DIR: str = str(BASE_DIR / "storage" / "uploads")

    # --- Video upload ---
    MAX_UPLOAD_SIZE_MB: int = 500

    # Total bytes of uploaded video one account may hold at once. Without
    # a cap, a single user can fill the disk one within-limit upload at a
    # time and take recording + analysis down for everyone. Set to 0 to
    # disable the quota.
    MAX_UPLOAD_STORAGE_PER_USER_MB: int = 5000
    ALLOWED_VIDEO_EXTENSIONS: str = ".mp4,.mov,.avi,.mkv,.webm"

    @property
    def allowed_video_extensions(self) -> List[str]:
        return [e.strip().lower() for e in self.ALLOWED_VIDEO_EXTENSIONS.split(",") if e.strip()]

    # How many frames to skip between processed frames when analyzing an
    # uploaded video (independent of live-camera DETECTION_FRAME_STRIDE,
    # since uploaded video is processed as fast as possible rather than in
    # real time).
    VIDEO_ANALYSIS_FRAME_STRIDE: int = 5

    # --- Models ---
    YOLO_POSE_MODEL: str = "yolov8n-pose.pt"
    YOLO_MODEL: str = "yolov8n.pt"
    MODEL_DEVICE: str = "cpu"

    # Hook for swapping in a trained temporal action-recognition model
    # (CNN+LSTM / 3D-CNN trained on RWF-2000) for violence detection in
    # production. Unused by the current MVP heuristic implementation.
    VIOLENCE_MODEL_PATH: str = ""

    # Hook for the trained fall classifier exported by ml/export.py (see
    # ml/README.md). When set to an existing file, fall_detection.py uses
    # it as a corroborating signal alongside the pose heuristic. Left
    # empty, fall detection runs on the heuristic alone.
    FALL_CLASSIFIER_MODEL_PATH: str = ""

    # --- Trained fall detector (the primary fall signal) ---
    # Path to the fine-tuned single-class YOLO "Fall" detector produced by
    # ml/detector/train.py and promoted to ml/exported/fall_detector_v1.pt
    # (see ml/MODEL_CARD.md). Unlike FALL_CLASSIFIER_MODEL_PATH above -
    # a keypoint MLP that only nudges confidence - this model detects
    # fallen people directly and, when loaded, becomes the source of truth
    # for fall events. Points at the committed artifact by default so a
    # fresh checkout runs the real model with no extra configuration;
    # set to "" to force the pose heuristic instead.
    FALL_DETECTOR_MODEL_PATH: str = str(BASE_DIR.parent / "ml" / "exported" / "fall_detector_v1.pt")

    # Which fall-detection strategy to run:
    #   "auto"      - the trained detector when it loads, else the pose
    #                 heuristic (default; degrades gracefully).
    #   "model"     - trained detector only; no fall events at all if it
    #                 cannot be loaded.
    #   "heuristic" - pose heuristic only, ignoring the trained detector.
    #   "hybrid"    - run both and report the union of their events
    #                 (higher recall, more false positives - see
    #                 ml/MODEL_CARD.md before enabling in production).
    FALL_DETECTION_MODE: str = "auto"

    # Minimum per-box confidence for the trained detector's output to be
    # considered at all. 0.4 sits just below the F1-optimal operating
    # point measured in ml/MODEL_CARD.md; the sustained-duration gate
    # below is what actually suppresses one-frame false positives.
    FALL_DETECTOR_MIN_CONFIDENCE: float = 0.4

    # How long a "Fall" box must persist for the same tracked subject
    # before an event fires. Shorter than the pose heuristic's 1.2s gate
    # because the model recognises the posture directly rather than
    # inferring it from a geometric proxy, but still long enough that a
    # single bad frame cannot raise an alert.
    FALL_DETECTOR_MIN_SUSTAINED_SECONDS: float = 0.6

    # Upper bound on video-upload analyses running at once. Each one
    # decodes a video and runs inference on a background thread, so
    # unbounded concurrency is a straightforward way to exhaust CPU and
    # memory; further uploads queue instead (status stays "pending").
    MAX_CONCURRENT_VIDEO_ANALYSES: int = 2

    # --- Streaming / recording ---
    STREAM_FPS: int = 30
    ROLLING_BUFFER_FRAMES: int = 90
    POST_EVENT_SECONDS: int = 3

    # How often (in frames) to run the detection pipeline. Running every
    # frame is unnecessarily expensive on CPU; every 2nd/3rd frame keeps
    # latency low while cutting inference load significantly. Tunable.
    DETECTION_FRAME_STRIDE: int = 3


settings = Settings()
