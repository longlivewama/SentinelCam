import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import Base, SessionLocal, engine
import app.models  # noqa: F401 - ensures all models are registered on Base.metadata
from app.models.camera import Camera
from app.api.routes import admin, alerts, analytics, auth, cameras, realtime, recordings, system, video_uploads
from app.services.detection.engine import detection_engine
from app.services.realtime import realtime_broadcaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SentinelCam", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(cameras.router, prefix="/api")
app.include_router(recordings.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(video_uploads.router, prefix="/api")
app.include_router(realtime.router, prefix="/api")
app.include_router(system.router, prefix="/api")


@app.on_event("startup")
def on_startup():
    if settings.ENVIRONMENT == "production" and settings.JWT_SECRET_KEY == "change-me-in-production":
        raise RuntimeError(
            "JWT_SECRET_KEY is still set to its insecure default while ENVIRONMENT=production. "
            "Set a real secret (e.g. `python -c \"import secrets; print(secrets.token_hex(32))\"`) "
            "in the environment before starting the app."
        )

    # Dev/test convenience: create any missing tables directly from the
    # SQLAlchemy models on boot. For production deployments with real
    # data, use `alembic upgrade head` (see backend/alembic/) for schema
    # changes going forward instead of relying on this - create_all only
    # ever adds missing tables/columns are NOT altered on existing tables.
    Base.metadata.create_all(bind=engine)

    # Bind the running asyncio event loop so background detection/
    # recording/video-analysis threads can push realtime events to
    # connected WebSocket clients via run_coroutine_threadsafe.
    realtime_broadcaster.bind_loop(asyncio.get_event_loop())

    # Resume AI detection for any camera that was left enabled, so
    # surveillance keeps running across restarts without requiring anyone
    # to open that camera's live stream first.
    with SessionLocal() as db:
        cameras = db.query(Camera).filter(
            Camera.is_active == True,  # noqa: E712
            Camera.ai_detection_enabled == True,  # noqa: E712
        ).all()
        for camera in cameras:
            try:
                detection_engine.ensure_running(camera.id, camera.url)
            except Exception:
                logger.exception("Failed to start detection for camera %s", camera.id)


@app.get("/")
def root():
    return {"service": "SentinelCam", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "environment": settings.ENVIRONMENT}
