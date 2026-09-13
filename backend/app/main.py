import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core import rate_limit
from app.core.http_hardening import SecurityHeadersMiddleware, unhandled_exception_handler
from app.core.logging_utils import install_log_redaction
from app.database import Base, SessionLocal, engine
import app.models  # noqa: F401 - ensures all models are registered on Base.metadata
from app.models.camera import Camera
from app.api.routes import admin, alerts, analytics, auth, cameras, realtime, recordings, system, video_uploads
from app.services.detection.engine import detection_engine
from app.services.realtime import realtime_broadcaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# At import time, not on startup: Uvicorn configures its loggers before it
# imports the application, and the very first request can be served before
# any startup hook has finished. Installing here means there is no window
# in which an access line is written unredacted.
install_log_redaction()

# The interactive docs enumerate every route, parameter and schema in the
# application. That is exactly what you want while developing and exactly
# what you do not want to publish from a production deployment, where it
# hands an unauthenticated visitor a complete map of the attack surface
# for free. Dev and test keep /docs, /redoc and /openapi.json unchanged.
_DOCS_ENABLED = settings.ENVIRONMENT != "production"

app = FastAPI(
    title="SentinelCam",
    version="1.0.0",
    docs_url="/docs" if _DOCS_ENABLED else None,
    redoc_url="/redoc" if _DOCS_ENABLED else None,
    openapi_url="/openapi.json" if _DOCS_ENABLED else None,
)

# Never "*": credentials are allowed on these requests, and the browser
# refuses that combination anyway. CORS_ORIGINS is an explicit list.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_exception_handler(Exception, unhandled_exception_handler)

# How often the periodic maintenance task runs (see _maintenance_loop).
MAINTENANCE_INTERVAL_SECONDS = 600

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

    asyncio.get_event_loop().create_task(_maintenance_loop())

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


async def _maintenance_loop():
    """Periodic housekeeping for in-process state that would otherwise
    only grow. Currently just the auth rate limiter's per-(path, IP)
    buckets, which are populated by unauthenticated traffic and so are
    attacker-influenced in size."""
    while True:
        await asyncio.sleep(MAINTENANCE_INTERVAL_SECONDS)
        try:
            pruned = rate_limit.prune_expired()
            if pruned:
                logger.debug("Maintenance: pruned %d expired rate-limit buckets", pruned)
        except Exception:
            logger.exception("Maintenance task iteration failed")


@app.get("/")
def root():
    return {"service": "SentinelCam", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok", "environment": settings.ENVIRONMENT}
