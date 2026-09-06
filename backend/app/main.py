import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, SessionLocal, engine
import app.models  # noqa: F401 - ensures all models are registered on Base.metadata
from app.models.camera import Camera
from app.api.routes import auth, cameras, recordings, admin
from app.services.detection.engine import detection_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SentinelCam", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(cameras.router, prefix="/api")
app.include_router(recordings.router, prefix="/api")
app.include_router(admin.router, prefix="/api")


@app.on_event("startup")
def on_startup():
    # No migrations tool in this project's scope - create any missing
    # tables directly from the SQLAlchemy models on boot.
    Base.metadata.create_all(bind=engine)

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
