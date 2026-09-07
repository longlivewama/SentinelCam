import os

from fastapi import APIRouter, Depends

from app.config import settings
from app.core.deps import require_operator
from app.models.user import User
from app.services.detection.engine import detection_engine
from app.services.detection.fall_classifier import fall_classifier

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/status")
def get_system_status(_operator: User = Depends(require_operator)):
    return {
        "environment": settings.ENVIRONMENT,
        "model_device": settings.MODEL_DEVICE,
        "models": {
            "pose_model": settings.YOLO_POSE_MODEL,
            "object_model": settings.YOLO_MODEL,
            "pose_object_models_loaded": detection_engine.models_loaded,
            "violence_model_configured": bool(settings.VIOLENCE_MODEL_PATH),
            "fall_classifier_configured": bool(settings.FALL_CLASSIFIER_MODEL_PATH),
            "fall_classifier_loaded": fall_classifier.is_available,
        },
        "detection": {
            "active_camera_detection_loops": detection_engine.active_camera_count,
            "detection_frame_stride": settings.DETECTION_FRAME_STRIDE,
            "video_analysis_frame_stride": settings.VIDEO_ANALYSIS_FRAME_STRIDE,
        },
        "notifications": {
            "channels_enabled": settings.notification_channels,
            "alert_recipients_configured": bool(settings.alert_recipients),
        },
        "storage": {
            "recordings_dir_writable": os.access(settings.RECORDINGS_DIR, os.W_OK) if os.path.isdir(settings.RECORDINGS_DIR) else False,
            "uploads_dir_writable": os.access(settings.UPLOADS_DIR, os.W_OK) if os.path.isdir(settings.UPLOADS_DIR) else False,
        },
    }
