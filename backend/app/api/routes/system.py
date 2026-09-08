import os

from fastapi import APIRouter, Depends

from app.config import settings
from app.core.deps import require_operator
from app.models.user import User
from app.services.detection.engine import detection_engine
from app.services.detection.fall_classifier import fall_classifier
from app.services.detection.fall_object_detector import fall_object_detector
from app.services.detection.fall_pipeline import resolve_mode

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
            # The trained single-class YOLO fall detector - the primary
            # fall signal when it loads. `fall_detection_mode` below is
            # what is actually running, which is the number an operator
            # needs: a configured-but-unloadable model silently falls back
            # to the weaker heuristic in "auto" mode, and this is where
            # that shows up.
            "fall_detector": fall_object_detector.model_info,
        },
        "detection": {
            "active_camera_detection_loops": detection_engine.active_camera_count,
            "detection_frame_stride": settings.DETECTION_FRAME_STRIDE,
            "video_analysis_frame_stride": settings.VIDEO_ANALYSIS_FRAME_STRIDE,
            "fall_detection_mode_configured": settings.FALL_DETECTION_MODE,
            "fall_detection_mode_active": resolve_mode(),
            "max_concurrent_video_analyses": settings.MAX_CONCURRENT_VIDEO_ANALYSES,
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
