"""
Pins the live-camera event clip window (see recording_engine.py's module
docstring): PRE_EVENT_SECONDS of buffered footage before the fall plus
POST_EVENT_SECONDS collected afterwards. Both default to 5s per product
requirement (a ~10s clip centred on the event, not the whole shift).
"""
import numpy as np

from app.config import settings
from app.services.stream_manager import CameraStream


def test_defaults_are_five_seconds_each_side_of_the_event():
    assert settings.PRE_EVENT_SECONDS == 5
    assert settings.POST_EVENT_SECONDS == 5


def test_the_rolling_pre_event_buffer_holds_pre_event_seconds_of_frames():
    stream = CameraStream(camera_id=999999, url="0")  # never started - no capture device touched
    expected_frames = settings.PRE_EVENT_SECONDS * settings.STREAM_FPS

    frame = np.zeros((4, 4, 3), dtype="uint8")
    for _ in range(expected_frames + 25):
        stream._frame_buffer.append(frame)

    assert len(stream.get_buffer_snapshot()) == expected_frames
