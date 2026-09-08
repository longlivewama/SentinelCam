"""
The frame-iteration core shared by the production upload analyser
(services/video_analysis.py) and the offline validation runner
(ml/validation/).

Why this exists as its own module: the validation framework has to measure
what the *product* actually does. If it re-implemented "open the video,
step frames at the configured stride, compute each frame's position on the
video's own timeline, run the fall pipeline", then every metric it reports
would describe that copy rather than production - and the copy would drift
the first time either side changed. Extracting the loop means there is one
implementation, and the evaluator's numbers are statements about the code
that runs for real users.

What stays OUT of here, deliberately: database writes, progress reporting,
clip extraction, notifications. Those are the upload feature's concerns,
not "what does the detector do to this video", and keeping them in
video_analysis.py is what lets the evaluator run against a bare file path
with no application state at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Sequence

import cv2

from app.config import settings

logger = logging.getLogger(__name__)

# Used when a container reports no (or a nonsensical) frame rate. Video
# time is derived from frame index / fps, so a wrong fps skews every
# reported timestamp - but refusing to process the file would be worse.
DEFAULT_FPS = 25.0


class VideoOpenError(RuntimeError):
    """The file could not be opened or decoded."""


@dataclass(frozen=True)
class VideoProperties:
    fps: float
    frame_count: int
    duration_seconds: Optional[float]


@dataclass
class ScannedFrame:
    """One decoded frame, plus whatever the detection pipeline made of it.

    `processed` is False for frames skipped by the stride - they are still
    yielded, because the caller needs every frame for the pre/post-event
    clip buffer even though no inference ran on them.
    """

    index: int
    frame: object                       # numpy ndarray (BGR), as OpenCV returns
    video_time_seconds: float           # position on the video's own timeline
    processed: bool
    people: Sequence = field(default_factory=tuple)
    fall_events: List[dict] = field(default_factory=list)


def open_video(path: str) -> "tuple[cv2.VideoCapture, VideoProperties]":
    """Opens `path` and reads its properties. Raises VideoOpenError with a
    message safe to show a user (it becomes VideoUpload.error_message)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoOpenError(
            "Could not open uploaded video file - it may be corrupt or in an unsupported format"
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    if fps <= 0:
        fps = DEFAULT_FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = (frame_count / fps) if frame_count else None

    return cap, VideoProperties(fps=fps, frame_count=frame_count, duration_seconds=duration_seconds)


def scan_video(
    cap,
    properties: VideoProperties,
    pipeline,
    extract_people: Optional[Callable] = None,
    stride: Optional[int] = None,
) -> Iterator[ScannedFrame]:
    """Steps through `cap`, running `pipeline` on every stride-th frame,
    and yields one ScannedFrame per decoded frame.

    `pipeline` is a detection/fall_pipeline.FallPipeline (or anything with
    the same `update(frame, people, now)` signature). It carries
    per-subject tracking state, so one instance must drive exactly one
    video.

    `extract_people` is the pose-model call. It is injected rather than
    imported so the validation runner can skip it when the configured mode
    makes it dead weight - in "model" mode FallPipeline ignores `people`
    entirely, so skipping pose changes nothing about the fall events. Left
    None, no pose inference runs and `people` is empty on every frame.

    Time is measured on the VIDEO's timeline (frame index / fps), never
    wall clock: the sustained-duration gate must behave identically
    regardless of how fast the host happens to decode, and a reported
    detection time has to mean "this many seconds into the footage".
    """
    if stride is None:
        stride = settings.VIDEO_ANALYSIS_FRAME_STRIDE
    stride = max(int(stride), 1)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_index % stride == 0:
            video_time_seconds = frame_index / properties.fps
            people = extract_people(frame) if extract_people is not None else ()
            fall_events = pipeline.update(frame, people, now=video_time_seconds)
            yield ScannedFrame(
                index=frame_index,
                frame=frame,
                video_time_seconds=video_time_seconds,
                processed=True,
                people=people,
                fall_events=fall_events,
            )
        else:
            yield ScannedFrame(
                index=frame_index,
                frame=frame,
                video_time_seconds=frame_index / properties.fps,
                processed=False,
            )

        frame_index += 1
