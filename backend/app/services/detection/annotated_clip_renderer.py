"""
Drawing the annotation onto a fall clip's frames.

Two jobs, kept apart from tracking and association so each can be tested
on its own:

  BoxTimeline    turns a track's sparse observations (one per PROCESSED
                 frame - at stride 5 that is every 5th frame) into a box
                 for every frame of the clip, by interpolating between
                 two real observations. It refuses to extrapolate: before
                 the first observation, after the last, or across a gap
                 longer than MAX_INTERPOLATION_GAP_SECONDS it returns
                 None, and the renderer then draws nothing for that frame.
                 A missing box is the correct output for "we lost the
                 subject"; an invented one is not.

  the drawing    a rectangle plus two label bands - "FALL DETECTED" above
                 and the track id + confidence below - sized relative to
                 the frame so it is legible on a 320px clip and on a 4K
                 one.

Nothing here decides anything: it is handed a timeline and a label and it
draws them. It also never mutates a caller's frame - clip frames are
shared with the analyser's pre-event ring buffer and with any other clip
still collecting - so every annotated frame is a copy.

The clip's timing, codec, container and database rows are untouched by
this module. It changes pixels inside frames that were going to be
written anyway.
"""
from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]

# Longest gap between two real observations that may still be bridged by
# interpolation. Comfortably more than one stride at any sane setting
# (stride 5 at 30fps = 0.17s), and short enough that a subject lost for
# nearly a second does not get a straight line drawn through wherever it
# actually went.
MAX_INTERPOLATION_GAP_SECONDS = 0.75

# BGR. Solid red for a fall attributed to a tracked person; amber, drawn
# dashed, for the detector's own region when no track could be named.
COLOR_MATCHED = (0, 0, 255)
COLOR_UNMATCHED = (0, 170, 255)
COLOR_TEXT = (255, 255, 255)

HEADLINE = "FALL DETECTED"
UNMATCHED_DETAIL = "UNMATCHED"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_REFERENCE_WIDTH = 900.0     # font scale 1.0 is tuned for a frame this wide
_MIN_FONT_SCALE = 0.42
_MAX_FONT_SCALE = 1.10
_DASH_LENGTH = 14
_DASH_GAP = 10


@dataclass(frozen=True)
class AnnotationLabel:
    """Exactly what the two bands say, and how to draw them."""

    headline: str
    detail: str
    matched: bool

    @property
    def color(self) -> Tuple[int, int, int]:
        return COLOR_MATCHED if self.matched else COLOR_UNMATCHED


def label_for(track_id: Optional[int], confidence: float) -> AnnotationLabel:
    """`ID 3 | 0.74` when a person track was named, `UNMATCHED | 0.74`
    when it was not. The confidence is the event's own - the number the
    alert and the database row carry - never a re-derived one."""
    if track_id is None:
        return AnnotationLabel(headline=HEADLINE, detail=f"{UNMATCHED_DETAIL} | {confidence:.2f}", matched=False)
    return AnnotationLabel(headline=HEADLINE, detail=f"ID {track_id} | {confidence:.2f}", matched=True)


class BoxTimeline:
    """Per-frame boxes from per-processed-frame observations."""

    def __init__(
        self,
        observations: Sequence[Tuple[int, BBox]],
        max_gap_frames: int,
    ):
        deduped: dict = {}
        for frame_index, bbox in observations:
            deduped[int(frame_index)] = tuple(float(v) for v in bbox)
        self._frames: List[int] = sorted(deduped)
        self._boxes: List[BBox] = [deduped[f] for f in self._frames]
        self._max_gap_frames = max(int(max_gap_frames), 1)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def span(self) -> Optional[Tuple[int, int]]:
        if not self._frames:
            return None
        return self._frames[0], self._frames[-1]

    def box_at(self, frame_index: int) -> Optional[BBox]:
        if not self._frames:
            return None
        if frame_index < self._frames[0] or frame_index > self._frames[-1]:
            return None

        position = bisect.bisect_left(self._frames, frame_index)
        if position < len(self._frames) and self._frames[position] == frame_index:
            return self._boxes[position]

        before, after = position - 1, position
        gap = self._frames[after] - self._frames[before]
        if gap > self._max_gap_frames:
            # The subject was not seen for too long to honestly draw a
            # path between the two sightings.
            return None

        ratio = (frame_index - self._frames[before]) / gap
        start, end = self._boxes[before], self._boxes[after]
        return tuple(start[i] + (end[i] - start[i]) * ratio for i in range(4))


def max_gap_frames_for(fps: float, stride: int) -> int:
    """Interpolation budget in frames, from the analyser's own cadence.

    At least one stride - otherwise nothing between two processed frames
    could ever be filled in - and never more than
    MAX_INTERPOLATION_GAP_SECONDS of footage."""
    fps = fps if fps and fps > 0 else 25.0
    stride = max(int(stride), 1)
    return max(stride, int(round(MAX_INTERPOLATION_GAP_SECONDS * fps)))


def is_renderable(frame) -> bool:
    """Guards the drawing calls against frames a decoder handed back
    broken. A corrupt frame must cost this clip its annotation, not the
    analysis of the whole video."""
    if frame is None:
        return False
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) != 3 or shape[2] != 3:
        return False
    return shape[0] > 0 and shape[1] > 0


def _font_scale(frame_width: int) -> float:
    return max(_MIN_FONT_SCALE, min(_MAX_FONT_SCALE, frame_width / _REFERENCE_WIDTH))


def _clamp_box(bbox: BBox, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = int(round(max(0.0, min(bbox[0], bbox[2]))))
    y1 = int(round(max(0.0, min(bbox[1], bbox[3]))))
    x2 = int(round(min(float(width - 1), max(bbox[0], bbox[2]))))
    y2 = int(round(min(float(height - 1), max(bbox[1], bbox[3]))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return x1, y1, x2, y2


def _draw_dashed_rect(frame, x1: int, y1: int, x2: int, y2: int, color, thickness: int) -> None:
    step = _DASH_LENGTH + _DASH_GAP
    for x in range(x1, x2, step):
        end = min(x + _DASH_LENGTH, x2)
        cv2.line(frame, (x, y1), (end, y1), color, thickness)
        cv2.line(frame, (x, y2), (end, y2), color, thickness)
    for y in range(y1, y2, step):
        end = min(y + _DASH_LENGTH, y2)
        cv2.line(frame, (x1, y), (x1, end), color, thickness)
        cv2.line(frame, (x2, y), (x2, end), color, thickness)


def _draw_band(frame, text: str, left: int, baseline_top: int, color, scale: float, thickness: int) -> None:
    """A filled colour band with the text on it, clamped into the frame."""
    height, width = frame.shape[:2]
    (text_width, text_height), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    pad_x = max(4, int(6 * scale))
    pad_y = max(3, int(5 * scale))

    band_width = text_width + 2 * pad_x
    band_height = text_height + 2 * pad_y

    left = max(0, min(left, width - band_width - 1))
    if left < 0:
        left = 0
    top = max(0, min(baseline_top, height - band_height - 1))

    right = min(width - 1, left + band_width)
    bottom = min(height - 1, top + band_height)
    if right - left < 4 or bottom - top < 4:
        return

    cv2.rectangle(frame, (left, top), (right, bottom), color, cv2.FILLED)
    cv2.putText(
        frame, text, (left + pad_x, bottom - pad_y), _FONT, scale, COLOR_TEXT, thickness, cv2.LINE_AA,
    )


def draw_annotation(frame, bbox: BBox, label: AnnotationLabel) -> bool:
    """Draws the box and both bands onto `frame`, in place.

    Returns False without touching the frame if the box does not land
    inside it - a caller that gets False has written an unannotated but
    otherwise correct frame, which is the intended failure mode."""
    if not is_renderable(frame):
        return False

    height, width = frame.shape[:2]
    clamped = _clamp_box(bbox, width, height)
    if clamped is None:
        return False
    x1, y1, x2, y2 = clamped

    scale = _font_scale(width)
    box_thickness = max(2, int(round(3 * scale)))
    text_thickness = max(1, int(round(2 * scale)))
    color = label.color

    if label.matched:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
    else:
        _draw_dashed_rect(frame, x1, y1, x2, y2, color, box_thickness)

    (_, headline_height), _ = cv2.getTextSize(label.headline, _FONT, scale, text_thickness)
    band_height = headline_height + 2 * max(3, int(5 * scale))

    # Above the box where there is room, otherwise tucked inside its top
    # edge, so the headline is never pushed off-frame.
    headline_top = y1 - band_height - 2
    if headline_top < 0:
        headline_top = y1 + 2
    _draw_band(frame, label.headline, x1, headline_top, color, scale, text_thickness)

    detail_top = y2 + 2
    if detail_top + band_height >= height:
        detail_top = y2 - band_height - 2
    _draw_band(frame, label.detail, x1, detail_top, color, scale, text_thickness)
    return True


def iter_annotated_frames(
    frames: Sequence,
    first_frame_index: int,
    timeline: BoxTimeline,
    label: AnnotationLabel,
) -> Iterator:
    """Yields each clip frame, annotated where the timeline has a box.

    Frames with no box are yielded unchanged and uncopied - most of a
    clip is usually annotated, but a frame outside the track's span costs
    nothing. Frames that do get a box are copied first: the caller's
    frames are shared with the analyser's ring buffer and with any other
    clip still collecting post-event footage, so drawing in place would
    corrupt them."""
    annotated_count = 0
    failures = 0
    for offset, frame in enumerate(frames):
        try:
            bbox = timeline.box_at(first_frame_index + offset)
            if bbox is None or not is_renderable(frame):
                yield frame
                continue
            canvas = frame.copy()
            if draw_annotation(canvas, bbox, label):
                annotated_count += 1
                yield canvas
                continue
        except Exception:
            # A frame that cannot be drawn on - a corrupt decode, an
            # unexpected dtype - loses its box, never the clip. Logged
            # once rather than per frame so a systematically bad clip
            # does not flood the log.
            failures += 1
            if failures == 1:
                logger.exception("Could not annotate a clip frame; writing it unannotated")
        yield frame
    logger.debug(
        "Annotated %d/%d clip frames (%d frame(s) could not be drawn on)",
        annotated_count, len(frames), failures,
    )
