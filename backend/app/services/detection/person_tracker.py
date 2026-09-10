"""
Short-term multi-person tracking for the fall-clip annotation layer.

This is an ANNOTATION-ONLY component. It never decides whether a fall
happened - `fall_pipeline.FallPipeline` and the gates in
`fall_detection.py` remain the sole source of truth for that. All this
does is give every subject visible in a frame a temporary identity, so a
clip can draw a box that follows the *right* person and says which one it
is.

It is object tracking, nothing more: no faces, no identities, no
biometrics. A track id is a number that lives for a few seconds inside
one video analysis and is never persisted, compared across videos, or
associated with a real person.

Why a hand-rolled tracker rather than a library: the project already
ships two nearest-neighbour trackers in this package (the hip-centroid
tracker in `FallDetector` and the box tracker in `ModelFallDetector`),
neither of which uses an external dependency, and the annotation layer
needs the same modest capability - frame-to-frame association over a
handful of boxes at ~6 fps of processed frames. Ultralytics' ByteTrack /
BoT-SORT would pull in `lap`/`scipy` and, more importantly, would have to
run its own detector pass; this tracker consumes boxes the pipeline has
*already* computed, so it adds no inference at all.

Association is greedy over IoU, with a box-size-relative centre-distance
fallback so a subject that moves far between two processed frames (at
stride 5 the gap is ~167ms) is still recognised instead of being split
into a new track. Ties break deterministically, so the same video always
produces the same ids.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]

# An IoU at or above this counts as the same subject outright.
IOU_MATCH_THRESHOLD = 0.2

# Fallback for fast movers and brief detector wobble: a box whose centre
# has moved less than this fraction of the larger box's diagonal is still
# the same subject even when the boxes no longer overlap. Scaled by box
# size rather than fixed pixels, for the same reason ModelFallDetector
# scales its own radius - a subject filling the frame legitimately moves
# many more pixels per frame than a distant one.
CENTER_MATCH_FRACTION = 0.5

# Distance-only matches always score below IOU_MATCH_THRESHOLD so that an
# overlapping candidate wins over a merely nearby one.
_DISTANCE_MATCH_CEILING = IOU_MATCH_THRESHOLD

# A track survives this long without being seen before it is dropped.
# Long enough to ride out the pose model losing a subject for a few
# processed frames (which is common in dark footage), short enough that
# an id is not handed to a different person who walks through later.
MAX_AGE_SECONDS = 1.5

# Observations retained per track. The annotation layer needs the
# pre-event window, so this is sized by the caller from the clip length;
# the default covers ~40s of processed frames at stride 5 / 30fps.
DEFAULT_HISTORY_LENGTH = 240


# --- box geometry (shared with track_fall_association.py) ----------------

def box_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_diagonal(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def intersection_area(a: BBox, b: BBox) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    return width * height


def iou(a: BBox, b: BBox) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(a: BBox, b: BBox) -> float:
    """Intersection over the SMALLER box's area.

    IoU alone is the wrong measure when one box is a sub-region of the
    other, which is exactly what the trained fall detector produces
    mid-fall: it frames the part of the body that is going down while the
    pose model still frames the whole person. Those two boxes can sit at
    IoU ~0.45 while one is 99% inside the other."""
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    smaller = min(box_area(a), box_area(b))
    return inter / smaller if smaller > 0 else 0.0


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


# A box this far inside a larger one is the same subject reported twice.
# NMS runs inside each model at an IoU threshold, which is the wrong
# measure for this artifact: a full-body box and the same body plus an
# outstretched arm sit at IoU ~0.6 - under NMS's threshold, so both
# survive - while one is ~0.99 contained in the other. Left in, the
# duplicate opens a second track that then steals the subject on the next
# frame, and the clip's box changes id half way through.
#
# The cost of being wrong is bounded and mild: two genuinely different
# people, one almost entirely occluded behind the other, are tracked as
# one subject until they separate. For an annotation layer that is a far
# better failure than an id switch, and it cannot affect fall detection.
DUPLICATE_CONTAINMENT = 0.85


def suppress_duplicate_boxes(boxes: Sequence[BBox], containment_threshold: float = DUPLICATE_CONTAINMENT) -> List[BBox]:
    """One box per subject, keeping the larger of any near-duplicate pair.

    Order is by descending area then position, so the result is the same
    for the same input regardless of the detector's ordering."""
    ordered = sorted(
        (tuple(float(v) for v in box) for box in boxes),
        key=lambda box: (-box_area(box), box[0], box[1]),
    )
    kept: List[BBox] = []
    for box in ordered:
        if any(containment(box, larger) >= containment_threshold for larger in kept):
            continue
        kept.append(box)
    return kept


# --- tracks ---------------------------------------------------------------

class PersonTrack:
    """One tracked subject, plus the box history the renderer needs.

    `history` holds only frames on which the subject was actually
    observed. Gaps stay gaps - the renderer interpolates between two real
    observations but never past the last one, so a lost subject produces
    no box rather than an invented one."""

    __slots__ = ("track_id", "bbox", "first_seen", "last_seen", "last_frame_index", "hits", "history")

    def __init__(self, track_id: int, bbox: BBox, now: float, frame_index: int, history_length: int):
        self.track_id = track_id
        self.bbox: BBox = bbox
        self.first_seen = now
        self.last_seen = now
        self.last_frame_index = frame_index
        self.hits = 1
        self.history: Deque[Tuple[int, BBox]] = deque(maxlen=history_length)
        self.history.append((frame_index, bbox))

    def observe(self, bbox: BBox, now: float, frame_index: int) -> None:
        self.bbox = bbox
        self.last_seen = now
        self.last_frame_index = frame_index
        self.hits += 1
        self.history.append((frame_index, bbox))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PersonTrack(id={self.track_id}, bbox={self.bbox}, hits={self.hits})"


class PersonTracker:
    """Greedy IoU tracker over boxes the pipeline already produced.

    One instance per video analysis (or per camera): it carries
    per-subject state, so instances must not be shared across concurrent
    video sources - the same rule the fall gates follow.

    Track ids start at 1 and are never reused, so a stale id can only ever
    resolve to "gone", never to a different person.
    """

    def __init__(
        self,
        max_age_seconds: float = MAX_AGE_SECONDS,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        iou_match_threshold: float = IOU_MATCH_THRESHOLD,
        center_match_fraction: float = CENTER_MATCH_FRACTION,
    ):
        self._tracks: Dict[int, PersonTrack] = {}
        self._next_id = 1
        self._max_age_seconds = max_age_seconds
        self._history_length = max(int(history_length), 2)
        self._iou_match_threshold = iou_match_threshold
        self._center_match_fraction = center_match_fraction

    # -- queries ---------------------------------------------------------

    @property
    def tracks(self) -> List[PersonTrack]:
        return list(self._tracks.values())

    def get(self, track_id: int) -> Optional[PersonTrack]:
        return self._tracks.get(track_id)

    # -- update ----------------------------------------------------------

    def update(self, boxes: Sequence[BBox], now: float, frame_index: int) -> List[PersonTrack]:
        """Associates `boxes` with the live tracks and returns the tracks
        seen on this frame, in the order their boxes were given.

        `now` is the video's own timeline in seconds (frame index / fps),
        never wall clock - the same convention the fall gates use, so
        ageing behaves identically regardless of how fast the host
        decodes."""
        boxes = [tuple(float(v) for v in box) for box in boxes]

        pairs: List[Tuple[float, int, int]] = []
        for det_index, box in enumerate(boxes):
            for track in self._tracks.values():
                score = self._match_score(track.bbox, box)
                if score > 0:
                    pairs.append((score, det_index, track.track_id))

        # Highest score first; the index tie-breakers make the outcome
        # independent of dict ordering, so a given video always yields the
        # same ids.
        pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))

        claimed_detections: set = set()
        claimed_tracks: set = set()
        assignment: Dict[int, int] = {}
        for _score, det_index, track_id in pairs:
            if det_index in claimed_detections or track_id in claimed_tracks:
                continue
            claimed_detections.add(det_index)
            claimed_tracks.add(track_id)
            assignment[det_index] = track_id

        seen: List[PersonTrack] = []
        for det_index, box in enumerate(boxes):
            track_id = assignment.get(det_index)
            if track_id is None:
                track = PersonTrack(self._next_id, box, now, frame_index, self._history_length)
                self._tracks[track.track_id] = track
                self._next_id += 1
            else:
                track = self._tracks[track_id]
                track.observe(box, now, frame_index)
            seen.append(track)

        self._prune(now)
        return seen

    def _match_score(self, track_bbox: BBox, box: BBox) -> float:
        overlap = iou(track_bbox, box)
        if overlap >= self._iou_match_threshold:
            return overlap

        reach = self._center_match_fraction * max(box_diagonal(track_bbox), box_diagonal(box))
        if reach <= 0:
            return 0.0
        distance = center_distance(track_bbox, box)
        if distance > reach:
            return 0.0
        # Strictly below the IoU floor, so any overlapping candidate is
        # preferred over a merely nearby one.
        return _DISTANCE_MATCH_CEILING * (1.0 - distance / reach) * 0.99

    def _prune(self, now: float) -> None:
        stale = [
            track_id for track_id, track in self._tracks.items()
            if now - track.last_seen > self._max_age_seconds
        ]
        for track_id in stale:
            del self._tracks[track_id]
