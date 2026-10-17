"""
Multi-person tracking with pose, for one video source.

`SceneTracker` turns the pose model's per-frame people (boxes + 17
keypoints, already computed by `detection_engine.extract_people`) into
tracked people with a persistent id. It adds no inference: the pose model
IS the person detector, so tracking and pose share one forward pass.

Association is delegated to `person_tracker.PersonTracker` - the tracker
the fall-clip annotation already uses - so a person carries the SAME id in
the live overlay, the uploaded-video overlay, the fall-event association
and the boxes drawn into an annotated clip. Swapping the association
algorithm means passing a different tracker with the same
`update(boxes, now, frame_index) -> [track]` / `get(id)` shape; nothing
else here depends on how matching is done.

Why not Ultralytics' ByteTrack/BoT-SORT: at the pinned ultralytics 8.3.0
its tracker module imports `lap` (not installed) and falls back to
`check_requirements`, i.e. a `pip install` at runtime inside the server;
and its track ids come from a class-level counter shared by every tracker
instance, so resetting one camera's tracker renumbers every other camera
and video. Session isolation is a hard requirement here, so the
dependency-free tracker stays.

A track id is a temporary handle for one trajectory inside one session -
one camera detection loop, or one video analysis. It is not an identity:
it changes after occlusion longer than `max_age_seconds`, and the same
person in two sessions gets unrelated ids. Nothing here touches faces or
biometrics.

State per track: `tentative` until it has been observed on
`min_hits` processed frames (a single-frame false detection never reaches
`confirmed`), then `confirmed`; a confirmed track that was not observed on
the latest frame but is still within `max_age_seconds` is `lost` and is
kept so a brief detection drop does not cost the person their id.
High-frequency data stays in memory; only the caller decides what, if
anything, to persist.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from app.services.detection.person_tracker import (
    BBox,
    MAX_AGE_SECONDS,
    PersonTracker,
    suppress_duplicate_boxes,
)
from app.services.detection.track_fall_association import (
    Association,
    SupportLedger,
    associate,
    overlap_score,
)
from app.services.detection.pose import (
    DEFAULT_KEYPOINT_MIN_CONFIDENCE,
    Keypoint,
    normalize_keypoints,
    torso_angle_degrees,
)

STATE_TENTATIVE = "tentative"
STATE_CONFIRMED = "confirmed"
STATE_LOST = "lost"

DEFAULT_MIN_HITS = 2

# Recent pose samples kept per track, for posture evidence over a short
# window: ~4 s at the default upload stride, ~2.4 s on a live camera.
# `recent_torso_angles` then trims by time.
POSE_HISTORY_LENGTH = 24

# Lifetime summaries of tracks that have ended. Bounded so a camera that
# runs for months does not accumulate one entry per passer-by forever; a
# video analysis is bounded by its own length long before this matters.
DEFAULT_MAX_FINISHED_SUMMARIES = 500


@dataclass
class TrackedPerson:
    """One tracked person as of the latest processed frame."""

    track_id: int
    bbox: BBox
    confidence: Optional[float]
    keypoints: List[Keypoint]
    first_seen: float
    last_seen: float
    hits: int
    state: str
    torso_angle: Optional[float] = None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


@dataclass
class TrackSummary:
    """A track's whole lifetime, for review after the fact."""

    track_id: int
    first_seen: float
    last_seen: float
    frames_observed: int
    max_confidence: Optional[float]

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)


@dataclass
class _TrackState:
    confidence: Optional[float] = None
    keypoints: List[Keypoint] = field(default_factory=list)
    torso_angle: Optional[float] = None
    pose_history: Deque[Tuple[float, Optional[float]]] = field(
        default_factory=lambda: deque(maxlen=POSE_HISTORY_LENGTH)
    )
    max_confidence: Optional[float] = None


class SceneTracker:
    """One instance per video source. Not thread-safe and not shareable
    across sources - the same rule as `FallPipeline`, which owns one."""

    def __init__(
        self,
        tracker: Optional[PersonTracker] = None,
        min_hits: int = DEFAULT_MIN_HITS,
        keypoint_min_confidence: float = DEFAULT_KEYPOINT_MIN_CONFIDENCE,
        max_age_seconds: float = MAX_AGE_SECONDS,
        max_finished_summaries: int = DEFAULT_MAX_FINISHED_SUMMARIES,
    ):
        self.tracker = tracker or PersonTracker(max_age_seconds=max_age_seconds)
        self.min_hits = max(int(min_hits), 1)
        self.keypoint_min_confidence = keypoint_min_confidence
        self._states: Dict[int, _TrackState] = {}
        self._summaries: Dict[int, TrackSummary] = {}
        self._finished: Deque[int] = deque()
        self._max_finished = max(int(max_finished_summaries), 0)
        self._current: List[TrackedPerson] = []
        self._seen_ids_now: set = set()
        self.last_frame_index: Optional[int] = None
        self.last_time: Optional[float] = None
        self._support = SupportLedger()
        self._support_frame_index: Optional[int] = None

    # -- update ----------------------------------------------------------

    def update(self, people: Sequence, now: float, frame_index: int) -> List[TrackedPerson]:
        """Feeds one processed frame's people. `now` is the source's own
        timeline in seconds (video time for uploads, wall clock for live
        cameras). Returns every live track: observed ones first, then
        tracks currently `lost`."""
        by_box: Dict[BBox, object] = {}
        for person in people:
            bbox = getattr(person, "bbox", None)
            if bbox is None:
                continue
            key = tuple(float(v) for v in bbox)
            # The first person reported for an identical box wins; exact
            # duplicates carry no extra information.
            by_box.setdefault(key, person)

        # Same de-duplication the annotation layer always applied, so the
        # ids here are exactly the ids drawn into annotated clips.
        boxes = suppress_duplicate_boxes(list(by_box.keys()))
        tracks = self.tracker.update(boxes, now, frame_index)

        self.last_frame_index = frame_index
        self.last_time = now
        self._seen_ids_now = set()

        observed: List[TrackedPerson] = []
        for track in tracks:
            person = by_box.get(tuple(track.bbox))
            state = self._states.setdefault(track.track_id, _TrackState())
            confidence = _confidence_of(person)
            keypoints = normalize_keypoints(getattr(person, "keypoints", None)) if person is not None else []
            angle = torso_angle_degrees(keypoints, self.keypoint_min_confidence) if keypoints else None

            state.confidence = confidence
            state.keypoints = keypoints
            state.torso_angle = angle
            state.pose_history.append((now, angle))
            if confidence is not None:
                state.max_confidence = confidence if state.max_confidence is None else max(state.max_confidence, confidence)

            self._seen_ids_now.add(track.track_id)
            observed.append(self._as_person(track, state, observed_now=True))
            self._summaries[track.track_id] = TrackSummary(
                track_id=track.track_id,
                first_seen=track.first_seen,
                last_seen=track.last_seen,
                frames_observed=track.hits,
                max_confidence=state.max_confidence,
            )

        lost: List[TrackedPerson] = []
        live_ids = set()
        for track in self.tracker.tracks:
            live_ids.add(track.track_id)
            if track.track_id in self._seen_ids_now:
                continue
            state = self._states.get(track.track_id)
            if state is None or track.hits < self.min_hits:
                continue
            lost.append(self._as_person(track, state, observed_now=False))

        self._forget_dead(live_ids)
        self._current = observed + lost
        return list(self._current)

    def _as_person(self, track, state: _TrackState, observed_now: bool) -> TrackedPerson:
        if not observed_now:
            status = STATE_LOST
        elif track.hits >= self.min_hits:
            status = STATE_CONFIRMED
        else:
            status = STATE_TENTATIVE
        return TrackedPerson(
            track_id=track.track_id,
            bbox=tuple(track.bbox),
            confidence=state.confidence,
            keypoints=list(state.keypoints),
            first_seen=track.first_seen,
            last_seen=track.last_seen,
            hits=track.hits,
            state=status,
            torso_angle=state.torso_angle,
        )

    def _forget_dead(self, live_ids: set) -> None:
        for track_id in [tid for tid in self._states if tid not in live_ids]:
            del self._states[track_id]
            if track_id in self._summaries:
                self._finished.append(track_id)
        while len(self._finished) > self._max_finished:
            self._summaries.pop(self._finished.popleft(), None)

    # -- fall attribution --------------------------------------------------

    def record_fall_evidence(self, fall_boxes: Sequence, frame_index: int) -> None:
        """Records how well each person observed on this frame overlaps the
        fall detector's boxes for the same frame. Called once per processed
        frame, after `update()`; a second call for the same frame is a
        no-op, so the pipeline and the clip annotator can both call it.

        This continuity record is what lets attribution prefer the person
        the detector has been firing on for the whole sustain window over
        one who merely overlaps on the final frame."""
        if self._support_frame_index == frame_index or self.last_time is None:
            return
        self._support_frame_index = frame_index
        fall_bboxes = suppress_duplicate_boxes(
            [tuple(f.bbox) for f in fall_boxes if getattr(f, "bbox", None) is not None]
        )
        scores: Dict[int, float] = {}
        people = self.observed_boxes()
        for fall_bbox in fall_bboxes:
            for track_id, person_bbox in people:
                score = overlap_score(fall_bbox, person_bbox)
                if score > 0:
                    scores[track_id] = max(scores.get(track_id, 0.0), score)
        self._support.record_frame(self.last_time, scores)

    def attribute(self, evidence_bbox: BBox, sustained_seconds: Optional[float] = None) -> Optional[Association]:
        """The tracked person a fall box belongs to, or None when no person
        can be credibly named (no overlap, too weak, or two candidates too
        close to call). Never falls back to "nearest person".

        The single attribution rule for the whole system: the fall pipeline
        uses it to stamp `person_track_id` on events and the clip annotator
        uses it to decide whose box to draw, so both always agree."""
        window = None
        if sustained_seconds is not None:
            try:
                window = max(float(sustained_seconds), 0.0) + 1.0
            except (TypeError, ValueError):
                window = None
        return associate(
            tuple(float(v) for v in evidence_bbox),
            self.observed_boxes(),
            support_ledger=self._support,
            video_time_seconds=self.last_time or 0.0,
            support_window_seconds=window,
        )

    # -- queries ---------------------------------------------------------

    @property
    def current(self) -> List[TrackedPerson]:
        """Tracks as of the latest frame (observed, then lost)."""
        return list(self._current)

    def observed_boxes(self) -> List[Tuple[int, BBox]]:
        """(track_id, bbox) for every person observed on the latest frame,
        tentative included - the candidate list fall association and the
        clip annotator have always used."""
        return [(p.track_id, p.bbox) for p in self._current if p.state != STATE_LOST]

    def get(self, track_id: int) -> Optional[TrackedPerson]:
        for person in self._current:
            if person.track_id == track_id:
                return person
        return None

    def recent_torso_angles(self, track_id: int, window_seconds: float) -> List[float]:
        """Torso angles measured for `track_id` within the last
        `window_seconds` of this source's timeline. Frames where no torso
        was visible are skipped, never counted as upright or fallen."""
        state = self._states.get(track_id)
        if state is None or self.last_time is None:
            return []
        cutoff = self.last_time - max(window_seconds, 0.0)
        return [angle for t, angle in state.pose_history if t >= cutoff and angle is not None]

    def torso_angles_after(self, track_id: int, after: float) -> Optional[List[float]]:
        """Torso angles measured for `track_id` strictly after `after`, in
        time order - or None once the track has ended, which is different
        from a live track that simply had no torso visible."""
        state = self._states.get(track_id)
        if state is None:
            return None
        return [angle for t, angle in state.pose_history if t > after and angle is not None]

    def summaries(self) -> List[TrackSummary]:
        """Every track this session has seen (bounded), oldest id first."""
        return [self._summaries[tid] for tid in sorted(self._summaries)]


def _confidence_of(person) -> Optional[float]:
    if person is None:
        return None
    value = getattr(person, "confidence", None)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
