"""
The seam between the fall pipeline and the annotation layer.

`video_analysis.py` drives this with two calls and nothing else:

    annotator.observe(frame_index, video_time, people, fall_boxes)   # per processed frame
    plan = annotator.plan_for_event(event, frame_index, video_time)  # when a fall fires

and hands the returned plan to the clip writer. Everything the annotation
needs - who was where, which of them the fall belongs to, and the boxes to
draw on every frame of the clip - is derived from data the pipeline had
already computed for its own reasons. No model runs here, and no
inference is added to the analysis.

Explicitly NOT this module's business: whether a fall happened, when it
happened, how confident it was, which frames the clip spans, or whether a
clip is written at all. Those are decided upstream and are passed through
untouched. If everything in this file failed, the same events, clips,
alerts and rows would still be produced - just without boxes drawn on
them.

--- how a plan stays correct across the post-event window ---

A fall clip is PRE_EVENT_SECONDS of buffered footage plus
POST_EVENT_SECONDS collected after the event, so the clip is written
several seconds after the plan is created. A plan therefore subscribes to
its track: it is seeded with the track's history (the pre-event boxes)
and keeps receiving observations on every later processed frame until the
writer releases it. That way the tracker's own pruning - which must stay
aggressive so ids are not recycled onto other people - cannot take the
second half of a clip's boxes with it.

--- when the person cannot be named ---

`track_fall_association` is allowed to answer "I don't know", and on real
footage it does: a detector false positive on a dark car door has no
person under it to name. Rather than pick the nearest person and assert
they fell, the plan then follows the DETECTOR'S OWN region - real boxes
from the model, tracked frame to frame - and the clip is labelled
`UNMATCHED` with no track id. Nothing is invented in either branch.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from app.services.detection.annotated_clip_renderer import (
    AnnotationLabel,
    BoxTimeline,
    label_for,
    max_gap_frames_for,
)
from app.services.detection.person_tracker import BBox, PersonTracker, suppress_duplicate_boxes
from app.services.detection.scene_tracking import SceneTracker
from app.services.detection.track_fall_association import best_overlap

logger = logging.getLogger(__name__)

# Extra seconds of track history kept beyond the clip window, so a plan
# created a moment late still has the whole pre-event span available.
HISTORY_MARGIN_SECONDS = 2.0


@dataclass
class AnnotationPlan:
    """What to draw on one fall clip.

    `track_id` is None when no person could be credibly named; the boxes
    are then the fall detector's own region. `observations` grows while
    the clip is still collecting post-event frames and is frozen by
    `release()`."""

    confidence: float
    track_id: Optional[int]
    observations: List[Tuple[int, BBox]] = field(default_factory=list)
    max_gap_frames: int = 1
    fired_at_frame: int = 0
    association_detail: Optional[dict] = None

    # Internal: which tracker/track this plan is still listening to.
    closed: bool = False
    _tracker: Optional[PersonTracker] = None
    _source_track_id: Optional[int] = None

    @property
    def matched(self) -> bool:
        return self.track_id is not None

    @property
    def label(self) -> AnnotationLabel:
        return label_for(self.track_id, self.confidence)

    def timeline(self) -> BoxTimeline:
        return BoxTimeline(self.observations, self.max_gap_frames)

    def _absorb(self, frame_index: int, bbox: BBox) -> None:
        if self.observations and self.observations[-1][0] == frame_index:
            return
        self.observations.append((frame_index, bbox))


class FallClipAnnotator:
    """Per-video-analysis annotation state.

    Holds two trackers over boxes the pipeline already produced on every
    processed frame:

      people  the pose model's person boxes. These carry the track ids a
              clip can display, because "a person is here" is what a
              person detector is entitled to assert.
      falls   the trained detector's `Fall` boxes. Tracked only so an
              unmatched fall still has a coherent region to follow across
              the clip; these ids are never displayed.

    One instance per video, like `FallPipeline` and the gates - it carries
    per-subject state.
    """

    def __init__(
        self,
        fps: float,
        stride: int,
        pre_event_seconds: float,
        post_event_seconds: float,
        scene: Optional[SceneTracker] = None,
    ):
        self.fps = fps if fps and fps > 0 else 25.0
        self.stride = max(int(stride), 1)

        self.history_length = self.history_length_for(
            self.fps, self.stride, pre_event_seconds, post_event_seconds,
        )

        # The person tracker is a SceneTracker's: either the one the fall
        # pipeline owns (the upload analyser hands it in, so a clip's drawn
        # id, the event's person_track_id and the overlay all agree), or a
        # private one when none is given. Its box history must cover the
        # clip window - see history_length_for.
        self._scene = scene or SceneTracker(tracker=PersonTracker(history_length=self.history_length))
        self._people = self._scene.tracker
        self._falls = PersonTracker(history_length=self.history_length)
        self._max_gap_frames = max_gap_frames_for(self.fps, self.stride)
        self._open_plans: List[AnnotationPlan] = []

        # Last processed frame's state, used to answer plan_for_event.
        self._last_frame_index = -1
        self._person_boxes_now: List[Tuple[int, BBox]] = []
        self._fall_boxes_now: List[Tuple[int, BBox]] = []

    # -- per processed frame ---------------------------------------------

    def observe(
        self,
        frame_index: int,
        video_time_seconds: float,
        people: Sequence,
        fall_boxes: Sequence,
    ) -> None:
        """Feeds one processed frame's detections to both trackers.

        `people` is whatever `detection_engine.extract_people` returned
        (anything with a `.bbox`); `fall_boxes` is the trained detector's
        output for the same frame (anything with a `.bbox`). Both were
        computed by the pipeline for its own purposes - this only reads
        them."""
        # A shared scene has already seen this frame (the fall pipeline
        # updates it before the analyser gets here); updating it twice
        # would age every track by a phantom frame. Person boxes are
        # deduplicated inside the scene before tracking, never before
        # counting: the upload's `persons_detected` still comes from the
        # raw `people` list in video_analysis.py.
        if self._scene.last_frame_index != frame_index:
            self._scene.update(people, video_time_seconds, frame_index)
        # Continuity evidence for attribution (idempotent per frame): every
        # person observed now, scored against every fall box now.
        self._scene.record_fall_evidence(fall_boxes, frame_index)

        fall_bboxes = suppress_duplicate_boxes(
            [tuple(f.bbox) for f in fall_boxes if getattr(f, "bbox", None) is not None]
        )
        fall_tracks = self._falls.update(fall_bboxes, video_time_seconds, frame_index)

        self._last_frame_index = frame_index
        self._person_boxes_now = self._scene.observed_boxes()
        self._fall_boxes_now = [(t.track_id, t.bbox) for t in fall_tracks]

        self._advance_open_plans(frame_index)

    def _advance_open_plans(self, frame_index: int) -> None:
        for plan in self._open_plans:
            tracker = plan._tracker
            if tracker is None or plan._source_track_id is None:
                continue
            track = tracker.get(plan._source_track_id)
            if track is not None and track.last_frame_index == frame_index:
                plan._absorb(frame_index, track.bbox)

    # -- when a fall fires -------------------------------------------------

    def plan_for_event(
        self,
        event: dict,
        frame_index: int,
        video_time_seconds: float,
    ) -> Optional[AnnotationPlan]:
        """Builds the annotation plan for one fall event.

        `event` is the dict the fall gate produced. The only key read
        beyond `confidence` is `bbox` - the box the gate actually fired
        on - so an event from either strategy works and neither strategy's
        behaviour is affected. Returns None when the event carries no box
        (nothing to anchor an annotation to), and the clip is then written
        exactly as it was before this feature existed."""
        evidence = event.get("bbox")
        if evidence is None:
            return None
        evidence = tuple(float(v) for v in evidence)
        confidence = float(event.get("confidence", 0.0))

        if "person_track_id" in event:
            # The pipeline already attributed this event when it fired. A
            # held event reaches here frames later, and re-attributing its
            # fire-time box against the people visible NOW could name
            # someone else - the clip must show the person the event,
            # its overlay and its record name.
            attributed = event["person_track_id"]
            match = None
        else:
            match = self._scene.attribute(evidence, event.get("sustained_seconds"))
            attributed = match.track_id if match is not None else None

        if attributed is not None:
            tracker, source_id, track_id = self._people, attributed, attributed
            detail = match.as_log_fields() if match is not None else None
            logger.info(
                "Fall at %.2fs attributed to person track %s (%s)",
                video_time_seconds, track_id, detail,
            )
        else:
            source_id = self._fall_region_track_for(evidence)
            tracker, track_id, detail = self._falls, None, None
            logger.info(
                "Fall at %.2fs could not be attributed to a tracked person (%d candidate(s)); "
                "annotating the detector's own region without a track id",
                video_time_seconds, len(self._person_boxes_now),
            )
            if source_id is None:
                return None

        plan = AnnotationPlan(
            confidence=confidence,
            track_id=track_id,
            max_gap_frames=self._max_gap_frames,
            fired_at_frame=frame_index,
            association_detail=detail,
        )
        plan._tracker = tracker
        plan._source_track_id = source_id

        track = tracker.get(source_id)
        if track is not None:
            for observed_frame, bbox in track.history:
                plan._absorb(observed_frame, bbox)

        self._open_plans.append(plan)
        return plan

    def release(self, plan: Optional[AnnotationPlan]) -> None:
        """Stops feeding a plan - called once its clip has been collected.
        Idempotent, so the analyser's normal path and its
        end-of-video flush can both call it."""
        if plan is None:
            return
        plan.closed = True
        plan._tracker = None
        try:
            self._open_plans.remove(plan)
        except ValueError:
            pass

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def history_length_for(fps: float, stride: int, pre_event_seconds: float, post_event_seconds: float) -> int:
        """Processed frames of box history a track must keep so an
        annotation plan can be seeded with the clip's whole pre-event
        window."""
        fps = fps if fps and fps > 0 else 25.0
        clip_seconds = pre_event_seconds + post_event_seconds + HISTORY_MARGIN_SECONDS
        return max(4, int(math.ceil(clip_seconds * fps / max(int(stride), 1))) + 2)

    def _fall_region_track_for(self, evidence: BBox) -> Optional[int]:
        """The fall-region track whose box is the one that just fired."""
        track_id, score = best_overlap(evidence, self._fall_boxes_now)
        if track_id is not None and score > 0:
            return track_id
        return None
