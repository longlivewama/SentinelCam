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
from app.services.detection.track_fall_association import (
    SupportLedger,
    associate,
    best_overlap,
    overlap_score,
)

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
    ):
        self.fps = fps if fps and fps > 0 else 25.0
        self.stride = max(int(stride), 1)

        clip_seconds = pre_event_seconds + post_event_seconds + HISTORY_MARGIN_SECONDS
        history_length = max(4, int(math.ceil(clip_seconds * self.fps / self.stride)) + 2)

        self._people = PersonTracker(history_length=history_length)
        self._falls = PersonTracker(history_length=history_length)
        self._support = SupportLedger()
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
        # Deduplicated before tracking, never before counting: the
        # upload's own `persons_detected` figure still comes from the raw
        # `people` list in video_analysis.py and is unaffected by this.
        person_boxes = suppress_duplicate_boxes(
            [tuple(p.bbox) for p in people if getattr(p, "bbox", None) is not None]
        )
        fall_bboxes = suppress_duplicate_boxes(
            [tuple(f.bbox) for f in fall_boxes if getattr(f, "bbox", None) is not None]
        )

        person_tracks = self._people.update(person_boxes, video_time_seconds, frame_index)
        fall_tracks = self._falls.update(fall_bboxes, video_time_seconds, frame_index)

        self._last_frame_index = frame_index
        self._person_boxes_now = [(t.track_id, t.bbox) for t in person_tracks]
        self._fall_boxes_now = [(t.track_id, t.bbox) for t in fall_tracks]

        # Continuity evidence: on every processed frame, score every
        # person track against every fall box. `associate` weighs this
        # alongside the firing frame, so a sustained overlap outranks a
        # coincidental one - and a frame where nobody overlapped still
        # counts, as a gap in whoever's continuity.
        frame_scores: dict = {}
        for _fall_track_id, fall_bbox in self._fall_boxes_now:
            for person_track_id, person_bbox in self._person_boxes_now:
                score = overlap_score(fall_bbox, person_bbox)
                if score > 0:
                    frame_scores[person_track_id] = max(frame_scores.get(person_track_id, 0.0), score)
        self._support.record_frame(video_time_seconds, frame_scores)

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

        match = associate(
            evidence,
            self._person_boxes_now,
            support_ledger=self._support,
            video_time_seconds=video_time_seconds,
            support_window_seconds=self._support_window_for(event),
        )

        if match is not None:
            tracker, source_id, track_id = self._people, match.track_id, match.track_id
            detail = match.as_log_fields()
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

    def _support_window_for(self, event: dict) -> Optional[float]:
        """Weigh continuity over the window the gate itself used, so the
        support term describes the same stretch of footage the event was
        built from."""
        sustained = event.get("sustained_seconds")
        if sustained is None:
            return None
        try:
            return max(float(sustained), 0.0) + 1.0
        except (TypeError, ValueError):
            return None

    def _fall_region_track_for(self, evidence: BBox) -> Optional[int]:
        """The fall-region track whose box is the one that just fired."""
        track_id, score = best_overlap(evidence, self._fall_boxes_now)
        if track_id is not None and score > 0:
            return track_id
        return None
