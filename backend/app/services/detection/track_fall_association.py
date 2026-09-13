"""
Deciding WHICH tracked person a fall event belongs to.

The fall gates in `fall_detection.py` answer "did someone fall, when, and
how confidently". They do not answer "who" - and in a scene with several
people, drawing the box on the wrong one is worse than drawing no box at
all, because it makes a confident claim about a person who did not fall.

This module answers only the "who", from geometry the pipeline has
already computed, and it is allowed to answer "I don't know".

Signals, in the order they matter:

  containment  Intersection over the smaller box. The primary signal,
               because the trained detector's box and the pose model's
               box frame different things: mid-fall the fall box is the
               lower body going down *inside* a full-body pose box (IoU
               ~0.45, containment ~0.99), and once the subject is on the
               ground the pose box sits *inside* a wider fall box. IoU
               alone under-scores both.
  IoU          Guards against containment's blind spot: a small box is
               fully "contained" in any large box that swallows it, so a
               fall box covering half the frame would score 1.0 against
               every person in it. IoU falls away in exactly that case.
  centre       A weak tie-breaker between candidates that overlap the
               fall box similarly.
  support      Continuity over time: how strongly this same track matched
               the fall evidence across the preceding processed frames.
               A real fall has the subject under the fall box for the
               whole sustain window, so a passer-by who happens to
               overlap on the firing frame alone does not win.

Failing safe is a first-class outcome. `associate` returns None when the
best candidate is too weak *or* when it is not clearly better than the
runner-up, and the caller then annotates the detector's own region
without inventing a person or a track id.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from app.services.detection.person_tracker import (
    BBox,
    box_diagonal,
    center_distance,
    containment,
    iou,
)

# Weights sum to 1.0. Containment and IoU carry the decision; the centre
# term only separates otherwise-comparable candidates.
WEIGHT_CONTAINMENT = 0.45
WEIGHT_IOU = 0.45
WEIGHT_CENTER = 0.10

# How much of the final score comes from the firing frame versus the
# preceding frames' accumulated evidence. The firing frame dominates -
# it is the frame the gate actually fired on - but continuity gets a real
# vote so a one-frame coincidence cannot outrank a sustained overlap.
WEIGHT_FRAME = 0.75
WEIGHT_SUPPORT = 0.25

# Below this, the best candidate is not credible enough to name. Chosen so
# that a fall box merely adjacent to a person (touching corners, ~0.1 IoU
# and ~0.25 containment, which is what a false positive on a nearby object
# looks like) stays unmatched, while a fall box that is a sub-region of a
# person (~0.45 IoU, ~0.99 containment) matches comfortably.
MIN_ASSOCIATION_SCORE = 0.35

# The best candidate must beat the runner-up by this much. Two people
# overlapping the fall evidence equally well is precisely the case where
# picking one is a coin flip presented as a fact.
AMBIGUITY_MARGIN = 0.08

# Support votes older than this (in video seconds) no longer count.
SUPPORT_WINDOW_SECONDS = 2.0


@dataclass(frozen=True)
class Association:
    """The track a fall was attributed to, and why."""

    track_id: int
    score: float
    frame_score: float
    support: float
    runner_up_score: float

    def as_log_fields(self) -> dict:
        return {
            "track_id": self.track_id,
            "score": round(self.score, 3),
            "frame_score": round(self.frame_score, 3),
            "support": round(self.support, 3),
            "runner_up": round(self.runner_up_score, 3),
        }


def overlap_score(fall_bbox: BBox, person_bbox: BBox) -> float:
    """How well one person box explains one fall box, in [0, 1].

    Zero when the boxes do not touch at all: without any intersection
    there is no evidence to weigh, and a centre-distance-only score would
    let a person standing near a false positive be named as the faller."""
    inter_iou = iou(fall_bbox, person_bbox)
    if inter_iou <= 0:
        return 0.0

    inter_containment = containment(fall_bbox, person_bbox)

    reach = 0.5 * (box_diagonal(fall_bbox) + box_diagonal(person_bbox))
    if reach > 0:
        center_term = max(0.0, 1.0 - center_distance(fall_bbox, person_bbox) / reach)
    else:
        center_term = 0.0

    return (
        WEIGHT_CONTAINMENT * inter_containment
        + WEIGHT_IOU * inter_iou
        + WEIGHT_CENTER * center_term
    )


def best_overlap(fall_bbox: BBox, candidates: Sequence[Tuple[int, BBox]]) -> Tuple[Optional[int], float]:
    """The single best (track_id, score) for one fall box, ungated.

    Used to record per-frame support votes; `associate` is what applies
    the credibility and ambiguity gates."""
    best_id: Optional[int] = None
    best_score = 0.0
    for track_id, bbox in candidates:
        score = overlap_score(fall_bbox, bbox)
        if score > best_score:
            best_score = score
            best_id = track_id
    return best_id, best_score


class SupportLedger:
    """Recent per-frame evidence that a track is the one falling.

    One entry per processed frame, holding every track that overlapped a
    fall box on that frame and how well. Support is then the track's total
    score divided by the number of FRAMES in the window, not by the number
    of frames it happened to appear in - so it measures persistence as
    well as strength. A person under the fall box for the whole sustain
    window scores near their per-frame overlap; one who clips it on a
    single frame scores a fraction of theirs.

    Scoring every overlapping track rather than only the best one is what
    keeps a genuinely ambiguous pair ambiguous: awarding continuity to
    just the leader would quietly break every tie in the leader's favour,
    which is the outcome the ambiguity guard exists to prevent.

    Bounded by time, so it describes at most the last couple of seconds
    however long the video runs.
    """

    def __init__(self, window_seconds: float = SUPPORT_WINDOW_SECONDS):
        self._window = window_seconds
        self._frames: Deque[Tuple[float, Dict[int, float]]] = deque()

    def record_frame(self, video_time_seconds: float, scores: Dict[int, float]) -> None:
        """One processed frame's overlap scores, keyed by track id. Frames
        where nothing overlapped are still recorded (with an empty dict) -
        they are what makes a gap count against continuity."""
        self._frames.append((video_time_seconds, dict(scores)))
        while self._frames and video_time_seconds - self._frames[0][0] > self._window:
            self._frames.popleft()

    def support(self, track_id: int, video_time_seconds: float, window_seconds: Optional[float] = None) -> float:
        """Continuity of `track_id`'s overlap over the window ending now,
        in [0, 1]. Zero for a track with no recent overlap at all - the
        honest answer for someone who has only just come near the fall."""
        window = self._window if window_seconds is None else window_seconds
        recent = [scores for timestamp, scores in self._frames if video_time_seconds - timestamp <= window]
        if not recent:
            return 0.0
        total = sum(scores.get(track_id, 0.0) for scores in recent)
        return min(1.0, total / len(recent))


def associate(
    fall_bbox: BBox,
    candidates: Sequence[Tuple[int, BBox]],
    support_ledger: Optional[SupportLedger] = None,
    video_time_seconds: float = 0.0,
    support_window_seconds: Optional[float] = None,
    min_score: float = MIN_ASSOCIATION_SCORE,
    ambiguity_margin: float = AMBIGUITY_MARGIN,
) -> Optional[Association]:
    """Names the track a fall belongs to, or None when it cannot be named.

    `candidates` is [(track_id, bbox), ...] for the tracks visible on the
    firing frame. Never returns the "first" or "closest" track as a
    fallback: a None here means the caller must not claim a person."""
    if not candidates:
        return None

    scored: List[Tuple[float, float, float, int]] = []
    for track_id, bbox in candidates:
        frame_score = overlap_score(fall_bbox, bbox)
        if frame_score <= 0:
            continue
        support = (
            support_ledger.support(track_id, video_time_seconds, support_window_seconds)
            if support_ledger is not None
            else 0.0
        )
        total = WEIGHT_FRAME * frame_score + WEIGHT_SUPPORT * support
        # track_id last so ties resolve deterministically (lowest id wins)
        # rather than by dict iteration order.
        scored.append((total, frame_score, support, track_id))

    if not scored:
        return None

    scored.sort(key=lambda entry: (-entry[0], entry[3]))
    total, frame_score, support, track_id = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0

    if total < min_score:
        return None
    if len(scored) > 1 and (total - runner_up) < ambiguity_margin:
        return None

    return Association(
        track_id=track_id,
        score=total,
        frame_score=frame_score,
        support=support,
        runner_up_score=runner_up,
    )
