"""
Unit tests for the sustained-on-ground-posture fall detection logic (the
false-positive fix described in fall_detection.py's module docstring).
"""
from app.services.detection.fall_detection import (
    FALL_DEBOUNCE_SECONDS,
    FALL_MIN_SUSTAINED_SECONDS,
    FallDetector,
    PersonDetection,
)

CONF = 1.0


def _keypoints(shoulder_y: float, hip_y: float) -> list:
    kps = [(0.0, 0.0, 0.0)] * 17
    kps[5] = (100.0, shoulder_y, CONF)
    kps[6] = (120.0, shoulder_y, CONF)
    kps[11] = (100.0, hip_y, CONF)
    kps[12] = (120.0, hip_y, CONF)
    return kps


def _standing_person(hip_y=300.0) -> PersonDetection:
    # Narrow/tall bbox, shoulders clearly above hips.
    return PersonDetection(bbox=(90, 50, 140, 350), keypoints=_keypoints(shoulder_y=100.0, hip_y=hip_y))


def _fallen_person(hip_y=300.0) -> PersonDetection:
    # Wide/short bbox, shoulders bunched near hips (lying down).
    return PersonDetection(bbox=(50, 280, 250, 340), keypoints=_keypoints(shoulder_y=290.0, hip_y=hip_y))


def test_standing_person_never_fires():
    detector = FallDetector()
    events = []
    for i in range(50):
        events += detector.update([_standing_person()], now=i * 0.1)
    assert events == []


def test_brief_posture_change_does_not_fire_a_fall():
    """A quick bend/sit that recovers before the sustain threshold must
    NOT be flagged - this is the exact false-positive this logic exists
    to prevent."""
    detector = FallDetector()
    events = []
    # On the ground for well under FALL_MIN_SUSTAINED_SECONDS, then recovers.
    brief_duration = FALL_MIN_SUSTAINED_SECONDS * 0.4
    steps = 5
    for i in range(steps):
        t = (i / (steps - 1)) * brief_duration
        events += detector.update([_fallen_person()], now=t)
    # Recovers to standing.
    events += detector.update([_standing_person()], now=brief_duration + 0.1)
    assert events == []


def test_sustained_fall_fires_exactly_once_then_debounces():
    detector = FallDetector()
    events = []
    # Same tracked person, on the ground continuously, well past the
    # sustain threshold, sampled frequently.
    for i in range(60):
        t = i * 0.1  # 0.0s .. 5.9s
        events += detector.update([_fallen_person()], now=t)

    assert len(events) == 1
    assert events[0]["confidence"] > 0
    assert events[0]["sustained_seconds"] >= FALL_MIN_SUSTAINED_SECONDS

    fired_at = FALL_MIN_SUSTAINED_SECONDS  # first update() call where sustained_seconds crossed the threshold

    # Still on the ground, still within the debounce window - must not
    # fire again.
    more_events = detector.update([_fallen_person()], now=fired_at + (FALL_DEBOUNCE_SECONDS - 1))
    assert more_events == []


def test_fall_can_fire_again_after_debounce_and_recovery():
    detector = FallDetector()
    all_events = []
    for i in range(60):
        all_events += detector.update([_fallen_person()], now=i * 0.1)
    assert len(all_events) == 1

    # Person gets up, then falls again after the debounce window: should
    # be detected as a new event.
    t = 6.0
    detector.update([_standing_person()], now=t)
    t += FALL_DEBOUNCE_SECONDS + 1
    for i in range(20):
        all_events += detector.update([_fallen_person()], now=t + i * 0.1)

    assert len(all_events) == 2
