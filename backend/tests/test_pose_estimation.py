"""
Pose estimation: the keypoint format, parsing the model's output, posture
geometry, model-loading failures - and one real-inference smoke test.

Everything except the smoke test is deterministic and needs no model. The
smoke test runs the actual `yolov8n-pose.pt` on a real photograph shipped
with Ultralytics; it is skipped (and says so) when the weights are not
present locally, so it never downloads anything in CI.
"""
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.services.detection import engine as engine_mod
from app.services.detection.engine import DetectionEngine, PersonDetection, parse_pose_results
from app.services.detection.pose import (
    COCO_KEYPOINT_NAMES,
    COCO_SKELETON,
    normalize_keypoints,
    pose_format,
    torso_angle_degrees,
    visible_edges,
)


# --- format -----------------------------------------------------------------

def test_skeleton_is_ultralytics_own_topology():
    """Copied, not invented - and kept honest against the installed library."""
    from ultralytics.utils.plotting import Annotator

    annotator = Annotator(np.zeros((8, 8, 3), dtype=np.uint8))
    expected = tuple((a - 1, b - 1) for a, b in annotator.skeleton)
    assert COCO_SKELETON == expected


def test_seventeen_coco_keypoints_and_every_edge_in_range():
    assert len(COCO_KEYPOINT_NAMES) == 17
    assert all(0 <= a < 17 and 0 <= b < 17 for a, b in COCO_SKELETON)
    assert pose_format()["skeleton"] == [list(edge) for edge in COCO_SKELETON]


def _pose(confidence=0.9, **overrides):
    """A standing person: shoulders at y=100, hips at y=200."""
    points = [[100.0, 50.0 + i, confidence] for i in range(17)]
    points[5] = [90.0, 100.0, confidence]   # left shoulder
    points[6] = [110.0, 100.0, confidence]  # right shoulder
    points[11] = [92.0, 200.0, confidence]  # left hip
    points[12] = [108.0, 200.0, confidence]  # right hip
    for index, point in overrides.items():
        points[int(index)] = point
    return [tuple(p) for p in points]


def test_malformed_keypoints_are_no_pose_rather_than_an_error():
    assert normalize_keypoints(None) == []
    assert normalize_keypoints([]) == []
    assert normalize_keypoints([(1, 2, 3)] * 5) == []          # wrong count
    assert normalize_keypoints([("a", "b", "c")] * 17) == []    # not numbers
    assert normalize_keypoints([(1,)] * 17) == []               # no y


def test_non_finite_keypoints_are_treated_as_invisible():
    points = _pose()
    points[0] = (float("nan"), 3.0, 0.9)
    parsed = normalize_keypoints(points)
    assert parsed[0][2] == 0.0


def test_low_confidence_keypoints_drop_their_edges_only():
    points = _pose()
    points[9] = (0.0, 0.0, 0.05)  # left wrist barely seen
    edges = visible_edges(points, 0.3)
    assert (7, 9) not in edges          # elbow-wrist dropped
    assert (5, 7) in edges              # shoulder-elbow kept
    assert len(edges) == len(COCO_SKELETON) - 1


def test_torso_angle_upright_and_horizontal():
    assert torso_angle_degrees(_pose()) == pytest.approx(0.0, abs=1.0)
    lying = _pose(**{"5": (200.0, 100.0, 0.9), "6": (200.0, 110.0, 0.9), "11": (100.0, 100.0, 0.9), "12": (100.0, 110.0, 0.9)})
    assert torso_angle_degrees(lying) == pytest.approx(90.0, abs=1.0)


def test_torso_angle_uses_one_side_when_the_other_is_occluded():
    points = _pose(**{"6": (0.0, 0.0, 0.0), "12": (0.0, 0.0, 0.0)})
    assert torso_angle_degrees(points) == pytest.approx(0.0, abs=2.0)


def test_no_shoulders_or_no_hips_is_no_evidence():
    no_shoulders = _pose(**{"5": (0.0, 0.0, 0.1), "6": (0.0, 0.0, 0.1)})
    assert torso_angle_degrees(no_shoulders) is None
    assert torso_angle_degrees([]) is None


# --- parsing the model's output ---------------------------------------------

class _T:
    """Stands in for a torch tensor: just .cpu().numpy()."""

    def __init__(self, array):
        self._array = np.asarray(array, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __len__(self):
        return len(self._array)


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _T(xyxy)
        self.conf = _T(conf)

    def __len__(self):
        return len(self.xyxy)


class _Keypoints:
    def __init__(self, data):
        self.data = _T(data)


class _Result:
    def __init__(self, boxes=None, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


def test_multiple_people_keep_box_confidence_and_their_own_keypoints():
    kps_a = np.full((17, 3), 1.0)
    kps_b = np.full((17, 3), 2.0)
    result = _Result(
        boxes=_Boxes([[0, 0, 10, 20], [50, 0, 60, 20]], [0.91, 0.47]),
        keypoints=_Keypoints([kps_a, kps_b]),
    )
    people = parse_pose_results([result])
    assert [p.bbox for p in people] == [(0.0, 0.0, 10.0, 20.0), (50.0, 0.0, 60.0, 20.0)]
    assert [round(p.confidence, 2) for p in people] == [0.91, 0.47]
    assert people[1].keypoints[0] == (2.0, 2.0, 2.0)


@pytest.mark.parametrize("results", [
    [],
    None,
    [_Result(boxes=None, keypoints=None)],
    [_Result(boxes=_Boxes(np.zeros((0, 4)), []), keypoints=_Keypoints(np.zeros((0, 17, 3))))],
])
def test_frames_with_nobody_are_empty_not_errors(results):
    assert parse_pose_results(results) == []


def test_person_detection_stays_constructible_without_confidence():
    """Existing callers build PersonDetection(bbox, keypoints)."""
    person = PersonDetection(bbox=(0, 0, 1, 1), keypoints=[])
    assert person.confidence is None


# --- model availability -----------------------------------------------------

def test_a_pose_model_that_fails_to_load_is_reported_not_raised_and_not_retried(monkeypatch):
    calls = []

    def failing_yolo(path):
        calls.append(path)
        raise RuntimeError("weights corrupt")

    monkeypatch.setattr(engine_mod, "YOLO", failing_yolo)
    engine = DetectionEngine()
    engine.ensure_models()
    engine.ensure_models()

    assert engine.pose_available is False
    assert "weights corrupt" in engine.pose_model_error
    assert len(calls) == 2  # pose + object, once each - not on every call
    assert engine.extract_people(np.zeros((4, 4, 3), dtype=np.uint8)) == []
    assert engine.extract_objects(np.zeros((4, 4, 3), dtype=np.uint8)) == ([], [])


def test_pose_runs_when_enabled_or_when_the_fall_mode_needs_it(monkeypatch):
    engine = DetectionEngine()

    class Pipeline:
        def __init__(self, needs_pose):
            self.needs_pose = needs_pose

    monkeypatch.setattr(settings, "POSE_ESTIMATION_ENABLED", False)
    assert engine.should_run_pose(Pipeline(False)) is False
    assert engine.should_run_pose(Pipeline(True)) is True     # heuristic mode
    monkeypatch.setattr(settings, "POSE_ESTIMATION_ENABLED", True)
    assert engine.should_run_pose(Pipeline(False)) is True


# --- real inference ---------------------------------------------------------

POSE_WEIGHTS = Path(settings.YOLO_POSE_MODEL)


@pytest.mark.skipif(not POSE_WEIGHTS.exists(), reason="yolov8n-pose.pt not present locally; not downloading in tests")
def test_real_pose_inference_on_a_real_photo():
    """Not a mock: the shipped pose model on Ultralytics' own sample photo
    (a street scene with several people). Asserts only what any working
    pose model must produce, not exact coordinates."""
    import cv2
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    assert image is not None

    engine = DetectionEngine()
    engine.ensure_models()
    people = engine.extract_people(image)

    assert len(people) >= 3
    height, width = image.shape[:2]
    for person in people:
        x1, y1, x2, y2 = person.bbox
        assert 0 <= x1 < x2 <= width + 1 and 0 <= y1 < y2 <= height + 1
        assert 0.0 < person.confidence <= 1.0
        assert len(normalize_keypoints(person.keypoints)) == 17
    # Standing pedestrians: at least one clearly upright torso.
    angles = [torso_angle_degrees(normalize_keypoints(p.keypoints)) for p in people]
    assert any(a is not None and a < 30 for a in angles)


@pytest.mark.skipif(not POSE_WEIGHTS.exists(), reason="yolov8n-pose.pt not present locally; not downloading in tests")
def test_concurrent_first_inference_on_the_shared_pose_model_is_safe():
    """Regression, measured: two threads making their FIRST predict() on one
    freshly loaded Ultralytics model at the same moment crashed one of them
    with "AttributeError: 'Conv' object has no attribute 'bn'" (both raced
    through layer fusion) in 15 of 15 trials without the inference lock,
    and 0 of 15 with it. Two uploads - or a camera and an upload - doing
    exactly that after a restart is ordinary."""
    import threading

    import cv2
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    engine = DetectionEngine()
    engine.ensure_models()
    barrier = threading.Barrier(2)
    outcomes = {}

    def infer(name):
        barrier.wait()
        try:
            outcomes[name] = len(engine.extract_people(image))
        except Exception as exc:  # pragma: no cover - the regression itself
            outcomes[name] = exc

    threads = [threading.Thread(target=infer, args=(n,)) for n in ("a", "b")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert outcomes["a"] == outcomes["b"] and outcomes["a"] >= 3, outcomes


def test_concurrent_first_inference_on_the_shared_fall_detector_is_safe():
    """Same race, the trained fall detector's own shared model."""
    import threading

    import cv2
    from ultralytics.utils import ASSETS

    from app.services.detection.fall_object_detector import FallObjectDetector

    detector = FallObjectDetector()
    if not detector.is_available:
        pytest.skip(f"trained fall detector not loadable here: {detector.load_error}")
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    barrier = threading.Barrier(2)
    outcomes = {}

    def infer(name):
        barrier.wait()
        # detect() swallows inference errors and returns [] - so compare
        # both threads' answers rather than looking for an exception.
        outcomes[name] = [(round(b.confidence, 3), tuple(round(v) for v in b.bbox)) for b in detector.detect(image)]

    threads = [threading.Thread(target=infer, args=(n,)) for n in ("a", "b")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert outcomes["a"] == outcomes["b"], outcomes


def test_keypoints_the_model_could_not_place_are_not_visible():
    """Regression, seen on real footage: a person cut off by the frame edge
    had a keypoint reported at (0, 0) with confidence above the floor, which
    drew a limb from them to the frame's corner. Ultralytics' own plotter
    treats a coordinate on the frame edge (or outside it) as missing."""
    from app.services.detection.engine import mark_missing_keypoint

    shape = (720, 540)  # (height, width), as Results.orig_shape
    assert mark_missing_keypoint((0.0, 0.0, 0.62), shape)[2] == 0.0
    assert mark_missing_keypoint((120.0, 0.0, 0.9), shape)[2] == 0.0
    assert mark_missing_keypoint((540.0, 300.0, 0.9), shape)[2] == 0.0   # on the right edge
    assert mark_missing_keypoint((-3.0, 300.0, 0.9), shape)[2] == 0.0
    assert mark_missing_keypoint((120.0, 300.0, 0.9), shape) == (120.0, 300.0, 0.9)

    result = _Result(
        boxes=_Boxes([[400, 10, 540, 700]], [0.9]),
        keypoints=_Keypoints([[[0.0, 0.0, 0.62]] + [[450.0, 50.0 + i, 0.9] for i in range(16)]]),
    )
    result.orig_shape = shape
    person = parse_pose_results([result])[0]
    assert person.keypoints[0][2] == 0.0
    assert visible_edges(normalize_keypoints(person.keypoints), 0.3) == [
        e for e in COCO_SKELETON if 0 not in e
    ]
    # And for records parsed some other way, the coordinate rule still holds.
    assert normalize_keypoints([(0.0, 0.0, 0.9)] + [(5.0, 5.0, 0.9)] * 16)[0][2] == 0.0


def test_a_live_camera_runs_pose_even_when_pose_estimation_is_disabled(monkeypatch):
    """POSE_ESTIMATION_ENABLED=false trims pose from uploaded-video analysis
    only. A live camera's violence detector works from keypoints alone, so
    skipping pose there would silently switch violence detection off."""
    monkeypatch.setattr(settings, "POSE_ESTIMATION_ENABLED", False)
    engine = DetectionEngine()
    sentinel = [object()]
    monkeypatch.setattr(engine, "extract_people", lambda frame: sentinel)
    monkeypatch.setattr(engine, "extract_objects", lambda frame: ([], []))
    monkeypatch.setattr("app.services.detection.engine.recording_engine.trigger_event", lambda *a, **k: None)

    seen = {}

    class Pipeline:
        scene = None
        needs_pose = False

        def update(self, frame, people, now=None, frame_index=None):
            return []

    class Violence:
        def update(self, people):
            seen["people"] = people
            return []

    class Crowd:
        def update(self, person_count, threshold):
            return None

    engine._process_frame(1, np.zeros((48, 64, 3), dtype="uint8"), Pipeline(), Violence(), Crowd(), 20)
    assert seen["people"] is sentinel
