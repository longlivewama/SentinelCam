"""
Deciding WHICH tracked person a fall belongs to.

The rule these tests exist to hold: the annotation names a person only
when the geometry actually says so. Picking "the first person", "the
nearest person", or "the only person" as a fallback would put a
`FALL DETECTED` box over someone who did not fall, which is a confident
false statement about a real person - so every ambiguous or weak case
here must come back as None.
"""
import pytest

from app.services.detection.track_fall_association import (
    AMBIGUITY_MARGIN,
    MIN_ASSOCIATION_SCORE,
    SupportLedger,
    associate,
    best_overlap,
    overlap_score,
)


def person(x, y, w=200, h=400):
    return (float(x), float(y), float(x + w), float(y + h))


# --- scoring --------------------------------------------------------------

def test_a_fall_box_inside_a_person_box_scores_highly():
    """Mid-fall the detector frames the part of the body going down,
    inside the pose model's full-body box."""
    subject = person(100, 100)
    lower_body = (100.0, 300.0, 300.0, 500.0)
    assert overlap_score(lower_body, subject) > 0.7


def test_a_person_box_inside_a_fall_box_also_scores_highly():
    """Once someone is on the ground the boxes invert - the fall box is
    the wider one."""
    on_ground_region = (0.0, 400.0, 460.0, 710.0)
    subject = (50.0, 470.0, 440.0, 690.0)
    assert overlap_score(on_ground_region, subject) > 0.7


def test_a_fall_box_merely_next_to_a_person_scores_below_the_bar():
    """A detector false positive on an object beside someone must not
    reach the credibility floor - this is the real-video case where the
    model fires on a dark car door with a bystander alongside."""
    bystander = (345.0, 337.0, 442.0, 479.0)
    car_door = (251.0, 350.0, 371.0, 507.0)
    assert overlap_score(car_door, bystander) < MIN_ASSOCIATION_SCORE


def test_boxes_that_do_not_touch_score_zero():
    assert overlap_score(person(0, 0), person(800, 800)) == 0.0


# --- naming the faller ----------------------------------------------------

def test_one_person_one_fall_names_that_person():
    subject = person(100, 100)
    match = associate((120.0, 300.0, 280.0, 480.0), [(7, subject)])

    assert match is not None
    assert match.track_id == 7


def test_three_people_one_falls_names_only_the_one_who_fell():
    standing = person(20, 100)
    walking = person(260, 100)
    falling = person(520, 100)
    fall_box = (540.0, 300.0, 700.0, 490.0)   # inside `falling`

    match = associate(fall_box, [(1, standing), (2, walking), (3, falling)])

    assert match is not None
    assert match.track_id == 3
    assert match.runner_up_score < match.score


def test_a_lone_person_far_from_the_fall_box_is_not_named():
    """"Only one candidate" is not evidence. Without this the annotation
    would label whoever happened to be in frame."""
    assert associate((0.0, 0.0, 80.0, 80.0), [(1, person(600, 600))]) is None


def test_two_equally_overlapping_people_are_too_ambiguous_to_name():
    """A coin flip presented as a fact is the worst outcome, so it is a
    refusal instead."""
    left = (100.0, 100.0, 300.0, 500.0)
    right = (300.0, 100.0, 500.0, 500.0)
    straddling = (250.0, 100.0, 350.0, 500.0)

    assert associate(straddling, [(1, left), (2, right)]) is None


def test_a_clear_winner_beats_a_close_but_weaker_candidate():
    winner = (100.0, 100.0, 300.0, 500.0)
    neighbour = (290.0, 100.0, 490.0, 500.0)
    fall_box = (110.0, 300.0, 300.0, 500.0)

    match = associate(fall_box, [(1, winner), (2, neighbour)])
    assert match is not None
    assert match.track_id == 1
    assert match.score - match.runner_up_score >= AMBIGUITY_MARGIN


def test_no_candidates_at_all_yields_no_association():
    assert associate((10.0, 10.0, 50.0, 50.0), []) is None


def test_ties_resolve_to_the_lowest_id_rather_than_dict_order():
    """Determinism matters: the same video must annotate identically on
    every run. (Identical candidates are still refused as ambiguous - this
    only pins down that the ordering itself is stable.)"""
    same = person(100, 100)
    forward = associate((120.0, 300.0, 280.0, 480.0), [(4, same), (9, same)], ambiguity_margin=0.0)
    reverse = associate((120.0, 300.0, 280.0, 480.0), [(9, same), (4, same)], ambiguity_margin=0.0)

    assert forward is not None and reverse is not None
    assert forward.track_id == reverse.track_id == 4


# --- continuity -----------------------------------------------------------

def test_sustained_overlap_outweighs_a_one_frame_coincidence():
    """A passer-by who happens to clip the fall box on the firing frame
    should lose to the person who has been under it the whole time. On
    the firing frame alone the two are only 0.08 apart - support is what
    makes the answer clear."""
    faller = (100.0, 100.0, 300.0, 500.0)
    passerby = (280.0, 100.0, 480.0, 500.0)
    fall_box = (170.0, 250.0, 380.0, 470.0)

    ledger = SupportLedger()
    for step in range(5):
        ledger.record_frame(step * 0.2, {1: 0.85})       # the faller, frame after frame
    ledger.record_frame(1.0, {1: 0.85, 2: 0.5})          # the passer-by arrives

    match = associate(fall_box, [(1, faller), (2, passerby)], support_ledger=ledger, video_time_seconds=1.0)

    assert match is not None
    assert match.track_id == 1
    assert match.support > 0.5
    assert match.score - match.runner_up_score > AMBIGUITY_MARGIN


def test_a_track_seen_on_one_frame_out_of_many_has_weak_support():
    """Support measures persistence, not just strength: appearing once at
    a high score must not look the same as being there throughout."""
    ledger = SupportLedger()
    for step in range(9):
        ledger.record_frame(step * 0.1, {1: 0.8})
    ledger.record_frame(0.9, {1: 0.8, 2: 0.8})

    assert ledger.support(1, 0.9) == pytest.approx(0.8, abs=0.01)
    assert ledger.support(2, 0.9) == pytest.approx(0.08, abs=0.01)


def test_support_older_than_the_window_stops_counting():
    ledger = SupportLedger(window_seconds=1.0)
    ledger.record_frame(0.0, {1: 0.9})
    assert ledger.support(1, 0.5) == pytest.approx(0.9)
    assert ledger.support(1, 5.0) == 0.0


def test_support_for_an_unknown_track_is_zero_not_an_error():
    ledger = SupportLedger()
    ledger.record_frame(0.0, {1: 0.5})
    assert ledger.support(42, 0.1) == 0.0
    assert SupportLedger().support(42, 1.0) == 0.0


def test_the_ledger_only_ever_holds_its_own_window():
    """Otherwise it grows for the length of a long video."""
    ledger = SupportLedger(window_seconds=1.0)
    for step in range(500):
        ledger.record_frame(step * 0.1, {1: 0.5})

    assert len(ledger._frames) <= 12


def test_best_overlap_reports_the_strongest_candidate_without_gating():
    """Used to record per-frame votes; gating is `associate`'s job."""
    a = person(100, 100)
    b = person(600, 100)
    track_id, score = best_overlap((150.0, 300.0, 280.0, 480.0), [(1, a), (2, b)])

    assert track_id == 1
    assert score > 0


def test_best_overlap_reports_nothing_when_nothing_overlaps():
    assert best_overlap((0.0, 0.0, 10.0, 10.0), [(1, person(600, 600))]) == (None, 0.0)
