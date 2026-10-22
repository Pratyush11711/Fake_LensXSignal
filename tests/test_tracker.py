import pytest
from src.detection.face_tracker import FaceTrack, SingleSubjectTracker, compute_iou


def test_compute_iou():
    boxA = (0.0, 0.0, 100.0, 100.0)
    boxB = (50.0, 0.0, 100.0, 100.0)
    iou = compute_iou(boxA, boxB)
    assert round(iou, 4) == round(5000.0 / 15000.0, 4)  # 0.3333

    # Disjoint boxes
    boxC = (200.0, 200.0, 50.0, 50.0)
    assert compute_iou(boxA, boxC) == 0.0


def test_tracker_initialization_picks_largest_face():
    tracker = SingleSubjectTracker()
    candidates = [
        (10.0, 10.0, 50.0, 50.0, 0.9),    # area = 2500
        (100.0, 100.0, 120.0, 120.0, 0.95),# area = 14400 (largest)
        (50.0, 50.0, 60.0, 60.0, 0.8),    # area = 3600
    ]
    tracked = tracker.update(candidates, (400, 400))
    assert tracked == (100.0, 100.0, 120.0, 120.0)
    assert tracker.active_track is not None
    assert tracker.active_track.total_detections == 1


def test_tracker_matches_consistent_subject():
    tracker = SingleSubjectTracker(smoothing_alpha=0.5)
    tracker.update([(100.0, 100.0, 100.0, 100.0, 0.9)], (400, 400))

    # Next frame: slight motion to (104, 102)
    next_candidates = [
        (10.0, 10.0, 40.0, 40.0, 0.8),
        (104.0, 102.0, 100.0, 100.0, 0.92),
    ]
    tracked = tracker.update(next_candidates, (400, 400))
    assert tracked is not None
    assert tracker.active_track.consecutive_misses == 0
    assert tracker.active_track.total_detections == 2
    # Smoothed x = 0.5 * 104 + 0.5 * 100 = 102.0
    assert tracker.active_track.x == 102.0


def test_tracker_handles_temporary_occlusion():
    tracker = SingleSubjectTracker(max_misses=2)
    tracker.update([(100.0, 100.0, 100.0, 100.0, 0.9)], (400, 400))

    # Frame 2: Face occluded (empty candidates) -> retains position
    box2 = tracker.update([], (400, 400))
    assert box2 == (100.0, 100.0, 100.0, 100.0)
    assert tracker.active_track.consecutive_misses == 1

    # Frame 3: Face occluded again
    box3 = tracker.update([], (400, 400))
    assert box3 == (100.0, 100.0, 100.0, 100.0)
    assert tracker.active_track.consecutive_misses == 2

    # Frame 4: Max misses exceeded -> track dropped
    box4 = tracker.update([], (400, 400))
    assert box4 is None
    assert tracker.active_track is None
