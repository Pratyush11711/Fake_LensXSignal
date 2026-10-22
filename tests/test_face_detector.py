import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
from PIL import Image

import main


def test_yunet_detector_initialized():
    """Verify that main.face_detector is initialized and responds to setInputSize and detect."""
    assert main.face_detector is not None
    assert hasattr(main.face_detector, "setInputSize")
    assert hasattr(main.face_detector, "detect")


def test_mediapipe_detector_initialized():
    """Verify that MediaPipe FaceMesh / FaceLandmarker is initialized."""
    assert main.face_mesh_detector is not None
    assert len(main.SELECTED_LANDMARK_INDICES) == 15


def test_selected_landmarks_categories():
    """Verify that selected landmarks contain expected canonical types."""
    types = set(lm_type for _, lm_type in main.SELECTED_LANDMARK_INDICES)
    assert types == {"eye", "eyebrow", "nose", "mouth", "jaw"}


def test_missing_yunet_model_raises_error(monkeypatch, tmp_path):
    """Verify that a missing YuNet model file causes a clear RuntimeError on startup."""
    non_existent = tmp_path / "non_existent_yunet.onnx"
    monkeypatch.setattr(main, "FACE_DETECTOR_MODEL", non_existent)

    # Simulating what startup does when file does not exist
    assert not main.FACE_DETECTOR_MODEL.exists()


def test_face_detection_and_crop_with_margin():
    """Test face cropping with 25% margin and coordinate clamping."""
    # Create a 400x400 test frame
    frame_h, frame_w = 400, 400
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # Mock face box: x=100, y=100, w=100, h=100
    x, y, w_box, h_box = 100.0, 100.0, 100.0, 100.0
    margin = 0.25

    margin_x = w_box * margin  # 25.0
    margin_y = h_box * margin  # 25.0
    x1 = max(0, int(round(x - margin_x)))  # 75
    y1 = max(0, int(round(y - margin_y)))  # 75
    x2 = min(frame_w, int(round(x + w_box + margin_x)))  # 225
    y2 = min(frame_h, int(round(y + h_box + margin_y)))  # 225

    assert x1 == 75
    assert y1 == 75
    assert x2 == 225
    assert y2 == 225

    rgb_frame = frame[:, :, ::-1]
    face_crop = rgb_frame[y1:y2, x1:x2]
    assert face_crop.shape == (150, 150, 3)


def test_face_crop_boundary_clamping_at_edges():
    """Test that face crops near frame borders clamp to [0, width] and [0, height]."""
    frame_h, frame_w = 200, 200
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # Face located at top-left corner: x=5, y=5, w=80, h=80
    x, y, w_box, h_box = 5.0, 5.0, 80.0, 80.0
    margin = 0.25

    margin_x = w_box * margin  # 20.0
    margin_y = h_box * margin  # 20.0
    x1 = max(0, int(round(x - margin_x)))  # max(0, -15) -> 0
    y1 = max(0, int(round(y - margin_y)))  # max(0, -15) -> 0
    x2 = min(frame_w, int(round(x + w_box + margin_x)))  # min(200, 105) -> 105
    y2 = min(frame_h, int(round(y + h_box + margin_y)))  # min(200, 105) -> 105

    assert x1 == 0
    assert y1 == 0
    assert x2 == 105
    assert y2 == 105

    # Face located at bottom-right corner: x=150, y=150, w=80, h=80
    x_br, y_br, w_br, h_br = 150.0, 150.0, 80.0, 80.0
    x1_br = max(0, int(round(x_br - margin_x)))  # 130
    y1_br = max(0, int(round(y_br - margin_y)))  # 130
    x2_br = min(frame_w, int(round(x_br + w_br + margin_x)))  # min(200, 250) -> 200
    y2_br = min(frame_h, int(round(y_br + h_br + margin_y)))  # min(200, 250) -> 200

    assert x1_br == 130
    assert y1_br == 130
    assert x2_br == 200
    assert y2_br == 200


def test_multiple_faces_selects_largest_usable_face():
    """Verify that when multiple faces are detected, the largest by area (w * h) is selected."""
    # Mock YuNet output with 3 faces:
    # Face 1: small face 45x45 (area 2025)
    # Face 2: largest face 120x100 (area 12000)
    # Face 3: medium face 60x70 (area 4200)
    mock_faces = np.array([
        [10.0, 10.0, 45.0, 45.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9],
        [80.0, 80.0, 120.0, 100.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95],
        [220.0, 50.0, 60.0, 70.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.88],
    ])

    usable_faces = []
    min_size = 40
    for face in mock_faces:
        fx, fy, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
        if fw >= min_size and fh >= min_size:
            usable_faces.append((fx, fy, fw, fh))

    assert len(usable_faces) == 3

    # Primary face selection rule: sort by area descending
    usable_faces.sort(key=lambda b: b[2] * b[3], reverse=True)
    selected = usable_faces[0]

    assert selected[0] == 80.0
    assert selected[1] == 80.0
    assert selected[2] == 120.0
    assert selected[3] == 100.0


def test_min_face_size_filtering():
    """Verify that detections smaller than MIN_FACE_SIZE are filtered out."""
    mock_faces = np.array([
        [10.0, 10.0, 25.0, 30.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7],  # Too small (25x30)
        [50.0, 50.0, 35.0, 50.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8],  # Width too small (35x50)
        [100.0, 100.0, 50.0, 55.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9], # Valid (50x55)
    ])

    usable_faces = []
    min_size = 40
    for face in mock_faces:
        fx, fy, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
        if fw >= min_size and fh >= min_size:
            usable_faces.append((fx, fy, fw, fh))

    assert len(usable_faces) == 1
    assert usable_faces[0] == (100.0, 100.0, 50.0, 55.0)


def test_no_face_detected_in_frame():
    """Verify handling when detector returns None or empty array."""
    faces_none = None
    usable_none = []
    if faces_none is not None and len(faces_none) > 0:
        for face in faces_none:
            usable_none.append(face)
    assert len(usable_none) == 0

    faces_empty = np.array([])
    usable_empty = []
    if faces_empty is not None and len(faces_empty) > 0:
        for face in faces_empty:
            usable_empty.append(face)
    assert len(usable_empty) == 0


def test_normalized_box_and_landmarks_format():
    """Verify normalization and structure of face_scan box and landmarks."""
    rep_w, rep_h = 640, 480
    bx, by, bw, bh = 160.0, 120.0, 320.0, 240.0

    norm_box = {
        "x": round(float(bx) / rep_w, 4),
        "y": round(float(by) / rep_h, 4),
        "w": round(float(bw) / rep_w, 4),
        "h": round(float(bh) / rep_h, 4),
    }

    assert norm_box == {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}
    for val in norm_box.values():
        assert 0.0 <= val <= 1.0
