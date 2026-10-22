import numpy as np
import pytest

from src.alignment.face_aligner import FaceAligner


def test_aligner_canonical_points_scaling():
    aligner_256 = FaceAligner(target_size=256)
    assert aligner_256.canonical_points.shape == (5, 2)
    assert np.allclose(aligner_256.canonical_points[0], [76.8, 97.28])

    aligner_299 = FaceAligner(target_size=299)
    scale = 299 / 256
    assert np.allclose(aligner_299.canonical_points[0], [76.8 * scale, 97.28 * scale])


def test_aligner_crop_fallback_with_face_box():
    aligner = FaceAligner(target_size=256, crop_margin=0.25)
    # Dummy black frame 400x400
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    face_box = (100.0, 100.0, 100.0, 100.0)

    aligned = aligner.align_face(frame, face_box=face_box)
    assert aligned is not None
    assert aligned.shape == (256, 256, 3)
