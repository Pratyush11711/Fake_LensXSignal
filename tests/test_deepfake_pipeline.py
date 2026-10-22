import pytest
from src.pipeline.calibrator import ThresholdCalibrator


def test_threshold_calibrator_optimal_f1():
    calibrator = ThresholdCalibrator()
    # Perfect separation with threshold at 0.6
    val_labels = [0, 0, 0, 0, 1, 1, 1, 1]
    val_scores = [0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.85, 0.9]

    tau = calibrator.calibrate(val_labels, val_scores)
    assert 0.4 < tau <= 0.7
    assert calibrator.calibrated

    verdict_real, _ = calibrator.predict(0.2)
    verdict_fake, _ = calibrator.predict(0.8)
    assert verdict_real == "REAL"
    assert verdict_fake == "FAKE"


def test_threshold_calibrator_single_class_fallback():
    calibrator = ThresholdCalibrator()
    val_labels = [0, 0, 0]
    val_scores = [0.1, 0.2, 0.3]
    tau = calibrator.calibrate(val_labels, val_scores)
    assert tau == 0.5
