from typing import Dict, List, Tuple

import numpy as np


class ThresholdCalibrator:
    """
    Computes and stores calibrated decision threshold based on validation set ROC / F1 optimization.
    Never tunes threshold on the test set.
    """

    def __init__(self, default_threshold: float = 0.5):
        self.threshold = default_threshold
        self.calibrated = False

    def calibrate(self, val_labels: List[int], val_scores: List[float]) -> float:
        """
        Finds optimal decision threshold maximizing F1 score on validation set.
        """
        y_true = np.array(val_labels, dtype=int)
        y_scores = np.array(val_scores, dtype=float)

        if len(np.unique(y_true)) < 2:
            self.threshold = 0.5
            return self.threshold

        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1 = -1.0
        best_tau = 0.5

        for tau in thresholds:
            y_pred = (y_scores >= tau).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)

        self.threshold = round(best_tau, 4)
        self.calibrated = True
        return self.threshold

    def predict(self, score: float) -> Tuple[str, float]:
        """
        Returns (verdict, distance_from_threshold / confidence).
        """
        verdict = "FAKE" if score >= self.threshold else "REAL"
        confidence = score if verdict == "FAKE" else (1.0 - score)
        return verdict, confidence
