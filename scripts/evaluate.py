"""
Comprehensive Video-Level Evaluation and Baseline Benchmark Comparison Script.
Evaluates both Baseline ViT and Upgraded Temporal Xception models on identical held-out test partitions.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from tabulate import tabulate
from tqdm import tqdm

from src.baseline.baseline_vit import BaselineViTDetector
from src.pipeline.inference_engine import TemporalInferenceEngine


def evaluate_model_on_test_set(
    test_meta_path: Path,
    temporal_weights: str = "models/best_temporal_model.pth",
    baseline_vit: bool = True,
    temporal_model: bool = True,
):
    if not test_meta_path.exists():
        print("\n" + "=" * 65)
        print("[!] TEST METADATA NOT DETECTED — EVALUATION HALTED (Integrity Rule)")
        print("=" * 65)
        print(f"Test metadata expected at: {test_meta_path.resolve()}")
        print("Please prepare the dataset using: python scripts/prepare_dataset.py")
        print("=" * 65)
        return

    with open(test_meta_path, "r", encoding="utf-8") as f:
        test_records = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"[*] Loaded {len(test_records)} test videos from {test_meta_path}")

    y_true = []
    baseline_preds = []
    baseline_probs = []
    temporal_preds = []
    temporal_probs = []

    # Initialize engines
    base_engine = BaselineViTDetector() if baseline_vit else None
    temp_engine = TemporalInferenceEngine(weights_path=temporal_weights) if temporal_model else None

    # Run evaluation
    for rec in tqdm(test_records, desc="Evaluating models"):
        v_path = rec.get("video_path")
        label = int(rec["label"])
        y_true.append(label)

        if base_engine and v_path:
            res, conf, scores, _ = base_engine.predict_video(v_path)
            score = float(np.mean(scores)) if scores else 0.5
            baseline_probs.append(score)
            baseline_preds.append(1 if res == "FAKE" else 0)

        if temp_engine and v_path:
            res_dict = temp_engine.process_video(v_path)
            score = res_dict.get("detection_score", 0.5)
            temporal_probs.append(score)
            temporal_preds.append(1 if res_dict.get("result") == "FAKE" else 0)

    # Compute metric tables
    results = []
    y_true = np.array(y_true)

    if baseline_preds:
        b_pred = np.array(baseline_preds)
        b_prob = np.array(baseline_probs)
        b_auc = roc_auc_score(y_true, b_prob) if len(np.unique(y_true)) > 1 else 0.5
        results.append([
            "Baseline (2D ViT Mean)",
            f"{accuracy_score(y_true, b_pred):.2%}",
            f"{precision_score(y_true, b_pred, zero_division=0):.4f}",
            f"{recall_score(y_true, b_pred, zero_division=0):.4f}",
            f"{f1_score(y_true, b_pred, zero_division=0):.4f}",
            f"{b_auc:.4f}",
        ])

    if temporal_preds:
        t_pred = np.array(temporal_preds)
        t_prob = np.array(temporal_probs)
        t_auc = roc_auc_score(y_true, t_prob) if len(np.unique(y_true)) > 1 else 0.5
        results.append([
            "Upgraded (Xception + Attention)",
            f"{accuracy_score(y_true, t_pred):.2%}",
            f"{precision_score(y_true, t_pred, zero_division=0):.4f}",
            f"{recall_score(y_true, t_pred, zero_division=0):.4f}",
            f"{f1_score(y_true, t_pred, zero_division=0):.4f}",
            f"{t_auc:.4f}",
        ])

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    print("\n" + tabulate(results, headers=headers, tablefmt="github") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline and Upgraded Models on Test Set")
    parser.add_argument("--test-meta", type=Path, default=Path("data/metadata/test_metadata.jsonl"))
    parser.add_argument("--weights", type=str, default="models/best_temporal_model.pth")
    args = parser.parse_args()

    evaluate_model_on_test_set(args.test_meta, temporal_weights=args.weights)
