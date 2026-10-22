"""
evaluation/run_stage5_eval.py

Stage 5: Comparative Evaluation & Calibration Adapter on Second Benchmark Dataset (DFDC).
- Evaluates Baseline ViT and GenD PE_L side-by-side on identical face crops.
- Calibrates temperature scaling T STRICTLY on Calibration subset (N=50: 25 REAL, 25 FAKE).
- Evaluates frozen calibrated model and raw models on untouched Final Test subset (N=50: 25 REAL, 25 FAKE).
- Outputs evaluation/stage5_results.json.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import scipy.optimize
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn import metrics
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.gend_detector import predict_frame_logit, get_gend_model


class YuNetFaceDetector:
    """YuNet ONNX Face Detector wrapper with standard 1.3x crop margin."""
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = str(PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx")
        self.model_path = model_path
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )

    def detect_largest_face(self, bgr_image: np.ndarray, margin: float = 0.3) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        h, w, _ = bgr_image.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr_image)
        if faces is None or len(faces) == 0:
            return None

        best_face = None
        max_area = 0
        for face in faces:
            fx, fy, fw, fh = map(int, face[:4])
            area = fw * fh
            if area > max_area and fw >= 30 and fh >= 30:
                max_area = area
                best_face = (fx, fy, fw, fh)

        if best_face is None:
            return None

        fx, fy, fw, fh = best_face
        margin_x = int(fw * margin)
        margin_y = int(fh * margin)
        x1 = max(0, fx - margin_x)
        y1 = max(0, fy - margin_y)
        x2 = min(w, fx + fw + margin_x)
        y2 = min(h, fy + fh + margin_y)

        crop = bgr_image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        return crop, (x1, y1, x2, y2)


class BaselineViTDetector:
    """Detector A: Existing Baseline ViT (dima806/deepfake_vs_real_image_detection)."""
    def __init__(self, device: str = "cpu"):
        self.model_id = "dima806/deepfake_vs_real_image_detection"
        self.device = torch.device(device)
        print(f"Loading Baseline ViT ({self.model_id}) on {self.device}...", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        id2label = self.model.config.id2label
        self.fake_index = 1
        for idx, label in id2label.items():
            if "fake" in label.lower():
                self.fake_index = idx
                break
        print(f"  Baseline ViT loaded. Fake class index: {self.fake_index} ({id2label.get(self.fake_index)})", flush=True)

    @torch.no_grad()
    def predict_face_crop(self, pil_image: Image.Image) -> float:
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        return float(probs[self.fake_index].item())


def calculate_metrics(y_true: List[int], y_scores: List[float], threshold: float = 0.5) -> Dict[str, Any]:
    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_scores)

    try:
        auc = float(metrics.roc_auc_score(y_true_np, y_scores_np))
    except Exception:
        auc = 0.0

    try:
        ap = float(metrics.average_precision_score(y_true_np, y_scores_np))
    except Exception:
        ap = 0.0

    try:
        fpr, tpr, _ = metrics.roc_curve(y_true_np, y_scores_np, pos_label=1)
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
        eer = float(fpr[eer_idx])
    except Exception:
        eer = 0.0

    y_pred = (y_scores_np >= threshold).astype(int)
    tp = int(np.sum((y_true_np == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true_np == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true_np == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true_np == 1) & (y_pred == 0)))

    acc = float((tp + tn) / len(y_true_np)) if len(y_true_np) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "auc_roc": round(auc, 4),
        "eer": round(eer, 4),
        "average_precision": round(ap, 4),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]]
    }


def fit_temperature_scaling(logits_list: List[float], labels_list: List[int]) -> float:
    """
    Fits single positive temperature parameter T by minimizing NLL / binary cross-entropy.
    sigmoid(logit / T) vs label.
    """
    z = np.array(logits_list, dtype=np.float64)
    y = np.array(labels_list, dtype=np.float64)

    def nll_loss(T_arr):
        T = T_arr[0]
        if T <= 1e-4:
            return 1e9
        scaled_z = z / T
        # Numerical stability for log(1 + exp)
        loss = np.mean(np.log1p(np.exp(-np.abs(scaled_z))) + np.maximum(scaled_z, 0) - y * scaled_z)
        return loss

    res = scipy.optimize.minimize(nll_loss, x0=[1.0], bounds=[(0.05, 20.0)], method="L-BFGS-B")
    optimal_T = float(res.x[0])
    return optimal_T


def run_stage5_evaluation():
    print("=" * 80, flush=True)
    print("STAGE 5: SECOND DATASET EVALUATION & CALIBRATION (DFDC BENCHMARK)", flush=True)
    print("=" * 80, flush=True)

    manifest_path = Path("D:/benchmark_dataset_dfdc/manifest.json")
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found at {manifest_path}", flush=True)
        return

    m = json.load(open(manifest_path, "r", encoding="utf-8"))
    reals = [v for v in m["videos"] if v["ground_truth"] == "REAL"]
    fakes = [v for v in m["videos"] if v["ground_truth"] == "FAKE"]

    # Disjoint splits: 50 Calibration (25 R + 25 F), 50 Final Test (25 R + 25 F)
    cal_videos = reals[:25] + fakes[:25]
    test_videos = reals[25:50] + fakes[25:50]

    for v in cal_videos:
        v["split_assignment"] = "CALIBRATION"
    for v in test_videos:
        v["split_assignment"] = "FINAL_TEST"

    all_videos = cal_videos + test_videos
    print(f"Loaded {len(all_videos)} DFDC videos: {len(cal_videos)} Calibration, {len(test_videos)} Final Test.\n", flush=True)

    # Initialize Detectors
    face_detector = YuNetFaceDetector()
    baseline_detector = BaselineViTDetector(device="cpu")
    print("Warming up GenD PE_L...", flush=True)
    get_gend_model("yermandy/GenD_PE_L", "cpu")
    print("GenD PE_L loaded successfully.\n", flush=True)

    t_start = time.perf_counter()
    eval_records = []

    print("Running side-by-side inference on all 100 DFDC videos...", flush=True)
    print("-" * 80, flush=True)

    for idx, vinfo in enumerate(all_videos, 1):
        vid = vinfo["id"]
        vpath = vinfo["local_path"]
        vname = vinfo["filename"]
        gt_str = vinfo["ground_truth"]
        gt_label = vinfo["ground_truth_label"]
        split_type = vinfo["split_assignment"]

        t0 = time.perf_counter()
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"[{idx:3d}/100] ERROR opening {vname}", flush=True)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 300

        # Uniform 15-frame sampling
        frames_to_sample = 15
        step = max(1, total_frames // frames_to_sample)
        sample_indices = [min(i * step, total_frames - 1) for i in range(frames_to_sample)]

        face_crops_pil = []
        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            det = face_detector.detect_largest_face(frame, margin=0.3)
            if det is not None:
                crop_bgr, _ = det
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                face_crops_pil.append(Image.fromarray(crop_rgb))
        cap.release()

        n_faces = len(face_crops_pil)
        if n_faces < 3:
            status = "INSUFFICIENT_EVIDENCE"
            elapsed = time.perf_counter() - t0
            print(f"[{idx:3d}/100] [{split_type[:3]}] {vname[:20]:<20} | GT: {gt_str:<4} | Faces: {n_faces}/15 | INSUFFICIENT ({elapsed:.2f}s)", flush=True)
            continue

        status = "ANALYZED"

        # Baseline ViT inference
        b_scores = [baseline_detector.predict_face_crop(c) for c in face_crops_pil]
        b_video_score = float(np.mean(b_scores))

        # GenD PE_L inference
        g_logits = [predict_frame_logit(c) for c in face_crops_pil]
        g_raw_probs = [1.0 / (1.0 + math.exp(-lg)) for lg in g_logits]

        # 10% trimmed mean for raw GenD
        if len(g_raw_probs) >= 10:
            sorted_p = sorted(g_raw_probs)
            g_raw_trimmed = float(np.mean(sorted_p[1:-1]))
            # Also raw trimmed logit
            sorted_l = sorted(g_logits)
            g_raw_trimmed_logit = float(np.mean(sorted_l[1:-1]))
        else:
            g_raw_trimmed = float(np.mean(g_raw_probs))
            g_raw_trimmed_logit = float(np.mean(g_logits))

        elapsed = time.perf_counter() - t0

        record = {
            "id": vid,
            "filename": vname,
            "split": split_type,
            "ground_truth": gt_str,
            "ground_truth_label": gt_label,
            "status": status,
            "face_frames": n_faces,
            "baseline_score": round(b_video_score, 4),
            "gend_raw_trimmed_logit": round(g_raw_trimmed_logit, 4),
            "gend_raw_score": round(g_raw_trimmed, 4),
            "gend_raw_mean": round(float(np.mean(g_raw_probs)), 4),
            "gend_raw_std": round(float(np.std(g_raw_probs)), 4),
            "elapsed_seconds": round(elapsed, 3)
        }
        eval_records.append(record)

        b_corr = "[OK]" if (b_video_score >= 0.5) == (gt_label == 1) else "[X]"
        g_corr = "[OK]" if (g_raw_trimmed >= 0.5) == (gt_label == 1) else "[X]"

        if idx % 5 == 0 or idx == len(all_videos):
            print(f"[{idx:3d}/100] [{split_type[:3]}] {vname[:20]:<20} | GT: {gt_str:<4} | Baseline: {b_video_score:.4f} {b_corr:<4} | GenD(Raw): {g_raw_trimmed:.4f} {g_corr:<4} | {elapsed:.2f}s", flush=True)

    total_time = time.perf_counter() - t_start

    # Partition records into Calibration and Final Test
    cal_records = [r for r in eval_records if r["split"] == "CALIBRATION" and r["status"] == "ANALYZED"]
    test_records = [r for r in eval_records if r["split"] == "FINAL_TEST" and r["status"] == "ANALYZED"]

    print("\n" + "=" * 80, flush=True)
    print("STEP 1: FITTING TEMPERATURE SCALING (CALIBRATION SUBSET ONLY)", flush=True)
    print("=" * 80, flush=True)

    cal_logits = [r["gend_raw_trimmed_logit"] for r in cal_records]
    cal_labels = [r["ground_truth_label"] for r in cal_records]

    # Fit T strictly on calibration subset
    frozen_T = fit_temperature_scaling(cal_logits, cal_labels)
    print(f"Fitted Optimal Temperature T = {frozen_T:.4f} (Fit strictly on {len(cal_records)} Calibration samples).", flush=True)

    # Compute calibrated scores for all records using FROZEN T
    for r in eval_records:
        if r["status"] == "ANALYZED":
            cal_prob = 1.0 / (1.0 + math.exp(-r["gend_raw_trimmed_logit"] / frozen_T))
            r["gend_calibrated_score"] = round(cal_prob, 4)

    # Re-extract calibrated scores for test subset
    test_y_true = [r["ground_truth_label"] for r in test_records]
    test_baseline_scores = [r["baseline_score"] for r in test_records]
    test_gend_raw_scores = [r["gend_raw_score"] for r in test_records]
    test_gend_cal_scores = [r["gend_calibrated_score"] for r in test_records]

    # Step 2: Compute Metrics on Untouched Final Test Split
    print("\n" + "=" * 80, flush=True)
    print("STEP 2: FINAL EVALUATION ON UNTOUCHED TEST SUBSET (N = 50)", flush=True)
    print("=" * 80, flush=True)

    test_metrics_baseline = calculate_metrics(test_y_true, test_baseline_scores, threshold=0.50)
    test_metrics_gend_raw = calculate_metrics(test_y_true, test_gend_raw_scores, threshold=0.50)
    test_metrics_gend_cal = calculate_metrics(test_y_true, test_gend_cal_scores, threshold=0.50)

    # Calibration set metrics for comparison
    cal_y_true = [r["ground_truth_label"] for r in cal_records]
    cal_gend_raw_scores = [r["gend_raw_score"] for r in cal_records]
    cal_gend_cal_scores = [r["gend_calibrated_score"] for r in cal_records]
    cal_metrics_gend_raw = calculate_metrics(cal_y_true, cal_gend_raw_scores, threshold=0.50)
    cal_metrics_gend_cal = calculate_metrics(cal_y_true, cal_gend_cal_scores, threshold=0.50)

    # Frame variance analysis on DFDC
    real_stds = [r["gend_raw_std"] for r in test_records if r["ground_truth_label"] == 0]
    fake_stds = [r["gend_raw_std"] for r in test_records if r["ground_truth_label"] == 1]
    real_means = [r["gend_raw_mean"] for r in test_records if r["ground_truth_label"] == 0]
    fake_means = [r["gend_raw_mean"] for r in test_records if r["ground_truth_label"] == 1]

    frame_stats = {
        "real_videos": {
            "count": len(real_stds),
            "avg_probability_mean": round(float(np.mean(real_means)), 4) if real_means else 0.0,
            "avg_inter_frame_std": round(float(np.mean(real_stds)), 4) if real_stds else 0.0,
        },
        "fake_videos": {
            "count": len(fake_stds),
            "avg_probability_mean": round(float(np.mean(fake_means)), 4) if fake_means else 0.0,
            "avg_inter_frame_std": round(float(np.mean(fake_stds)), 4) if fake_stds else 0.0,
        }
    }

    # Compile Final JSON Output
    results_payload = {
        "metadata": {
            "stage": "STAGE 5 — SECOND DATASET BENCHMARK EVALUATION (DFDC)",
            "dataset_name": m.get("benchmark_name"),
            "dataset_source": m.get("dataset_source"),
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_videos": len(all_videos),
            "calibration_videos": len(cal_records),
            "test_videos": len(test_records),
            "frozen_temperature_T": round(frozen_T, 4),
            "total_evaluation_time_seconds": round(total_time, 2)
        },
        "calibration_subset_metrics": {
            "gend_raw": cal_metrics_gend_raw,
            "gend_calibrated": cal_metrics_gend_cal
        },
        "final_test_subset_metrics": {
            "baseline_vit": test_metrics_baseline,
            "gend_pe_l_raw": test_metrics_gend_raw,
            "gend_pe_l_calibrated_frozen": test_metrics_gend_cal
        },
        "frame_level_variance_test_split": frame_stats,
        "video_records": eval_records
    }

    out_json = PROJECT_ROOT / "evaluation" / "stage5_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\nFINAL TEST SPLIT RESULTS SUMMARY (Threshold = 0.50):", flush=True)
    print(f"{'Metric':<25} | {'Baseline ViT':<15} | {'GenD PE_L (Raw)':<18} | {'GenD PE_L (Calibrated)':<22}", flush=True)
    print("-" * 85, flush=True)
    for k in ["auc_roc", "accuracy", "precision", "recall", "f1_score", "eer", "average_precision"]:
        b_val = test_metrics_baseline[k]
        g_val = test_metrics_gend_raw[k]
        gc_val = test_metrics_gend_cal[k]
        print(f"{k:<25} | {b_val:<15.4f} | {g_val:<18.4f} | {gc_val:<22.4f}", flush=True)

    print("\nCONFUSION MATRICES (Final Test Split, N = 50):", flush=True)
    print(f"Baseline ViT: TP={test_metrics_baseline['true_positives']}, TN={test_metrics_baseline['true_negatives']}, FP={test_metrics_baseline['false_positives']}, FN={test_metrics_baseline['false_negatives']}", flush=True)
    print(f"GenD (Raw):   TP={test_metrics_gend_raw['true_positives']}, TN={test_metrics_gend_raw['true_negatives']}, FP={test_metrics_gend_raw['false_positives']}, FN={test_metrics_gend_raw['false_negatives']}", flush=True)
    print(f"GenD (Calib): TP={test_metrics_gend_cal['true_positives']}, TN={test_metrics_gend_cal['true_negatives']}, FP={test_metrics_gend_cal['false_positives']}, FN={test_metrics_gend_cal['false_negatives']}", flush=True)
    print(f"\n[OK] Output written to {out_json}", flush=True)


if __name__ == "__main__":
    run_stage5_evaluation()
