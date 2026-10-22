"""
evaluation/run_benchmark_eval.py

Stage 4B: Isolated Non-Production Benchmark Evaluation Adapter.
Evaluates both Detector A (Baseline ViT) and Detector B (GenD) side-by-side
on identical sampled frames from the 100-video balanced benchmark dataset.
Outputs evaluation/benchmark_results.json with rigorous statistical metrics.
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
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn import metrics
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Ensure project root is in sys.path
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

        # Find largest face by area
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
        # Apply margin (1.3x box)
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
    """Detector A: Existing Production Baseline ViT (dima806/deepfake_vs_real_image_detection)."""
    def __init__(self, device: str = "cpu"):
        self.model_id = "dima806/deepfake_vs_real_image_detection"
        self.device = torch.device(device)
        print(f"Loading Baseline ViT ({self.model_id}) on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

        # Identify fake class index
        id2label = self.model.config.id2label
        self.fake_index = 1
        for idx, label in id2label.items():
            if "fake" in label.lower():
                self.fake_index = idx
                break
        print(f"  Baseline ViT loaded. Fake class index: {self.fake_index} ({id2label.get(self.fake_index)})")

    @torch.no_grad()
    def predict_face_crop(self, pil_image: Image.Image) -> float:
        """Returns softmax fake probability for a single face crop."""
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        return float(probs[self.fake_index].item())


def calculate_evaluation_metrics(y_true: List[int], y_scores: List[float], threshold: float = 0.5) -> Dict[str, Any]:
    """Computes comprehensive classification metrics given ground truth binary labels and continuous scores."""
    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_scores)

    # ROC-AUC
    try:
        auc = float(metrics.roc_auc_score(y_true_np, y_scores_np))
    except Exception:
        auc = 0.0

    # Average Precision (PR-AUC)
    try:
        ap = float(metrics.average_precision_score(y_true_np, y_scores_np))
    except Exception:
        ap = 0.0

    # ROC Curve & EER
    try:
        fpr, tpr, thresholds = metrics.roc_curve(y_true_np, y_scores_np, pos_label=1)
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
        eer = float(fpr[eer_idx])
    except Exception:
        eer = 0.0

    # Binary predictions at chosen threshold
    y_pred = (y_scores_np >= threshold).astype(int)

    # Confusion matrix components
    # 0 = REAL (Negative), 1 = FAKE (Positive)
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


def run_benchmark_evaluation():
    print("=" * 80)
    print("STAGE 4B — CONTROLLED BENCHMARK EVALUATION (BASELINE vs GenD)")
    print("=" * 80)

    benchmark_manifest_path = Path("D:/benchmark_dataset/manifest.json")
    if not benchmark_manifest_path.exists():
        print(f"[ERROR] Benchmark manifest not found at {benchmark_manifest_path}.")
        print("Please run scripts/download_benchmark_dataset.py first.")
        return

    with open(benchmark_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    videos = manifest.get("videos", [])
    print(f"Loaded {len(videos)} benchmark videos from manifest ({manifest.get('dataset_source')}).")
    print(f"Real: {manifest.get('total_real')}, Fake: {manifest.get('total_fake')}\n")

    # 1. Initialize Detectors
    face_detector = YuNetFaceDetector()
    baseline_detector = BaselineViTDetector(device="cpu")
    # Warm up GenD
    print("Warming up GenD (yermandy/GenD_PE_L)...")
    get_gend_model("yermandy/GenD_PE_L", "cpu")
    print("GenD loaded successfully.\n")

    results_baseline = []
    results_gend = []

    total_start_time = time.perf_counter()
    total_frames_inferred = 0
    total_crops_detected = 0

    print("Starting side-by-side benchmark evaluation across 100 videos...")
    print("-" * 80)

    for idx, vinfo in enumerate(videos, 1):
        vid = vinfo["id"]
        vpath = vinfo["local_path"]
        vname = vinfo["filename"]
        gt_str = vinfo["ground_truth"]
        gt_label = vinfo["ground_truth_label"]

        t0 = time.perf_counter()

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print(f"[{idx:3d}/100] ERROR: Could not open {vname}")
            continue

        total_vframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_vframes <= 0:
            total_vframes = 300

        # Uniformly sample 15 frame indices
        frames_to_sample = 15
        step = max(1, total_vframes // frames_to_sample)
        sample_indices = [min(i * step, total_vframes - 1) for i in range(frames_to_sample)]

        face_crops_pil = []
        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            detection = face_detector.detect_largest_face(frame, margin=0.3)
            if detection is not None:
                crop_bgr, _ = detection
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(crop_rgb)
                face_crops_pil.append(pil_img)

        cap.release()

        n_sampled = len(sample_indices)
        n_faces = len(face_crops_pil)
        total_crops_detected += n_faces

        # Check evidence threshold
        if n_faces < 3:
            status = "INSUFFICIENT_EVIDENCE" if n_faces > 0 else "NO_FACE_DETECTED"
            elapsed = time.perf_counter() - t0
            print(f"[{idx:3d}/100] {vname[:30]:<30} | GT: {gt_str:<4} | Faces: {n_faces:2d}/15 | Status: {status} ({elapsed:.2f}s)")
            
            # Record non-scored entry
            v_res_base = {
                "id": vid,
                "filename": vname,
                "detector": "Baseline_ViT",
                "ground_truth": gt_str,
                "ground_truth_label": gt_label,
                "status": status,
                "sampled_frames": n_sampled,
                "detected_face_frames": n_faces,
                "face_frames": n_faces,
                "inferred_frames": 0,
                "no_face_frames": n_sampled - n_faces,
                "video_score": None,
                "elapsed_time": round(elapsed, 3),
                "elapsed_seconds": round(elapsed, 3)
            }
            v_res_gend = {
                "id": vid,
                "filename": vname,
                "detector": "GenD_PE_L",
                "ground_truth": gt_str,
                "ground_truth_label": gt_label,
                "status": status,
                "sampled_frames": n_sampled,
                "detected_face_frames": n_faces,
                "face_frames": n_faces,
                "inferred_frames": 0,
                "no_face_frames": n_sampled - n_faces,
                "video_score": None,
                "elapsed_time": round(elapsed, 3),
                "elapsed_seconds": round(elapsed, 3)
            }
            results_baseline.append(v_res_base)
            results_gend.append(v_res_gend)
            continue

        # Valid face evidence -> run both detectors on the identical face crops
        status = "ANALYZED"

        # A. Baseline ViT inference
        b_scores = []
        for crop in face_crops_pil:
            b_scores.append(baseline_detector.predict_face_crop(crop))
        b_video_score = float(np.mean(b_scores))

        # B. GenD inference
        g_logits = []
        g_probs = []
        for crop in face_crops_pil:
            logit = predict_frame_logit(crop)
            prob = 1.0 / (1.0 + math.exp(-logit))
            g_logits.append(logit)
            g_probs.append(prob)

        total_frames_inferred += len(face_crops_pil)

        # GenD aggregations
        g_mean = float(np.mean(g_probs))
        g_median = float(np.median(g_probs))
        g_std = float(np.std(g_probs))
        g_min = float(np.min(g_probs))
        g_max = float(np.max(g_probs))

        # 10% trimmed mean
        if len(g_probs) >= 10:
            sorted_probs = sorted(g_probs)
            trim_k = 1  # drop 1 top and 1 bottom
            trimmed_slice = sorted_probs[trim_k:-trim_k]
            g_trimmed = float(np.mean(trimmed_slice))
        else:
            g_trimmed = g_mean

        elapsed = time.perf_counter() - t0

        results_baseline.append({
            "id": vid,
            "filename": vname,
            "detector": "Baseline_ViT",
            "ground_truth": gt_str,
            "ground_truth_label": gt_label,
            "status": status,
            "sampled_frames": n_sampled,
            "detected_face_frames": n_faces,
            "face_frames": n_faces,
            "inferred_frames": n_faces,
            "no_face_frames": n_sampled - n_faces,
            "frame_scores": [round(s, 4) for s in b_scores],
            "video_score": round(b_video_score, 4),
            "prediction_at_0_5": "FAKE" if b_video_score >= 0.5 else "REAL",
            "is_correct": (b_video_score >= 0.5) == (gt_label == 1),
            "elapsed_time": round(elapsed, 3),
            "elapsed_seconds": round(elapsed, 3)
        })

        results_gend.append({
            "id": vid,
            "filename": vname,
            "detector": "GenD_PE_L",
            "ground_truth": gt_str,
            "ground_truth_label": gt_label,
            "status": status,
            "sampled_frames": n_sampled,
            "detected_face_frames": n_faces,
            "face_frames": n_faces,
            "inferred_frames": n_faces,
            "no_face_frames": n_sampled - n_faces,
            "frame_probabilities": [round(p, 4) for p in g_probs],
            "gend_mean": round(g_mean, 4),
            "gend_median": round(g_median, 4),
            "gend_trimmed_mean": round(g_trimmed, 4),
            "video_score": round(g_trimmed, 4),
            "probability_std": round(g_std, 4),
            "min_probability": round(g_min, 4),
            "max_probability": round(g_max, 4),
            "prediction_at_0_5": "FAKE" if g_trimmed >= 0.5 else "REAL",
            "prediction_at_0_4": "FAKE" if g_trimmed >= 0.4 else "REAL",
            "is_correct_at_0_5": (g_trimmed >= 0.5) == (gt_label == 1),
            "is_correct_at_0_4": (g_trimmed >= 0.4) == (gt_label == 1),
            "elapsed_time": round(elapsed, 3),
            "elapsed_seconds": round(elapsed, 3)
        })

        b_correct = "[OK]" if (b_video_score >= 0.5) == (gt_label == 1) else "[X]"
        g_correct = "[OK]" if (g_trimmed >= 0.5) == (gt_label == 1) else "[X]"

        if idx % 5 == 0 or idx == len(videos):
            print(f"[{idx:3d}/100] {vname[:25]:<25} | GT: {gt_str:<4} | Baseline: {b_video_score:.4f} {b_correct:<4} | GenD: {g_trimmed:.4f} {g_correct:<4} | {elapsed:.2f}s", flush=True)

    total_eval_time = time.perf_counter() - total_start_time

    # 2. Filter successfully scored videos for statistical metrics
    valid_baseline = [r for r in results_baseline if r["status"] == "ANALYZED"]
    valid_gend = [r for r in results_gend if r["status"] == "ANALYZED"]

    y_true = [r["ground_truth_label"] for r in valid_baseline]
    b_scores_all = [r["video_score"] for r in valid_baseline]
    g_scores_all = [r["video_score"] for r in valid_gend]

    # Metrics at standard threshold 0.50
    metrics_baseline_05 = calculate_evaluation_metrics(y_true, b_scores_all, threshold=0.50)
    metrics_gend_05 = calculate_evaluation_metrics(y_true, g_scores_all, threshold=0.50)
    metrics_gend_04 = calculate_evaluation_metrics(y_true, g_scores_all, threshold=0.40)

    # Threshold sensitivity sweep across [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    candidate_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    thresh_sweep_baseline = {}
    thresh_sweep_gend = {}

    for t in candidate_thresholds:
        thresh_sweep_baseline[f"thresh_{t:.2f}"] = calculate_evaluation_metrics(y_true, b_scores_all, threshold=t)
        thresh_sweep_gend[f"thresh_{t:.2f}"] = calculate_evaluation_metrics(y_true, g_scores_all, threshold=t)

    # Frame variance analysis between REAL and FAKE videos for GenD
    real_gend_stds = [r["probability_std"] for r in valid_gend if r["ground_truth_label"] == 0]
    fake_gend_stds = [r["probability_std"] for r in valid_gend if r["ground_truth_label"] == 1]
    real_gend_means = [r["gend_mean"] for r in valid_gend if r["ground_truth_label"] == 0]
    fake_gend_means = [r["gend_mean"] for r in valid_gend if r["ground_truth_label"] == 1]

    frame_variance_analysis = {
        "real_videos": {
            "count": len(real_gend_stds),
            "avg_probability_mean": round(float(np.mean(real_gend_means)), 4) if real_gend_means else 0.0,
            "avg_inter_frame_std": round(float(np.mean(real_gend_stds)), 4) if real_gend_stds else 0.0,
        },
        "fake_videos": {
            "count": len(fake_gend_stds),
            "avg_probability_mean": round(float(np.mean(fake_gend_means)), 4) if fake_gend_means else 0.0,
            "avg_inter_frame_std": round(float(np.mean(fake_gend_stds)), 4) if fake_gend_stds else 0.0,
        }
    }

    # Compile Final Output Artifact
    output_data = {
        "metadata": {
            "evaluation_scope": "STAGE 4B CONTROLLED BENCHMARK EVALUATION — NON-PRODUCTION ONLY",
            "statistical_caution": "100 videos is a controlled pilot benchmark, not sufficient to establish universal real-world accuracy.",
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_name": manifest.get("benchmark_name"),
            "dataset_source": manifest.get("dataset_source"),
            "total_videos": len(videos),
            "analyzed_videos": len(valid_baseline),
            "insufficient_evidence_videos": len(videos) - len(valid_baseline),
            "total_frames_inferred": total_frames_inferred,
            "total_evaluation_time_seconds": round(total_eval_time, 2),
            "avg_time_per_video_seconds": round(total_eval_time / len(videos), 3),
            "avg_time_per_crop_seconds": round(total_eval_time / max(1, total_frames_inferred * 2), 4)
        },
        "primary_metrics_at_0_50": {
            "detector_a_baseline_vit": metrics_baseline_05,
            "detector_b_gend_pe_l": metrics_gend_05,
            "detector_b_gend_pe_l_at_0_40": metrics_gend_04
        },
        "threshold_sensitivity_sweep": {
            "note": "NON-PRODUCTION ANALYSIS ONLY",
            "candidate_thresholds": candidate_thresholds,
            "baseline_vit_sweep": thresh_sweep_baseline,
            "gend_pe_l_sweep": thresh_sweep_gend
        },
        "frame_level_analysis": frame_variance_analysis,
        "video_results": {
            "baseline_vit": results_baseline,
            "gend_pe_l": results_gend
        }
    }

    output_path = PROJECT_ROOT / "evaluation" / "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 4B EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Videos: {len(videos)} (Scored: {len(valid_baseline)}, Insufficient: {len(videos) - len(valid_baseline)})")
    print(f"Total Evaluation Time: {total_eval_time:.1f}s ({total_eval_time / len(videos):.2f}s/video)")
    print("\nPRIMARY METRICS COMPARISON (Threshold = 0.50):")
    print(f"{'Metric':<25} | {'Baseline ViT':<15} | {'GenD (0.50)':<15} | {'GenD (0.40)':<15}")
    print("-" * 75)
    for m in ["auc_roc", "accuracy", "precision", "recall", "f1_score", "eer", "average_precision"]:
        b_val = metrics_baseline_05[m]
        g_val = metrics_gend_05[m]
        g4_val = metrics_gend_04[m]
        print(f"{m:<25} | {b_val:<15.4f} | {g_val:<15.4f} | {g4_val:<15.4f}")

    print("\nCONFUSION MATRICES (Threshold = 0.50):")
    print(f"Baseline ViT: TP={metrics_baseline_05['true_positives']}, TN={metrics_baseline_05['true_negatives']}, FP={metrics_baseline_05['false_positives']}, FN={metrics_baseline_05['false_negatives']}")
    print(f"GenD PE_L:    TP={metrics_gend_05['true_positives']}, TN={metrics_gend_05['true_negatives']}, FP={metrics_gend_05['false_positives']}, FN={metrics_gend_05['false_negatives']}")
    print(f"\n[OK] Results written to {output_path}")


if __name__ == "__main__":
    run_benchmark_evaluation()
