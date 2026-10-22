"""
evaluation/run_stage6e_lipforensics_benchmark.py

STAGE 6E — CONTROLLED LIPFORENSICS BENCHMARK (PARALLEL MULTI-WORKER ENGINE)
Evaluates the verified LipForensics pretrained model (lipforensics_ff.pth)
on the exact 100-video balanced benchmark dataset (D:/benchmark_dataset).

Outputs:
- evaluation/STAGE_6E_LIPFORENSICS_BENCHMARK.json
- evaluation/STAGE_6E_LIPFORENSICS_BENCHMARK.md
"""

import concurrent.futures
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn import metrics

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detectors.lipforensics_detector import get_lipforensics_model, Lipreading


# Thread-local storage for detector instances
_thread_local = threading.local()


def get_thread_resources(yunet_model_path: str, checkpoint_path: Optional[str] = None):
    if not hasattr(_thread_local, "detector"):
        _thread_local.detector = cv2.FaceDetectorYN.create(
            model=yunet_model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )
    if not hasattr(_thread_local, "model"):
        _thread_local.model = get_lipforensics_model(checkpoint_path=checkpoint_path, device="cpu")
    return _thread_local.detector, _thread_local.model


def extract_mouth_crop(detector, frame_bgr: np.ndarray, target_size: Tuple[int, int] = (88, 88)) -> Optional[np.ndarray]:
    h, w = frame_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame_bgr)

    if faces is None or len(faces) == 0:
        return None

    primary_face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
    fw, fh = float(primary_face[2]), float(primary_face[3])
    rm_x, rm_y = float(primary_face[10]), float(primary_face[11])
    lm_x, lm_y = float(primary_face[12]), float(primary_face[13])

    center_x = (rm_x + lm_x) / 2.0
    center_y = (rm_y + lm_y) / 2.0

    mouth_span = max(float(np.hypot(rm_x - lm_x, rm_y - lm_y)), fw * 0.45, 16.0)
    half_size = int(round(mouth_span * 0.85))
    half_size = max(half_size, 16)

    x1 = max(0, int(round(center_x - half_size)))
    x2 = min(w, int(round(center_x + half_size)))
    y1 = max(0, int(round(center_y - half_size)))
    y2 = min(h, int(round(center_y + half_size)))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)
    return resized


def process_single_video(
    v_entry: Dict[str, Any],
    idx: int,
    total_videos: int,
    yunet_model_path: str,
    frames_per_clip: int = 25,
    num_clips_per_video: int = 4,
    target_crop_size: Tuple[int, int] = (88, 88)
) -> Dict[str, Any]:
    t0 = time.time()
    vid_id = v_entry.get("id", f"video_{idx}")
    local_path = v_entry.get("local_path")
    gt_label_str = v_entry.get("ground_truth", "UNKNOWN")
    gt_label_int = 1 if gt_label_str == "FAKE" else 0

    detector, model = get_thread_resources(yunet_model_path)

    if not os.path.exists(local_path):
        return {
            "video_id": vid_id,
            "filename": os.path.basename(local_path),
            "ground_truth": gt_label_str,
            "ground_truth_label": gt_label_int,
            "local_path": local_path,
            "status": "ERROR",
            "error": f"File not found: {local_path}",
            "fake_score": None,
            "mean_logit": None,
            "clips_analyzed": 0,
            "inference_time_sec": round(time.time() - t0, 4)
        }

    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        return {
            "video_id": vid_id,
            "filename": os.path.basename(local_path),
            "ground_truth": gt_label_str,
            "ground_truth_label": gt_label_int,
            "local_path": local_path,
            "status": "ERROR",
            "error": "Failed to open video with OpenCV",
            "fake_score": None,
            "mean_logit": None,
            "clips_analyzed": 0,
            "inference_time_sec": round(time.time() - t0, 4)
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < frames_per_clip:
        cap.release()
        return {
            "video_id": vid_id,
            "filename": os.path.basename(local_path),
            "ground_truth": gt_label_str,
            "ground_truth_label": gt_label_int,
            "local_path": local_path,
            "status": "INSUFFICIENT_MOUTH_EVIDENCE",
            "error": f"Total video frames ({total_frames}) < required clip length ({frames_per_clip})",
            "fake_score": None,
            "mean_logit": None,
            "clips_analyzed": 0,
            "inference_time_sec": round(time.time() - t0, 4)
        }

    ratios = [0.05, 0.30, 0.55, 0.80] if num_clips_per_video == 4 else [0.10, 0.50]
    max_start = max(0, total_frames - frames_per_clip - 1)
    start_indices = [min(int(total_frames * r), max_start) for r in ratios]
    start_indices = sorted(list(set(start_indices)))

    clip_logits: List[float] = []
    valid_face_frames = 0
    valid_mouth_crops = 0

    with torch.no_grad():
        for start_frame in start_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_clip_crops: List[np.ndarray] = []

            for _ in range(frames_per_clip):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                crop = extract_mouth_crop(detector, frame, target_crop_size)
                if crop is not None:
                    valid_face_frames += 1
                    valid_mouth_crops += 1
                    current_clip_crops.append(crop)

            if len(current_clip_crops) == frames_per_clip:
                clip_np = np.stack(current_clip_crops, axis=0) # (25, 88, 88)
                clip_tensor = torch.from_numpy(clip_np).float() / 255.0
                clip_tensor = (clip_tensor - 0.421) / 0.165
                clip_tensor = clip_tensor.unsqueeze(0).unsqueeze(0) # (1, 1, 25, 88, 88)

                logit = model(clip_tensor, lengths=[frames_per_clip])
                raw_val = float(logit.item()) if hasattr(logit, "item") else float(logit[0, 0])
                clip_logits.append(raw_val)

    cap.release()
    elapsed = round(time.time() - t0, 4)

    if len(clip_logits) == 0:
        status_str = "NO_FACE_DETECTED" if valid_mouth_crops == 0 else "INSUFFICIENT_MOUTH_EVIDENCE"
        err_msg = "No faces or mouth landmarks detected" if valid_mouth_crops == 0 else f"Extracted {valid_mouth_crops} crops, needed {frames_per_clip}"
        res = {
            "video_id": vid_id,
            "filename": os.path.basename(local_path),
            "ground_truth": gt_label_str,
            "ground_truth_label": gt_label_int,
            "local_path": local_path,
            "status": status_str,
            "error": err_msg,
            "fake_score": None,
            "mean_logit": None,
            "clips_analyzed": 0,
            "valid_mouth_crops": valid_mouth_crops,
            "inference_time_sec": elapsed
        }
        print(f"[{idx+1:03d}/{total_videos:03d}] {vid_id} ({gt_label_str}): SKIPPED ({status_str}) in {elapsed:.2f}s", flush=True)
        return res

    mean_logit = float(np.mean(clip_logits))
    fake_score = 1.0 / (1.0 + math.exp(-mean_logit))

    res = {
        "video_id": vid_id,
        "filename": os.path.basename(local_path),
        "ground_truth": gt_label_str,
        "ground_truth_label": gt_label_int,
        "local_path": local_path,
        "status": "ANALYZED",
        "fake_score": round(fake_score, 4),
        "mean_logit": round(mean_logit, 4),
        "raw_clip_logits": [round(l, 4) for l in clip_logits],
        "clips_analyzed": len(clip_logits),
        "valid_mouth_crops": valid_mouth_crops,
        "total_frames_in_video": total_frames,
        "inference_time_sec": elapsed,
        "error": None
    }
    print(f"[{idx+1:03d}/{total_videos:03d}] {vid_id} ({gt_label_str}): FakeScore={fake_score:.4f} (MeanLogit={mean_logit:.2f}, {len(clip_logits)} clips) in {elapsed:.2f}s", flush=True)
    return res


def compute_metrics(y_true: List[int], y_scores: List[float], threshold: float = 0.50) -> Dict[str, Any]:
    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_scores)

    if len(y_true_np) == 0:
        return {}

    try:
        auc = float(metrics.roc_auc_score(y_true_np, y_scores_np))
    except Exception:
        auc = 0.0

    try:
        ap = float(metrics.average_precision_score(y_true_np, y_scores_np))
    except Exception:
        ap = 0.0

    try:
        fpr_arr, tpr_arr, _ = metrics.roc_curve(y_true_np, y_scores_np, pos_label=1)
        fnr_arr = 1.0 - tpr_arr
        eer_idx = int(np.nanargmin(np.abs(fnr_arr - fpr_arr)))
        eer = float(fpr_arr[eer_idx])
    except Exception:
        eer = 0.0

    y_pred = (y_scores_np >= threshold).astype(int)

    tp = int(np.sum((y_true_np == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true_np == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true_np == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true_np == 1) & (y_pred == 0)))

    total = len(y_true_np)
    acc = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "auc_roc": round(auc, 4),
        "average_precision": round(ap, 4),
        "eer": round(eer, 4),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall_sensitivity": round(recall, 4),
        "specificity_tnr": round(specificity, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    }


def main():
    print("=" * 80)
    print("STAGE 6E — CONTROLLED LIPFORENSICS BENCHMARK (PARALLEL MULTI-CORE)")
    print("=" * 80)

    manifest_path = Path("D:/benchmark_dataset/manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest_data = json.load(f)

    videos = manifest_data.get("videos", [])
    total_videos = len(videos)
    yunet_model_path = str(PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx")

    print(f"Loaded {total_videos} videos from {manifest_path}")
    print("Starting parallel evaluation across 6 worker threads...\n", flush=True)

    start_time = time.time()
    results = [None] * total_videos

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_idx = {
            executor.submit(
                process_single_video, v_entry, idx, total_videos, yunet_model_path
            ): idx
            for idx, v_entry in enumerate(videos)
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
                results[idx] = res
            except Exception as e:
                print(f"[ERROR] Exception processing video index {idx}: {e}", flush=True)

    total_elapsed = round(time.time() - start_time, 2)
    print(f"\n[STAGE 6E] Benchmark completed in {total_elapsed}s ({total_elapsed/total_videos:.2f}s per video).")

    # Compile metrics
    y_true = []
    y_scores = []
    real_scores = []
    fake_scores = []
    analyzed_count = 0
    skipped_count = 0

    for r in results:
        if r is not None and r.get("status") == "ANALYZED" and r.get("fake_score") is not None:
            analyzed_count += 1
            gt = r["ground_truth_label"]
            score = r["fake_score"]
            y_true.append(gt)
            y_scores.append(score)
            if gt == 0:
                real_scores.append(score)
            else:
                fake_scores.append(score)
        else:
            skipped_count += 1

    metrics_dict = compute_metrics(y_true, y_scores, threshold=0.50)

    score_distribution = {
        "real_videos": {
            "count": len(real_scores),
            "mean_score": round(float(np.mean(real_scores)), 4) if real_scores else 0.0,
            "std_score": round(float(np.std(real_scores)), 4) if real_scores else 0.0,
            "min_score": round(float(np.min(real_scores)), 4) if real_scores else 0.0,
            "max_score": round(float(np.max(real_scores)), 4) if real_scores else 0.0,
        },
        "fake_videos": {
            "count": len(fake_scores),
            "mean_score": round(float(np.mean(fake_scores)), 4) if fake_scores else 0.0,
            "std_score": round(float(np.std(fake_scores)), 4) if fake_scores else 0.0,
            "min_score": round(float(np.min(fake_scores)), 4) if fake_scores else 0.0,
            "max_score": round(float(np.max(fake_scores)), 4) if fake_scores else 0.0,
        }
    }

    full_output = {
        "metadata": {
            "stage": "STAGE_6E",
            "stage_title": "Controlled LipForensics Benchmark",
            "date": "2026-09-07",
            "auditor": "Antigravity Agentic AI",
            "dataset_name": "Google/Jigsaw DeepfakeDetection (FaceForensics++ Component)",
            "dataset_path": "D:/benchmark_dataset",
            "total_videos_in_manifest": total_videos,
            "total_evaluated_videos": analyzed_count,
            "total_skipped_videos": skipped_count,
            "total_benchmark_time_seconds": total_elapsed,
            "avg_time_per_video_seconds": round(total_elapsed / total_videos, 3),
            "model_name": "LipForensics (CVPR 2021)",
            "checkpoint": "lipforensics_ff.pth (SHA-256: 4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d)",
            "threshold_used_for_binary_metrics": 0.50
        },
        "primary_metrics_at_0_50": metrics_dict,
        "score_distribution": score_distribution,
        "comparison_table": {
            "detector_a_baseline_vit": {
                "auc_roc": 0.4548,
                "accuracy": 0.4400,
                "precision": 0.4605,
                "recall": 0.7000,
                "specificity": 0.1800,
                "f1_score": 0.5556,
                "fpr": 0.8200,
                "fnr": 0.3000
            },
            "detector_b_gend_pe_l": {
                "auc_roc": 0.4224,
                "accuracy": 0.5100,
                "precision": 0.5106,
                "recall": 0.4800,
                "specificity": 0.5400,
                "f1_score": 0.4948,
                "fpr": 0.4600,
                "fnr": 0.5200
            },
            "detector_c_lipforensics": {
                "auc_roc": metrics_dict.get("auc_roc"),
                "accuracy": metrics_dict.get("accuracy"),
                "precision": metrics_dict.get("precision"),
                "recall": metrics_dict.get("recall_sensitivity"),
                "specificity": metrics_dict.get("specificity_tnr"),
                "f1_score": metrics_dict.get("f1_score"),
                "fpr": metrics_dict.get("false_positive_rate"),
                "fnr": metrics_dict.get("false_negative_rate")
            }
        },
        "per_video_results": results
    }

    out_file = PROJECT_ROOT / "evaluation" / "STAGE_6E_LIPFORENSICS_BENCHMARK.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[STAGE 6E] Benchmark artifact written to: {out_file}")
    print(f"ROC-AUC: {metrics_dict.get('auc_roc')}")
    print(f"Accuracy: {metrics_dict.get('accuracy')}")
    print(f"Precision: {metrics_dict.get('precision')}")
    print(f"Recall: {metrics_dict.get('recall_sensitivity')}")
    print(f"Specificity: {metrics_dict.get('specificity_tnr')}")
    print(f"F1 Score: {metrics_dict.get('f1_score')}")


if __name__ == "__main__":
    main()
