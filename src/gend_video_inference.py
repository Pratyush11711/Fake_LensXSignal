"""
src/gend_video_inference.py

Isolated Video Inference Pipeline using GenD (yermandy/GenD_PE_L).
Additive backend module for forensic video deepfake analysis.

Architecture:
  VIDEO -> Frame Sampling -> YuNet Face Detection -> Face Crop Extraction ->
  GenD Inference -> Per-frame Logits/Probabilities -> Robust Temporal Aggregation ->
  Diagnostic Result Object.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# Import existing YuNet Face Detector
from src.detection.yunet_detector import YuNetFaceDetector

# Import verified isolated GenD wrapper
from detectors.gend_detector import (
    get_gend_model,
    predict_frame_logit,
    predict_frame_prob,
)

logger = logging.getLogger("deepfake-detector.gend-video")


def calculate_trimmed_mean(values: List[float], trim_ratio: float = 0.1) -> float:
    """
    Computes deterministic trimmed mean removing extreme top/bottom percentiles.
    If fewer than 5 values exist, returns the standard arithmetic mean.
    """
    if not values:
        return 0.0
    n = len(values)
    if n < 5:
        return float(np.mean(values))

    sorted_vals = sorted(values)
    k = max(1, int(np.floor(n * trim_ratio)))
    # Ensure we retain at least 1 element
    if 2 * k >= n:
        k = max(1, (n - 1) // 2)

    trimmed = sorted_vals[k : n - k]
    if not trimmed:
        trimmed = sorted_vals
    return float(np.mean(trimmed))


class GenDVideoInferencePipeline:
    """
    Forensic video inference pipeline leveraging GenD and YuNet face detection.
    Maintains frame accounting, evidence quality assessment, and robust temporal aggregation.
    """

    def __init__(
        self,
        face_detector_path: str = "models/face_detection_yunet_2023mar.onnx",
        min_face_size: int = 40,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        crop_margin: float = 0.25,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.crop_margin = crop_margin
        self.min_face_size = min_face_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reusable YuNet face detector
        self.detector = YuNetFaceDetector(
            model_path=face_detector_path,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            min_face_size=min_face_size,
        )

        # Pre-warm GenD singleton model
        self.model = get_gend_model(device=self.device)

    def analyze_video(
        self,
        video_path: str,
        frames_to_sample: int = 15,
        max_edge: int = 720,
        max_duration_sec: int = 300,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end GenD deepfake analysis on a video file.

        Args:
            video_path: Path to the target video file.
            frames_to_sample: Number of frames to sample uniformly.
            max_edge: Maximum spatial dimension for face detection scaling.
            max_duration_sec: Maximum allowed video length in seconds.

        Returns:
            Dict containing full diagnostic metrics, temporal aggregations,
            evidence quality, and per-frame accounting.
        """
        start_time = time.perf_counter()

        if not os.path.isfile(video_path):
            return {
                "status": "ERROR",
                "error": "VIDEO_NOT_FOUND",
                "detail": f"File does not exist: {video_path}",
                "elapsed_seconds": round(time.perf_counter() - start_time, 4),
            }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "status": "ERROR",
                "error": "VIDEO_OPEN_FAILED",
                "detail": "Failed to open video container with OpenCV.",
                "elapsed_seconds": round(time.perf_counter() - start_time, 4),
            }

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return {
                "status": "ERROR",
                "error": "VIDEO_OPEN_FAILED",
                "detail": "Video contains zero readable frames.",
                "elapsed_seconds": round(time.perf_counter() - start_time, 4),
            }

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and not np.isnan(fps) and fps > 0:
            duration = total_frames / fps
            if duration > max_duration_sec:
                cap.release()
                return {
                    "status": "ERROR",
                    "error": "VIDEO_TOO_LONG",
                    "detail": f"Video duration ({duration:.1f}s) exceeds {max_duration_sec}s limit.",
                    "elapsed_seconds": round(time.perf_counter() - start_time, 4),
                }
        else:
            fps = 30.0

        frame_interval = max(1, total_frames // frames_to_sample)
        frame_records: List[Dict[str, Any]] = []
        inferred_logits: List[float] = []
        inferred_probs: List[float] = []

        frame_index = 0
        target_next = 0
        sampled_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index == target_next:
                sampled_count += 1
                curr_ts = frame_index / fps
                h, w = frame.shape[:2]

                # Scale large frames for detection efficiency
                curr_max = max(h, w)
                if curr_max > max_edge:
                    scale = max_edge / curr_max
                    new_w, new_h = int(w * scale), int(h * scale)
                    det_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    det_frame = frame
                    scale = 1.0

                candidates = self.detector.detect(det_frame)

                if candidates:
                    # Select largest face candidate by area
                    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
                    cx, cy, cw, ch, score = candidates[0]

                    # Unscale coordinates back to original frame space
                    orig_x = cx / scale
                    orig_y = cy / scale
                    orig_w = cw / scale
                    orig_h = ch / scale

                    # Expand bounding box with crop margin
                    mx = orig_w * self.crop_margin
                    my = orig_h * self.crop_margin
                    x1 = max(0, int(round(orig_x - mx)))
                    y1 = max(0, int(round(orig_y - my)))
                    x2 = min(w, int(round(orig_x + orig_w + mx)))
                    y2 = min(h, int(round(orig_y + orig_h + my)))

                    if x2 > x1 and y2 > y1:
                        crop_bgr = frame[y1:y2, x1:x2]
                        if crop_bgr.size > 0:
                            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                            crop_pil = Image.fromarray(crop_rgb)

                            # Run isolated GenD inference
                            fake_logit = predict_frame_logit(crop_pil, model=self.model, device=self.device)
                            fake_prob = predict_frame_prob(crop_pil, model=self.model, device=self.device)

                            inferred_logits.append(fake_logit)
                            inferred_probs.append(fake_prob)

                            frame_records.append(
                                {
                                    "frame_index": frame_index,
                                    "timestamp": round(curr_ts, 3),
                                    "face_detected": True,
                                    "box": [round(orig_x, 1), round(orig_y, 1), round(orig_w, 1), round(orig_h, 1)],
                                    "detection_score": round(score, 4),
                                    "fake_logit": round(fake_logit, 6),
                                    "fake_probability": round(fake_prob, 6),
                                }
                            )
                        else:
                            frame_records.append(
                                {
                                    "frame_index": frame_index,
                                    "timestamp": round(curr_ts, 3),
                                    "face_detected": False,
                                    "fake_logit": None,
                                    "fake_probability": None,
                                }
                            )
                    else:
                        frame_records.append(
                            {
                                "frame_index": frame_index,
                                "timestamp": round(curr_ts, 3),
                                "face_detected": False,
                                "fake_logit": None,
                                "fake_probability": None,
                            }
                        )
                else:
                    # No face detected in this frame
                    frame_records.append(
                        {
                            "frame_index": frame_index,
                            "timestamp": round(curr_ts, 3),
                            "face_detected": False,
                            "fake_logit": None,
                            "fake_probability": None,
                        }
                    )

                target_next += frame_interval
                if sampled_count >= frames_to_sample:
                    break

            frame_index += 1

        cap.release()
        elapsed_seconds = time.perf_counter() - start_time

        # Accounting Breakdown
        sampled_frames = len(frame_records)
        face_detected_frames = sum(1 for f in frame_records if f["face_detected"])
        frames_inferred_by_gend = len(inferred_probs)
        frames_without_face = sampled_frames - face_detected_frames
        face_coverage_ratio = round(face_detected_frames / max(1, sampled_frames), 4)

        # Handle No Faces Detected
        if frames_inferred_by_gend == 0:
            return {
                "status": "NO_FACE_DETECTED",
                "model_name": "yermandy/GenD_PE_L",
                "sampled_frames": sampled_frames,
                "face_detected_frames": 0,
                "frames_inferred_by_gend": 0,
                "frames_without_face": frames_without_face,
                "face_coverage_ratio": 0.0,
                "evidence_quality": "INSUFFICIENT",
                "gend_mean_fake_probability": None,
                "gend_median_fake_probability": None,
                "gend_trimmed_mean_fake_probability": None,
                "gend_min_fake_probability": None,
                "gend_max_fake_probability": None,
                "probability_std": None,
                "gend_mean_fake_logit": None,
                "gend_median_fake_logit": None,
                "per_frame_evidence": frame_records,
                "elapsed_seconds": round(elapsed_seconds, 4),
            }

        # Temporal Aggregations
        mean_fake_prob = round(float(np.mean(inferred_probs)), 4)
        median_fake_prob = round(float(np.median(inferred_probs)), 4)
        trimmed_mean_fake_prob = round(calculate_trimmed_mean(inferred_probs, trim_ratio=0.1), 4)
        min_fake_prob = round(float(np.min(inferred_probs)), 4)
        max_fake_prob = round(float(np.max(inferred_probs)), 4)
        prob_std = round(float(np.std(inferred_probs, ddof=1 if len(inferred_probs) > 1 else 0)), 4)

        mean_fake_logit = round(float(np.mean(inferred_logits)), 4)
        median_fake_logit = round(float(np.median(inferred_logits)), 4)

        # Deterministic Evidence Quality Rating
        if face_coverage_ratio < 0.35 or frames_inferred_by_gend < 3:
            evidence_quality = "LOW"
            status = "INSUFFICIENT_EVIDENCE"
        elif face_coverage_ratio >= 0.80 and frames_inferred_by_gend >= 8 and prob_std < 0.25:
            evidence_quality = "HIGH"
            status = "ANALYZED"
        else:
            evidence_quality = "MEDIUM"
            status = "ANALYZED"

        return {
            "status": status,
            "model_name": "yermandy/GenD_PE_L",
            "sampled_frames": sampled_frames,
            "face_detected_frames": face_detected_frames,
            "frames_inferred_by_gend": frames_inferred_by_gend,
            "frames_without_face": frames_without_face,
            "face_coverage_ratio": face_coverage_ratio,
            "evidence_quality": evidence_quality,
            "gend_mean_fake_probability": mean_fake_prob,
            "gend_median_fake_probability": median_fake_prob,
            "gend_trimmed_mean_fake_probability": trimmed_mean_fake_prob,
            "gend_min_fake_probability": min_fake_prob,
            "gend_max_fake_probability": max_fake_prob,
            "probability_std": prob_std,
            "gend_mean_fake_logit": mean_fake_logit,
            "gend_median_fake_logit": median_fake_logit,
            "per_frame_evidence": frame_records,
            "elapsed_seconds": round(elapsed_seconds, 4),
        }
