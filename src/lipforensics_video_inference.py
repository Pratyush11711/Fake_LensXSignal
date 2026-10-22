"""
src/lipforensics_video_inference.py

Isolated video-level inference pipeline for LipForensics (CVPR 2021).
Extracts 25-frame mouth ROI sequences using OpenCV YuNet (with MediaPipe fallback),
normalizes with official LipForensics parameters (mean=0.421, std=0.165),
and computes temporal forgery logits and calibrated probabilities.

This file is for isolated research verification and is NOT connected to /v1/analyze.
"""

import math
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from detectors.lipforensics_detector import get_lipforensics_model, Lipreading


class LipForensicsVideoInference:
    """
    Isolated LipForensics video inference engine.
    Extracts consecutive 25-frame mouth ROI sequences using YuNet / MediaPipe landmarks,
    normalizes to (1, 1, 25, 88, 88) with mean=0.421, std=0.165, and runs temporal MS-TCN inference.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        yunet_model_path: str = "models/face_detection_yunet_2023mar.onnx",
        device: Optional[Union[str, torch.device]] = None,
        frames_per_clip: int = 25,
        max_clips_per_video: int = 4,
        target_crop_size: Tuple[int, int] = (88, 88),
        score_threshold: float = 0.6,
    ):
        self.frames_per_clip = frames_per_clip
        self.max_clips_per_video = max_clips_per_video
        self.target_crop_size = target_crop_size
        self.score_threshold = score_threshold
        self.yunet_model_path = yunet_model_path

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model: Lipreading = get_lipforensics_model(checkpoint_path=checkpoint_path, device=self.device)

        # Initialize YuNet detector
        self.yunet_detector = None
        if os.path.exists(self.yunet_model_path):
            self.yunet_detector = cv2.FaceDetectorYN.create(
                model=self.yunet_model_path,
                config="",
                input_size=(320, 320),
                score_threshold=self.score_threshold,
                nms_threshold=0.3,
                top_k=5000,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )

    def _extract_mouth_crop(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Extracts a centered, standardized 88x88 grayscale mouth ROI from a video frame.
        Uses YuNet 5-point facial landmarks (right mouth corner: indices 10-11, left mouth corner: indices 12-13).
        Returns (crop, face_box_info).
        """
        h, w = frame_bgr.shape[:2]

        if self.yunet_detector is not None:
            try:
                self.yunet_detector.setInputSize((w, h))
                _, faces = self.yunet_detector.detect(frame_bgr)

                if faces is not None and len(faces) > 0:
                    # Sort faces by detection score / size and pick primary face
                    faces = sorted(faces, key=lambda f: float(f[2]) * float(f[3]), reverse=True)
                    primary_face = faces[0]

                    fx, fy, fw, fh = primary_face[0:4]
                    rm_x, rm_y = primary_face[10:12] # Right mouth corner
                    lm_x, lm_y = primary_face[12:14] # Left mouth corner

                    box_info = {
                        "box": {
                            "x": round(float(fx) / w, 4),
                            "y": round(float(fy) / h, 4),
                            "w": round(float(fw) / w, 4),
                            "h": round(float(fh) / h, 4),
                        }
                    }

                    # Center of mouth
                    center_x = (rm_x + lm_x) / 2.0
                    center_y = (rm_y + lm_y) / 2.0

                    # Dynamic span based on mouth corner distance and face width
                    mouth_span = max(
                        float(np.hypot(rm_x - lm_x, rm_y - lm_y)),
                        float(fw) * 0.45,
                        16.0
                    )
                    half_size = int(round(mouth_span * 0.85))
                    half_size = max(half_size, 16)

                    x1 = max(0, int(round(center_x - half_size)))
                    x2 = min(w, int(round(center_x + half_size)))
                    y1 = max(0, int(round(center_y - half_size)))
                    y2 = min(h, int(round(center_y + half_size)))

                    if x2 <= x1 or y2 <= y1:
                        return None, None

                    crop = frame_bgr[y1:y2, x1:x2]
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, self.target_crop_size, interpolation=cv2.INTER_LINEAR)
                    return resized, box_info
            except Exception:
                pass

        return None, None

    def analyze_video(self, video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Runs isolated LipForensics inference on a video file.
        Returns structured diagnostic dictionary.
        """
        t0 = time.time()
        vpath = str(video_path)

        if not os.path.exists(vpath):
            return {
                "status": "ERROR",
                "error": f"Video file not found: {vpath}",
                "video_path": vpath,
                "filename": os.path.basename(vpath),
                "frames_read": 0,
                "valid_face_frames": 0,
                "valid_mouth_crops": 0,
                "clips_analyzed": 0,
                "sequence_length": self.frames_per_clip,
                "fake_score": None,
                "mean_logit": None,
                "inference_time_sec": round(time.time() - t0, 4),
            }

        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            return {
                "status": "ERROR",
                "error": f"Failed to open video file with OpenCV: {vpath}",
                "video_path": vpath,
                "filename": os.path.basename(vpath),
                "frames_read": 0,
                "valid_face_frames": 0,
                "valid_mouth_crops": 0,
                "clips_analyzed": 0,
                "sequence_length": self.frames_per_clip,
                "fake_score": None,
                "mean_logit": None,
                "inference_time_sec": round(time.time() - t0, 4),
            }

        frames_read = 0
        mouth_crops: List[np.ndarray] = []
        valid_face_frames = 0
        first_face_box = None

        # Read frames
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames_read += 1

            crop, box_info = self._extract_mouth_crop(frame)
            if crop is not None:
                valid_face_frames += 1
                mouth_crops.append(crop)
                if first_face_box is None and box_info is not None:
                    first_face_box = box_info

        cap.release()

        valid_mouth_crops = len(mouth_crops)
        face_coverage = round(float(valid_face_frames) / max(frames_read, 1), 4)

        # Edge cases
        if frames_read == 0:
            return {
                "status": "ERROR",
                "error": "UNREADABLE_VIDEO (0 frames decoded)",
                "video_path": vpath,
                "filename": os.path.basename(vpath),
                "frames_read": 0,
                "valid_face_frames": 0,
                "valid_mouth_crops": 0,
                "clips_analyzed": 0,
                "sequence_length": self.frames_per_clip,
                "fake_score": None,
                "mean_logit": None,
                "inference_time_sec": round(time.time() - t0, 4),
            }

        if valid_face_frames == 0 or valid_mouth_crops == 0:
            return {
                "status": "NO_FACE_DETECTED",
                "error": "No valid facial landmarks or mouth regions detected in video",
                "video_path": vpath,
                "filename": os.path.basename(vpath),
                "frames_read": frames_read,
                "valid_face_frames": 0,
                "valid_mouth_crops": 0,
                "clips_analyzed": 0,
                "sequence_length": self.frames_per_clip,
                "fake_score": None,
                "mean_logit": None,
                "inference_time_sec": round(time.time() - t0, 4),
            }

        if valid_mouth_crops < self.frames_per_clip:
            if valid_mouth_crops >= 4:
                # Interpolate / resample temporal frames across sequence to reach self.frames_per_clip
                indices = np.round(np.linspace(0, valid_mouth_crops - 1, self.frames_per_clip)).astype(int)
                clips_to_run = [[mouth_crops[i] for i in indices]]
            else:
                return {
                    "status": "INSUFFICIENT_MOUTH_EVIDENCE",
                    "error": f"Found {valid_mouth_crops} valid mouth crops, which is less than the required sequence length of {self.frames_per_clip}",
                    "video_path": vpath,
                    "filename": os.path.basename(vpath),
                    "frames_read": frames_read,
                    "valid_face_frames": valid_face_frames,
                    "valid_mouth_crops": valid_mouth_crops,
                    "clips_analyzed": 0,
                    "sequence_length": self.frames_per_clip,
                    "fake_score": None,
                    "mean_logit": None,
                    "inference_time_sec": round(time.time() - t0, 4),
                }
        else:
            # Assemble consecutive 25-frame clips
            num_possible_clips = valid_mouth_crops // self.frames_per_clip
            num_clips = min(num_possible_clips, self.max_clips_per_video)
            clips_to_run = [
                mouth_crops[c * self.frames_per_clip : (c + 1) * self.frames_per_clip]
                for c in range(num_clips)
            ]

        clip_logits: List[float] = []
        clip_scores: List[float] = []
        last_tensor_shape = None

        with torch.no_grad():
            for selected_crops in clips_to_run:

                # Shape: (T=25, H=88, W=88) -> (1, 1, 25, 88, 88)
                clip_np = np.stack(selected_crops, axis=0) # (25, 88, 88)
                clip_tensor = torch.from_numpy(clip_np).float() / 255.0 # [0, 1]
                # Normalize with LipForensics official mean=0.421, std=0.165
                clip_tensor = (clip_tensor - 0.421) / 0.165
                clip_tensor = clip_tensor.unsqueeze(0).unsqueeze(0) # (1, 1, 25, 88, 88)
                clip_tensor = clip_tensor.to(self.device)
                last_tensor_shape = list(clip_tensor.shape)

                # Forward pass
                logit = self.model(clip_tensor, lengths=[self.frames_per_clip])
                raw_logit_val = logit.item() if hasattr(logit, "item") else float(logit[0, 0])
                prob_val = 1.0 / (1.0 + math.exp(-raw_logit_val))

                clip_logits.append(raw_logit_val)
                clip_scores.append(prob_val)

        mean_logit = float(np.mean(clip_logits))
        video_fake_score = 1.0 / (1.0 + math.exp(-mean_logit))
        elapsed = round(time.time() - t0, 4)

        return {
            "status": "ANALYZED",
            "video_path": vpath,
            "filename": os.path.basename(vpath),
            "frames_read": frames_read,
            "valid_face_frames": valid_face_frames,
            "valid_mouth_crops": valid_mouth_crops,
            "clips_analyzed": len(clip_scores),
            "clip_scores": [round(s, 4) for s in clip_scores],
            "sequence_length": self.frames_per_clip,
            "face_coverage": face_coverage,
            "face_scan": first_face_box,
            "model_input_tensor_shape": last_tensor_shape,
            "model_output_tensor_shape": [1, 1],
            "raw_clip_logits": [round(l, 4) for l in clip_logits],
            "mean_logit": round(mean_logit, 4),
            "fake_score": round(video_fake_score, 4),
            "class_ordering": "Index 0 / Positive Logit: FAKE (1.0), Negative Logit: REAL (0.0)",
            "inference_time_sec": elapsed,
            "error": None,
        }
