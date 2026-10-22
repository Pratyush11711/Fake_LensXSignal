import base64
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T

from ..alignment.face_aligner import FaceAligner
from ..detection.face_tracker import SingleSubjectTracker
from ..detection.yunet_detector import YuNetFaceDetector
from ..models.video_classifier import TemporalDeepfakeDetector
from .calibrator import ThresholdCalibrator

logger = logging.getLogger("deepfake-detector.pipeline")


class TemporalInferenceEngine:
    """
    End-to-end video-level inference engine with:
    - Frame sampling & dynamic sizing
    - YuNet face detection
    - Single-subject IoU tracking
    - MediaPipe 5-point affine alignment
    - Sequence construction (T frames)
    - Xception + Learned Temporal Attention forward pass
    - Calibrated decision & explainability telemetry
    - Insufficient evidence handling
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        face_detector_path: str = "models/face_detection_yunet_2023mar.onnx",
        sequence_length: int = 16,
        target_size: int = 256,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Components
        self.detector = YuNetFaceDetector(model_path=face_detector_path)
        self.tracker = SingleSubjectTracker(iou_threshold=0.35, max_misses=3)
        self.aligner = FaceAligner(target_size=target_size)
        self.calibrator = ThresholdCalibrator(default_threshold=threshold)

        # Transforms
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Initialize PyTorch Model
        self.model = TemporalDeepfakeDetector().to(self.device)
        self.weights_loaded = False

        if weights_path and Path(weights_path).exists():
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.weights_loaded = True
                logger.info("Loaded trained temporal weights from %s", weights_path)
            except Exception as exc:
                logger.warning("Failed to load temporal weights from %s: %s", weights_path, exc)

        self.model.eval()

    def process_video(
        self,
        video_path: str,
        frames_to_sample: int = 24,
        max_duration_sec: int = 300,
        min_face_coverage: float = 0.40,
    ) -> Dict:
        """
        Executes end-to-end temporal deepfake detection on the video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"status": "ERROR", "error": "VIDEO_OPEN_FAILED", "detail": "Could not open video file."}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return {"status": "ERROR", "error": "VIDEO_OPEN_FAILED", "detail": "Video contains no frames."}

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and not np.isnan(fps) and fps > 0:
            duration = total_frames / fps
            if duration > max_duration_sec:
                cap.release()
                return {"status": "ERROR", "error": "VIDEO_TOO_LONG", "detail": f"Video exceeds {max_duration_sec}s limit."}

        self.tracker.reset()
        frame_interval = max(1, total_frames // frames_to_sample)

        aligned_sequence: List[np.ndarray] = []
        timestamps: List[float] = []
        frames_processed = 0
        valid_faces_count = 0

        rep_frame: Optional[np.ndarray] = None
        rep_box: Optional[Tuple[float, float, float, float]] = None
        max_box_area: float = 0.0

        frame_idx = 0
        target_next = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx == target_next:
                frames_processed += 1
                curr_ts = frame_idx / (fps if fps and fps > 0 else 30.0)

                # Resize large frames for fast detection
                h, w = frame.shape[:2]
                max_edge = max(h, w)
                if max_edge > 720:
                    scale = 720.0 / max_edge
                    det_w, det_h = int(w * scale), int(h * scale)
                    det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
                else:
                    det_frame = frame
                    scale = 1.0

                # 1. Detect candidate faces
                candidates_scaled = self.detector.detect(det_frame)
                candidates = []
                for cx, cy, cw, ch, score in candidates_scaled:
                    candidates.append((cx / scale, cy / scale, cw / scale, ch / scale, score))

                # 2. Update single-subject tracker
                tracked_box = self.tracker.update(candidates, frame.shape)

                if tracked_box is not None:
                    valid_faces_count += 1
                    tx, ty, tw, th = tracked_box

                    # Track representative frame
                    box_area = tw * th
                    if box_area > max_box_area:
                        max_box_area = box_area
                        rep_frame = frame.copy()
                        rep_box = tracked_box

                    # 3. Canonical 5-point affine alignment
                    aligned_crop = self.aligner.align_face(frame, face_box=tracked_box)
                    if aligned_crop is not None:
                        aligned_sequence.append(aligned_crop)
                        timestamps.append(round(curr_ts, 2))

                target_next += frame_interval
                if frames_processed >= frames_to_sample:
                    break

            frame_idx += 1

        cap.release()

        # Insufficient evidence checks
        face_coverage = valid_faces_count / max(1, frames_processed)
        if len(aligned_sequence) < (self.sequence_length // 2) or face_coverage < min_face_coverage:
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "result": "UNCERTAIN",
                "detail": f"Face coverage too low ({face_coverage:.1%}) or insufficient tracked faces.",
                "face_coverage": round(face_coverage, 3),
                "frames_analyzed": frames_processed,
                "faces_detected": valid_faces_count,
            }

        # 4. Temporal Sequence Construction (fixed length T)
        indices = np.linspace(0, len(aligned_sequence) - 1, self.sequence_length, dtype=int)
        seq_sampled = [aligned_sequence[i] for i in indices]
        seq_timestamps = [timestamps[i] for i in indices]

        tensor_list = [self.transform(img) for img in seq_sampled]
        seq_tensor = torch.stack(tensor_list, dim=0).unsqueeze(0).to(self.device)  # (1, T, 3, 256, 256)

        # 5. Forward Pass through Temporal Deepfake Detector
        with torch.no_grad():
            logits, probs, attention = self.model(seq_tensor)
            fake_prob = float(probs[0, 0].item())
            attn_weights = [round(float(a), 4) for a in attention[0].tolist()]

        # 6. Calibrated Verdict
        verdict, confidence = self.calibrator.predict(fake_prob)

        # Top 3 most attended frames
        top_indices = np.argsort(attn_weights)[::-1][:3].tolist()

        # 7. Snapshot for visual scan
        face_scan = None
        if rep_frame is not None and rep_box is not None:
            try:
                rh, rw = rep_frame.shape[:2]
                bx, by, bw, bh = rep_box
                norm_box = {
                    "x": round(bx / rw, 4),
                    "y": round(by / rh, 4),
                    "w": round(bw / rw, 4),
                    "h": round(bh / rh, 4),
                }

                max_disp = max(rh, rw)
                if max_disp > 480:
                    s = 480.0 / max_disp
                    disp = cv2.resize(rep_frame, (int(rw * s), int(rh * s)), interpolation=cv2.INTER_AREA)
                else:
                    disp = rep_frame

                ok, buf = cv2.imencode(".jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    b64 = base64.b64encode(buf).decode("utf-8")
                    face_scan = {
                        "snapshot": f"data:image/jpeg;base64,{b64}",
                        "box": norm_box,
                    }
            except Exception:
                face_scan = None

        return {
            "status": "ANALYSIS_COMPLETE",
            "model_backend": "temporal_xception",
            "result": verdict,
            "detection_score": round(fake_prob, 4),
            "confidence": f"{confidence:.2%}",
            "decision_threshold": self.calibrator.threshold,
            "frames_analyzed": frames_processed,
            "faces_detected": valid_faces_count,
            "face_coverage": round(face_coverage, 3),
            "timestamps": seq_timestamps,
            "attention_weights": attn_weights,
            "top_attended_frames": top_indices,
            "face_scan": face_scan,
        }
