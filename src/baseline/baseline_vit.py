import base64
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
try:
    import mediapipe as mp
except ImportError:
    mp = None
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger("deepfake-detector.baseline")

SELECTED_LANDMARK_INDICES: List[Tuple[int, str]] = [
    (33, "eye"),
    (133, "eye"),
    (362, "eye"),
    (263, "eye"),
    (70, "eyebrow"),
    (334, "eyebrow"),
    (168, "nose"),
    (1, "nose"),
    (61, "mouth"),
    (291, "mouth"),
    (0, "mouth"),
    (17, "mouth"),
    (172, "jaw"),
    (397, "jaw"),
    (152, "jaw"),
]


class BaselineViTDetector:
    """
    Encapsulates the 2D Vision Transformer baseline pipeline:
    Frame sampling -> YuNet face detection -> 25% boundary crop ->
    Pretrained ViT inference -> Softmax averaging -> Binary verdict.
    """

    def __init__(
        self,
        model_name: str = "dima806/deepfake_vs_real_image_detection",
        face_detector_model: str = "models/face_detection_yunet_2023mar.onnx",
        crop_margin: float = 0.25,
        min_face_size: int = 40,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.crop_margin = crop_margin
        self.min_face_size = min_face_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ViT Model & Processor
        logger.info("Loading baseline ViT model '%s' on '%s'...", self.model_name, self.device)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self.fake_class_id = -1
        for cid, label in self.model.config.id2label.items():
            if "fake" in label.lower():
                self.fake_class_id = cid
                break
        if self.fake_class_id == -1:
            raise RuntimeError(f"Could not find 'fake' class in id2label: {self.model.config.id2label}")

        # Load Face Detector (YuNet)
        self.face_detector_path = Path(face_detector_model)
        if not self.face_detector_path.exists():
            raise RuntimeError(f"YuNet model not found at '{self.face_detector_path}'")

        self.face_detector = cv2.FaceDetectorYN.create(
            model=str(self.face_detector_path),
            config="",
            input_size=(320, 320),
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

        # Initialize MediaPipe Landmark Detector
        self.face_mesh = None
        self.use_tasks_api = False
        self._init_landmark_detector()

    def _init_landmark_detector(self):
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1, refine_landmarks=True
                )
            elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
                task_path = Path("models/face_landmarker.task")
                if task_path.exists():
                    base_opts = mp.tasks.BaseOptions(model_asset_path=str(task_path))
                    opts = mp.tasks.vision.FaceLandmarkerOptions(base_options=base_opts, num_faces=1)
                    self.face_mesh = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
                    self.use_tasks_api = True
        except Exception as exc:
            logger.warning("MediaPipe FaceMesh initialization skipped: %s", exc)

    def extract_landmarks(self, rgb_image: np.ndarray) -> Optional[List[dict]]:
        if self.face_mesh is None:
            return None
        try:
            raw_landmarks = None
            if self.use_tasks_api:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                res = self.face_mesh.detect(mp_img)
                if res and res.face_landmarks:
                    raw_landmarks = res.face_landmarks[0]
            else:
                res = self.face_mesh.process(rgb_image)
                if res and res.multi_face_landmarks:
                    raw_landmarks = res.multi_face_landmarks[0].landmark

            if not raw_landmarks:
                return None

            landmarks = []
            for idx, l_type in SELECTED_LANDMARK_INDICES:
                if idx < len(raw_landmarks):
                    lm = raw_landmarks[idx]
                    landmarks.append({
                        "x": round(float(lm.x), 4),
                        "y": round(float(lm.y), 4),
                        "type": l_type,
                    })
            return landmarks
        except Exception:
            return None

    def predict_video(
        self,
        video_path: str,
        frames_to_check: int = 15,
        max_edge: int = 720,
        max_duration_sec: int = 300,
    ) -> Tuple[Optional[str], Optional[float], List[float], Optional[dict]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "VIDEO_OPEN_FAILED", None, [], None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            return "VIDEO_OPEN_FAILED", None, [], None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and not np.isnan(fps) and fps > 0:
            if (total_frames / fps) > max_duration_sec:
                cap.release()
                return "VIDEO_TOO_LONG", None, [], None

        frame_interval = max(1, total_frames // frames_to_check)
        face_crops: List[Image.Image] = []
        frame_index = 0
        target_next = 0

        rep_frame: Optional[np.ndarray] = None
        rep_box: Optional[Tuple[float, float, float, float]] = None
        max_face_area: float = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index == target_next:
                h, w = frame.shape[:2]
                curr_max = max(h, w)
                if curr_max > max_edge:
                    scale = max_edge / curr_max
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    new_w, new_h = w, h

                self.face_detector.setInputSize((new_w, new_h))
                _, faces = self.face_detector.detect(frame)

                usable_faces = []
                if faces is not None and len(faces) > 0:
                    for f in faces:
                        fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                        if fw >= self.min_face_size and fh >= self.min_face_size:
                            usable_faces.append((fx, fy, fw, fh))

                if usable_faces:
                    usable_faces.sort(key=lambda b: b[2] * b[3], reverse=True)
                    x, y, w_box, h_box = usable_faces[0]

                    face_area = w_box * h_box
                    if face_area > max_face_area:
                        max_face_area = face_area
                        rep_frame = frame.copy()
                        rep_box = (x, y, w_box, h_box)

                    mx = w_box * self.crop_margin
                    my = h_box * self.crop_margin
                    x1 = max(0, int(round(x - mx)))
                    y1 = max(0, int(round(y - my)))
                    x2 = min(new_w, int(round(x + w_box + mx)))
                    y2 = min(new_h, int(round(y + h_box + my)))

                    if x2 > x1 and y2 > y1:
                        rgb = frame[:, :, ::-1]
                        crop = rgb[y1:y2, x1:x2]
                        if crop.size > 0:
                            face_crops.append(Image.fromarray(crop))

                target_next += frame_interval
                if len(face_crops) >= frames_to_check:
                    break

            frame_index += 1

        cap.release()

        if not face_crops:
            return "NO_FACE_DETECTED", None, [], None

        try:
            with torch.no_grad():
                inputs = self.image_processor(images=face_crops, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                fake_scores = probs[:, self.fake_class_id].tolist()
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "CUDA_OOM", None, [], None
        except Exception:
            return "INFERENCE_FAILED", None, [], None

        avg_score = float(np.mean(fake_scores))
        result = "FAKE" if avg_score > 0.5 else "REAL"
        confidence = avg_score if result == "FAKE" else 1.0 - avg_score

        # Generate face scan overlay
        face_scan: Optional[dict] = None
        if rep_frame is not None and rep_box is not None:
            try:
                rep_h, rep_w = rep_frame.shape[:2]
                landmarks = self.extract_landmarks(rep_frame[:, :, ::-1])
                if landmarks:
                    bx, by, bw, bh = rep_box
                    norm_box = {
                        "x": round(float(bx) / rep_w, 4),
                        "y": round(float(by) / rep_h, 4),
                        "w": round(float(bw) / rep_w, 4),
                        "h": round(float(bh) / rep_h, 4),
                    }
                    max_disp = max(rep_h, rep_w)
                    if max_disp > 480:
                        s = 480.0 / max_disp
                        disp = cv2.resize(rep_frame, (int(rep_w * s), int(rep_h * s)), interpolation=cv2.INTER_AREA)
                    else:
                        disp = rep_frame
                    ok, buf = cv2.imencode(".jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        b64 = base64.b64encode(buf).decode("utf-8")
                        face_scan = {
                            "snapshot": f"data:image/jpeg;base64,{b64}",
                            "box": norm_box,
                            "landmarks": landmarks,
                        }
            except Exception:
                face_scan = None

        return result, confidence, fake_scores, face_scan
