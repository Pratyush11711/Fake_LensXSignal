from pathlib import Path
from typing import Optional, Tuple

import cv2
try:
    import mediapipe as mp
except ImportError:
    mp = None
import numpy as np


class FaceAligner:
    """
    Performs 5-point similarity transformation alignment on detected facial regions
    using MediaPipe landmark anchors (left eye, right eye, nose tip, left mouth, right mouth).
    Standardizes crops to target canonical dimensions (default 256x256).
    """

    # Standard canonical 5-point template coordinates for 256x256 output
    # (Left eye, Right eye, Nose tip, Left mouth, Right mouth)
    CANONICAL_5_POINTS_256 = np.array(
        [
            [76.8, 97.28],   # Left eye center
            [179.2, 97.28],  # Right eye center
            [128.0, 143.36], # Nose tip
            [89.6, 192.0],   # Left mouth corner
            [166.4, 192.0],  # Right mouth corner
        ],
        dtype=np.float32,
    )

    def __init__(self, target_size: int = 256, crop_margin: float = 0.25):
        self.target_size = target_size
        self.crop_margin = crop_margin

        # Scale canonical points if target_size != 256
        scale = target_size / 256.0
        self.canonical_points = (self.CANONICAL_5_POINTS_256 * scale).astype(np.float32)

        # Initialize MediaPipe detector
        self.detector = None
        self.use_tasks_api = False
        self._init_mediapipe()

    def _init_mediapipe(self):
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.detector = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1, refine_landmarks=True
                )
            elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
                task_path = Path("models/face_landmarker.task")
                if task_path.exists():
                    base_opts = mp.tasks.BaseOptions(model_asset_path=str(task_path))
                    opts = mp.tasks.vision.FaceLandmarkerOptions(base_options=base_opts, num_faces=1)
                    self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
                    self.use_tasks_api = True
        except Exception:
            self.detector = None

    def _extract_5_points(self, rgb_image: np.ndarray) -> Optional[np.ndarray]:
        """Extracts 5 key points: left eye (33), right eye (263), nose tip (1), left mouth (61), right mouth (291)."""
        if self.detector is None:
            return None

        try:
            raw_landmarks = None
            if self.use_tasks_api:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                res = self.detector.detect(mp_img)
                if res and res.face_landmarks and len(res.face_landmarks) > 0:
                    raw_landmarks = res.face_landmarks[0]
            else:
                res = self.detector.process(rgb_image)
                if res and res.multi_face_landmarks and len(res.multi_face_landmarks) > 0:
                    raw_landmarks = res.multi_face_landmarks[0].landmark

            if not raw_landmarks:
                return None

            h, w = rgb_image.shape[:2]
            key_indices = [33, 263, 1, 61, 291]
            points = []
            for idx in key_indices:
                lm = raw_landmarks[idx]
                points.append([lm.x * w, lm.y * h])

            return np.array(points, dtype=np.float32)
        except Exception:
            return None

    def align_face(
        self, frame: np.ndarray, face_box: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[np.ndarray]:
        """
        Aligns the face in the frame using similarity transform estimation.
        If landmarks are missing, falls back to a padded bounding-box square crop.
        Returns: (target_size, target_size, 3) RGB image.
        """
        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1] if len(frame.shape) == 3 else frame

        # Attempt landmark-based affine alignment
        points = self._extract_5_points(rgb)
        if points is not None and len(points) == 5:
            # Estimate similarity transform using OpenCV estimateAffinePartial2D (Umeyama algorithm)
            matrix, inliers = cv2.estimateAffinePartial2D(points, self.canonical_points, method=cv2.LMEDS)
            if matrix is not None:
                aligned = cv2.warpAffine(
                    rgb,
                    matrix,
                    (self.target_size, self.target_size),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                return aligned

        # Fallback: Bounding box crop with margin and resize
        if face_box is not None:
            bx, by, bw, bh = face_box
            mx, my = bw * self.crop_margin, bh * self.crop_margin
            x1 = max(0, int(round(bx - mx)))
            y1 = max(0, int(round(by - my)))
            x2 = min(w, int(round(bx + bw + mx)))
            y2 = min(h, int(round(by + bh + my)))

            if x2 > x1 and y2 > y1:
                crop = rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    return cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_CUBIC)

        return None
