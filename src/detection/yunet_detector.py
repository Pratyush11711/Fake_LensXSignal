from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


class YuNetFaceDetector:
    """
    Wrapper around OpenCV YuNet FaceDetectorYN.
    """

    def __init__(
        self,
        model_path: str = "models/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        min_face_size: int = 40,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"YuNet model not found at {self.model_path}")

        self.min_face_size = min_face_size
        self.detector = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Detects faces in frame.
        Returns list of tuples: (x, y, w, h, score) for faces meeting min_face_size.
        """
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)

        usable_faces = []
        if faces is not None and len(faces) > 0:
            for face in faces:
                fx, fy, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
                score = float(face[-1]) if len(face) > 14 else 1.0
                if fw >= self.min_face_size and fh >= self.min_face_size:
                    usable_faces.append((fx, fy, fw, fh, score))

        return usable_faces
