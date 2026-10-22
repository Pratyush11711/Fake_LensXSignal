from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class FaceTrack:
    """Represents a single-subject tracked face state across video frames."""
    x: float
    y: float
    w: float
    h: float
    confidence: float
    consecutive_misses: int = 0
    total_detections: int = 0

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union between two (x, y, w, h) bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = boxA_area + boxB_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


class SingleSubjectTracker:
    """
    Maintains a single-subject face track across video frames using IoU matching,
    exponential bounding-box smoothing, and occlusion tolerance.
    """

    def __init__(
        self,
        iou_threshold: float = 0.35,
        max_misses: int = 3,
        smoothing_alpha: float = 0.7,
        max_displacement_ratio: float = 0.6,
    ):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.alpha = smoothing_alpha
        self.max_displacement_ratio = max_displacement_ratio
        self.active_track: Optional[FaceTrack] = None

    def reset(self):
        self.active_track = None

    def update(
        self, candidate_boxes: List[Tuple[float, float, float, float, float]], frame_shape: Tuple[int, int]
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Updates tracker with candidate detections in the current frame.
        Candidates: list of (x, y, w, h, score).
        Returns the tracked (x, y, w, h) or None if no valid track is active.
        """
        if not candidate_boxes:
            if self.active_track is not None:
                self.active_track.consecutive_misses += 1
                if self.active_track.consecutive_misses > self.max_misses:
                    self.active_track = None
                else:
                    return self.active_track.box
            return None

        # First frame / Re-initialization: Pick largest candidate by area
        if self.active_track is None:
            sorted_candidates = sorted(candidate_boxes, key=lambda b: b[2] * b[3], reverse=True)
            best_x, best_y, best_w, best_h, best_score = sorted_candidates[0]
            self.active_track = FaceTrack(
                x=best_x,
                y=best_y,
                w=best_w,
                h=best_h,
                confidence=best_score,
                consecutive_misses=0,
                total_detections=1,
            )
            return self.active_track.box

        # Match candidates to current track using IoU
        best_iou = 0.0
        best_candidate = None

        prev_cx, prev_cy = self.active_track.center
        frame_h, frame_w = frame_shape[:2]

        for cand in candidate_boxes:
            cx, cy, cw, ch, cscore = cand
            iou = compute_iou(self.active_track.box, (cx, cy, cw, ch))

            # Guard against drastic center jumps across consecutive frames
            curr_cx, curr_cy = cx + cw / 2.0, cy + ch / 2.0
            dist = ((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2) ** 0.5
            max_allowed_dist = max(frame_w, frame_h) * self.max_displacement_ratio

            if iou > best_iou and dist <= max_allowed_dist:
                best_iou = iou
                best_candidate = cand

        if best_candidate is not None and best_iou >= self.iou_threshold:
            # Update track with smoothing
            bx, by, bw, bh, bscore = best_candidate
            self.active_track.x = self.alpha * bx + (1.0 - self.alpha) * self.active_track.x
            self.active_track.y = self.alpha * by + (1.0 - self.alpha) * self.active_track.y
            self.active_track.w = self.alpha * bw + (1.0 - self.alpha) * self.active_track.w
            self.active_track.h = self.alpha * bh + (1.0 - self.alpha) * self.active_track.h
            self.active_track.confidence = bscore
            self.active_track.consecutive_misses = 0
            self.active_track.total_detections += 1
            return self.active_track.box
        else:
            # Handle missed detection
            self.active_track.consecutive_misses += 1
            if self.active_track.consecutive_misses > self.max_misses:
                self.active_track = None
                return None
            return self.active_track.box
