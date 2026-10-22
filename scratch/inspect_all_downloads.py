"""
scratch/inspect_all_downloads.py
"""
import os
import sys
import glob
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
from src.lipforensics_video_inference import LipForensicsVideoInference
from main import get_forensic_assessment_tier

engine = LipForensicsVideoInference(
    checkpoint_path="lipforensics_ff.pth",
    yunet_model_path="models/face_detection_yunet_2023mar.onnx",
    device=torch.device("cpu"),
    frames_per_clip=25,
    max_clips_per_video=4,
)

downloads = r"C:\Users\hp\Downloads"
exts = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm"]
files = []
for ext in exts:
    files.extend(glob.glob(os.path.join(downloads, ext)))
files.sort(key=os.path.getmtime, reverse=True)

print("=" * 80)
print("INSPECTING DOWNLOADS VIDEOS")
print("=" * 80)

for f in files[:8]:
    name = Path(f).name
    cap = cv2.VideoCapture(f)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    dur = frames / fps if fps > 0 else 0
    size_mb = os.path.getsize(f) / (1024 * 1024)
    
    print(f"\nEvaluating {name} ({w}x{h}, {fps:.1f}fps, {frames} frames, {dur:.1f}s, {size_mb:.2f}MB)")
    res = engine.analyze_video(f)
    print("  Status:", res.get("status"))
    print("  Fake Score:", res.get("fake_score"))
    print("  Valid Mouth Crops:", res.get("valid_mouth_crops"))
    print("  Clip Scores:", res.get("clip_scores"))
    print("  Raw Logits:", res.get("raw_logits"))
    if res.get("fake_score") is not None:
        score = float(res["fake_score"])
        tier, msg = get_forensic_assessment_tier(score)
        print("  3-Tier Assessment:", tier)
        print("  Legacy Verdict:", "FAKE" if score >= 0.50 else "REAL")
