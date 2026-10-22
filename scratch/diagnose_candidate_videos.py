"""
scratch/diagnose_candidate_videos.py

Diagnoses candidate uploaded videos using the exact production LipForensics pipeline.
"""
import os
import sys
import time
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from src.lipforensics_video_inference import LipForensicsVideoInference
from main import get_forensic_assessment_tier

candidates = [
    r"C:\Users\hp\Downloads\do_it_bro_please.mp4",
    r"C:\Users\hp\Downloads\instead_of_workday_oracle_serv.mp4",
    r"C:\Users\hp\Downloads\videoplayback (2).mp4",
]

device = torch.device("cpu")
engine = LipForensicsVideoInference(
    checkpoint_path="lipforensics_ff.pth",
    yunet_model_path="models/face_detection_yunet_2023mar.onnx",
    device=device,
    frames_per_clip=25,
    max_clips_per_video=4,
)

print("=" * 80)
print("DIAGNOSING CANDIDATE REAL-WORLD VIDEOS")
print("=" * 80)

for v in candidates:
    if not os.path.exists(v):
        continue
    filename = Path(v).name
    print(f"\n--- Analyzing {filename} ---")
    cap = cv2.VideoCapture(v)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    size_bytes = os.path.getsize(v)
    
    t0 = time.time()
    res = engine.analyze_video(v)
    elapsed = time.time() - t0
    
    status = res.get("status")
    fake_score = float(res.get("fake_score", 0.0))
    tier, msg = get_forensic_assessment_tier(fake_score)
    clip_scores = res.get("clip_scores", [])
    raw_logits = res.get("raw_logits", [])
    valid_mouth = res.get("valid_mouth_crops", 0)
    
    print(f"File: {filename}")
    print(f"  Size: {size_bytes} bytes ({size_bytes / (1024*1024):.2f} MB)")
    print(f"  Resolution: {w}x{h}, FPS: {fps:.2f}, Frames: {total_frames}, Duration: {total_frames/fps:.2f}s")
    print(f"  Status: {status}")
    print(f"  Valid Mouth Crops: {valid_mouth}")
    print(f"  Clip Count: {len(clip_scores)}")
    print(f"  Clip Raw Logits: {raw_logits}")
    print(f"  Clip Scores: {clip_scores}")
    print(f"  Final Fake Probability / Score: {fake_score:.4f}")
    print(f"  3-Tier Assessment: {tier}")
    print(f"  Legacy Verdict: {'FAKE' if fake_score >= 0.50 else 'REAL'}")
    print(f"  Elapsed: {elapsed:.2f}s")
