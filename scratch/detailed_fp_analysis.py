"""
scratch/detailed_fp_analysis.py

Runs isolated per-clip diagnostics with full frame-by-frame and logit tracing
on all high-scoring real-world false positive candidates.
"""
import os
import sys
import json
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

device = torch.device("cpu")
engine = LipForensicsVideoInference(
    checkpoint_path="lipforensics_ff.pth",
    yunet_model_path="models/face_detection_yunet_2023mar.onnx",
    device=device,
    frames_per_clip=25,
    max_clips_per_video=4,
)

test_files = [
    {
        "label": "videoplayback (2).mp4 (Uploaded user web video / Duplicate of primary real case)",
        "path": r"C:\Users\hp\Downloads\videoplayback (2).mp4",
        "provenance": "User-verified authentic web video (in-the-wild high anomaly)"
    },
    {
        "label": "download (6).mp4 (Direct web download / In-the-wild real)",
        "path": r"C:\Users\hp\Downloads\download (6).mp4",
        "provenance": "Direct web download (132x168 mobile-compressed authentic web footage)"
    },
    {
        "label": "download (5).mp4 (Direct web download / In-the-wild real)",
        "path": r"C:\Users\hp\Downloads\download (5).mp4",
        "provenance": "Direct web download (136x184 mobile-compressed authentic web footage)"
    },
    {
        "label": "Snapchat-550444171.mp4 (OnePlus smartphone recording / In-the-wild real)",
        "path": r"D:\One Plus\Snapchat-550444171.mp4",
        "provenance": "OnePlus smartphone front camera recording via Snapchat"
    },
    {
        "label": "download (2).mp4 (Direct web download / In-the-wild real)",
        "path": r"C:\Users\hp\Downloads\download (2).mp4",
        "provenance": "Direct web download (124x152 compressed authentic web footage)"
    },
    {
        "label": "videoplayback.mp4 (Primary known real case)",
        "path": r"C:\Users\hp\Downloads\videoplayback.mp4",
        "provenance": "User-verified authentic web video (YouTube / Web stream)"
    }
]

diagnostics = []

for item in test_files:
    filepath = item["path"]
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        continue
    
    filename = Path(filepath).name
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    duration_sec = total_frames / fps if fps > 0 else 0
    file_size_bytes = os.path.getsize(filepath)
    
    # Run production inference
    t0 = time.time()
    res = engine.analyze_video(filepath)
    inference_time = time.time() - t0
    
    fake_score = float(res.get("fake_score", 0.0))
    tier, advisory = get_forensic_assessment_tier(fake_score)
    clip_scores = res.get("clip_scores", [])
    raw_clip_logits = res.get("raw_clip_logits", [])
    valid_face = res.get("valid_face_frames", 0)
    valid_mouth = res.get("valid_mouth_crops", 0)
    mean_logit = res.get("mean_logit", 0.0)
    
    clip_details = []
    for idx in range(len(clip_scores)):
        cs = clip_scores[idx]
        cl = raw_clip_logits[idx] if idx < len(raw_clip_logits) else None
        clip_details.append({
            "clip_index": idx + 1,
            "raw_logit": cl,
            "fake_probability": cs
        })
    
    diag_record = {
        "label": item["label"],
        "filename": filename,
        "filepath": filepath,
        "provenance": item["provenance"],
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 3),
        "duration_sec": round(duration_sec, 2),
        "fps": round(fps, 2),
        "resolution": f"{w}x{h}",
        "width": w,
        "height": h,
        "total_frames": total_frames,
        "valid_face_frames": valid_face,
        "valid_mouth_crops": valid_mouth,
        "clip_count": len(clip_scores),
        "per_clip_raw_logits": raw_clip_logits,
        "per_clip_fake_probabilities": clip_scores,
        "clip_details": clip_details,
        "mean_logit": mean_logit,
        "final_fake_probability": fake_score,
        "production_verdict_legacy": "FAKE" if fake_score >= 0.50 else "REAL",
        "production_tier": tier,
        "advisory_message": advisory,
        "inference_time_sec": round(inference_time, 2)
    }
    diagnostics.append(diag_record)
    
    print(f"\n==========================================")
    print(f"FILE: {filename}")
    print(f"Resolution: {w}x{h}, {fps:.2f}fps, {duration_sec:.2f}s, {file_size_bytes} bytes")
    print(f"Face Frames: {valid_face}, Mouth Crops: {valid_mouth}, Clips: {len(clip_scores)}")
    print(f"Raw Logits: {raw_clip_logits}")
    print(f"Clip Probabilities: {clip_scores}")
    print(f"Mean Logit: {mean_logit}, Final Score: {fake_score:.4f} -> {tier}")
    print(f"==========================================")

with open("scratch/fp_diagnostics.json", "w") as f:
    json.dump(diagnostics, f, indent=2)

print("\nSaved diagnostics to scratch/fp_diagnostics.json")
