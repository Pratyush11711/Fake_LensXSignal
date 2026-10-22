"""
scratch/screen_real_videos.py

Screens non-benchmark real-world video files on the system to identify 20-30 diverse, authentic videos with valid face and mouth tracks for STAGE 6J.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
from src.lipforensics_video_inference import LipForensicsVideoInference

device = torch.device("cpu")
engine = LipForensicsVideoInference(
    checkpoint_path="lipforensics_ff.pth",
    yunet_model_path="models/face_detection_yunet_2023mar.onnx",
    device=device,
    frames_per_clip=25,
    max_clips_per_video=4,
)

candidate_paths = []

# 1. Downloads / Web videos
dl_dir = r"C:\Users\hp\Downloads"
if os.path.exists(dl_dir):
    for f in os.listdir(dl_dir):
        p = os.path.join(dl_dir, f)
        if os.path.isfile(p) and f.lower().endswith((".mp4", ".mov", ".webm", ".avi")):
            if "videoplayback" in f.lower():
                candidate_paths.append((p, "User-Verified Web In-The-Wild Video (YouTube / Web Stream)"))
            elif "whatsapp" in f.lower():
                candidate_paths.append((p, "WhatsApp Mobile Messaging Compressed Video"))
            elif "instead_of" in f.lower() or "do_it" in f.lower():
                candidate_paths.append((p, "Downloaded Web / Social Media Recording"))
            elif f.startswith("download"):
                candidate_paths.append((p, "Direct Web Download Clip"))

# 2. One Plus Phone / Snapchat / WhatsApp recordings
op_dir = r"D:\One Plus"
if os.path.exists(op_dir):
    for f in sorted(os.listdir(op_dir)):
        p = os.path.join(op_dir, f)
        if os.path.isfile(p) and f.lower().endswith((".mp4", ".mov", ".avi")):
            if f.startswith("Snapchat"):
                candidate_paths.append((p, "OnePlus Smartphone Front Camera (Snapchat Recording)"))
            elif f.startswith("VID"):
                candidate_paths.append((p, "OnePlus Smartphone Messaging Video (WhatsApp Media)"))
            elif f.startswith("InShot") or f.startswith("202"):
                candidate_paths.append((p, "OnePlus Smartphone Native Camera / InShot Video"))

print(f"Total candidate files to scan: {len(candidate_paths)}")

valid_candidates = []

for p, prov in candidate_paths:
    if not os.path.exists(p):
        continue
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total_frames / fps if fps > 0 else 0.0

    if total_frames < 25:
        cap.release()
        continue

    # Quick scan first 100 frames
    mouth_count = 0
    read_count = 0
    while read_count < min(total_frames, 120):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        read_count += 1
        crop, _ = engine._extract_mouth_crop(frame)
        if crop is not None:
            mouth_count += 1
    cap.release()

    if mouth_count >= 25: # Has at least one valid 25-frame clip
        sz_mb = os.path.getsize(p) / (1024 * 1024)
        valid_candidates.append({
            "path": p,
            "filename": os.path.basename(p),
            "provenance": prov,
            "resolution": f"{w}x{h}",
            "fps": round(fps, 1),
            "duration_sec": round(dur, 2),
            "total_frames": total_frames,
            "size_mb": round(sz_mb, 2),
            "sample_mouth_count": mouth_count,
        })

print(f"\nValid real-world candidates with mouth tracks found: {len(valid_candidates)}\n")
for idx, c in enumerate(valid_candidates[:30], 1):
    print(f"[{idx:2d}] {c['filename']:40s} | {c['resolution']:9s} | {c['duration_sec']:6.2f}s | {c['size_mb']:5.2f}MB | {c['provenance']}")
