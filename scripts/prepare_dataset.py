"""
Dataset Preparation Script for FaceForensics++ and Celeb-DF v2.
Extracts aligned face sequences and generates leak-free identity splits (train/val/test).
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm

from src.alignment.face_aligner import FaceAligner
from src.detection.face_tracker import SingleSubjectTracker
from src.detection.yunet_detector import YuNetFaceDetector


def extract_video_sequence(
    video_path: Path,
    output_dir: Path,
    detector: YuNetFaceDetector,
    tracker: SingleSubjectTracker,
    aligner: FaceAligner,
    max_frames: int = 32,
) -> List[str]:
    """Extracts aligned canonical face crops for a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    tracker.reset()

    interval = max(1, total_frames // max_frames)
    frame_idx = 0
    target_next = 0
    saved_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == target_next:
            candidates = detector.detect(frame)
            tracked_box = tracker.update(candidates, frame.shape)

            if tracked_box is not None:
                aligned = aligner.align_face(frame, face_box=tracked_box)
                if aligned is not None:
                    out_path = output_dir / f"frame_{len(saved_paths):04d}.jpg"
                    cv2.imwrite(str(out_path), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
                    saved_paths.append(str(out_path))

            target_next += interval
            if len(saved_paths) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved_paths


def process_dataset(
    raw_dir: Path,
    processed_dir: Path,
    metadata_dir: Path,
    dataset_type: str = "faceforensics",
    split_ratio=(0.8, 0.1, 0.1),
    seed: int = 42,
):
    random.seed(seed)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {raw_dir} for {dataset_type} videos...")
    video_files = list(raw_dir.rglob("*.mp4")) + list(raw_dir.rglob("*.avi"))
    if not video_files:
        print(f"\n[!] No video files found in {raw_dir}.")
        print(f"Please place your dataset videos into: {raw_dir.resolve()}")
        return

    detector = YuNetFaceDetector()
    tracker = SingleSubjectTracker()
    aligner = FaceAligner()

    metadata_records: List[Dict] = []
    print(f"Found {len(video_files)} videos. Extracting aligned sequences...")

    for v_path in tqdm(video_files):
        # Extract source identity to prevent identity leakage
        video_id = v_path.stem
        source_id = video_id.split("_")[0] if "_" in video_id else video_id

        # Determine label (Real=0, Fake=1)
        is_fake = "manipulated" in str(v_path).lower() or "fake" in str(v_path).lower() or "synthesis" in str(v_path).lower()
        label = 1 if is_fake else 0

        seq_dir = processed_dir / video_id
        frame_paths = extract_video_sequence(v_path, seq_dir, detector, tracker, aligner)

        if len(frame_paths) >= 8:
            metadata_records.append({
                "video_id": video_id,
                "source_identity": source_id,
                "label": label,
                "frame_count": len(frame_paths),
                "frame_paths": frame_paths,
            })

    # Group by source identity to prevent data leakage
    identity_map: Dict[str, List[Dict]] = {}
    for rec in metadata_records:
        ident = rec["source_identity"]
        identity_map.setdefault(ident, []).append(rec)

    all_identities = list(identity_map.keys())
    random.shuffle(all_identities)

    n_total = len(all_identities)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    train_ids = set(all_identities[:n_train])
    val_ids = set(all_identities[n_train:n_train + n_val])
    test_ids = set(all_identities[n_train + n_val:])

    train_records = [r for ident in train_ids for r in identity_map[ident]]
    val_records = [r for ident in val_ids for r in identity_map[ident]]
    test_records = [r for ident in test_ids for r in identity_map[ident]]

    for name, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        out_file = metadata_dir / f"{name}_metadata.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(records)} records to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FaceForensics++ or Celeb-DF v2 dataset")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--dataset", type=str, default="faceforensics", choices=["faceforensics", "celeb-df"])
    args = parser.parse_args()

    process_dataset(args.raw_dir, args.processed_dir, args.metadata_dir, dataset_type=args.dataset)
