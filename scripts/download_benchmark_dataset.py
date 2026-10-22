"""
scripts/download_benchmark_dataset.py

Ultra-fast parallel download of 100 benchmark videos (50 REAL, 50 FAKE)
using curl.exe --parallel into D:/benchmark_dataset with manifest.json.
"""

import json
import os
import subprocess
from pathlib import Path
from huggingface_hub import HfApi

TARGET_DIR = Path("D:/benchmark_dataset")
TARGET_DIR.mkdir(parents=True, exist_ok=True)
real_dir = TARGET_DIR / "real"
fake_dir = TARGET_DIR / "fake"
real_dir.mkdir(exist_ok=True)
fake_dir.mkdir(exist_ok=True)

api = HfApi()

print("Fetching file list from repository...")
real_files = [f.path for f in api.list_repo_tree("angads24/deepfake-video", path_in_repo="dataset/real", repo_type="dataset") if f.path.endswith(".mp4")][:50]
fake_files = [f.path for f in api.list_repo_tree("angads24/deepfake-video", path_in_repo="dataset/fake", repo_type="dataset") if f.path.endswith(".mp4")][:50]

print(f"Selected {len(real_files)} REAL and {len(fake_files)} FAKE benchmark videos.")

# Generate curl config file for parallel download
curl_config_path = TARGET_DIR / "curl_urls.txt"
manifest_videos = []

with open(curl_config_path, "w", encoding="utf-8") as f:
    for idx, p in enumerate(real_files, 1):
        fname = Path(p).name
        dest = real_dir / fname
        url = f"https://huggingface.co/datasets/angads24/deepfake-video/resolve/main/{p}"
        f.write(f'url = "{url}"\n')
        f.write(f'output = "{str(dest).replace(chr(92), "/")}"\n')
        manifest_videos.append({
            "id": f"real_{idx:03d}",
            "filename": fname,
            "local_path": str(dest),
            "ground_truth": "REAL",
            "ground_truth_label": 0,
            "source": "Google/Jigsaw DeepfakeDetection (FaceForensics++ / DeepfakeBench Component)",
            "file_size_bytes": 0
        })

    for idx, p in enumerate(fake_files, 1):
        fname = Path(p).name
        dest = fake_dir / fname
        url = f"https://huggingface.co/datasets/angads24/deepfake-video/resolve/main/{p}"
        f.write(f'url = "{url}"\n')
        f.write(f'output = "{str(dest).replace(chr(92), "/")}"\n')
        manifest_videos.append({
            "id": f"fake_{idx:03d}",
            "filename": fname,
            "local_path": str(dest),
            "ground_truth": "FAKE",
            "ground_truth_label": 1,
            "source": "Google/Jigsaw DeepfakeDetection (FaceForensics++ / DeepfakeBench Component)",
            "file_size_bytes": 0
        })

print("Executing curl.exe --parallel download...")
cmd = ["curl.exe", "-L", "--parallel", "--parallel-max", "16", "--config", str(curl_config_path)]
subprocess.run(cmd, check=True)

# Update manifest with actual file sizes
for item in manifest_videos:
    p = Path(item["local_path"])
    if p.exists():
        item["file_size_bytes"] = p.stat().st_size

manifest = {
    "benchmark_name": "Standard_Deepfake_Benchmark_Balanced_100",
    "dataset_source": "Google/Jigsaw DeepfakeDetection (FaceForensics++ / DeepfakeBench Component)",
    "official_repository": "angads24/deepfake-video",
    "total_videos": len(manifest_videos),
    "total_real": len(real_files),
    "total_fake": len(fake_files),
    "storage_directory": str(TARGET_DIR),
    "videos": manifest_videos
}

manifest_path = TARGET_DIR / "manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"\n[DONE] All {len(manifest_videos)} benchmark videos downloaded to {TARGET_DIR}.")
print(f"Manifest written to {manifest_path}.")
