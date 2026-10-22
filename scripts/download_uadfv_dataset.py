"""
scripts/download_uadfv_dataset.py

Downloads the canonical UADFV benchmark dataset (49 REAL, 49 FAKE = 98 videos)
from belkhir-nacim/deepfake-videos into D:/benchmark_dataset_uadfv with manifest.json.
"""

import json
import os
import subprocess
from pathlib import Path
from huggingface_hub import HfApi

TARGET_DIR = Path("D:/benchmark_dataset_uadfv")
TARGET_DIR.mkdir(parents=True, exist_ok=True)
real_dir = TARGET_DIR / "real"
fake_dir = TARGET_DIR / "fake"
real_dir.mkdir(exist_ok=True)
fake_dir.mkdir(exist_ok=True)

api = HfApi()

print("Fetching UADFV video list from repository (belkhir-nacim/deepfake-videos)...")
real_files = [f.path for f in api.list_repo_tree("belkhir-nacim/deepfake-videos", path_in_repo="videos/UADFV/real", repo_type="dataset") if f.path.endswith(".mp4")]
fake_files = [f.path for f in api.list_repo_tree("belkhir-nacim/deepfake-videos", path_in_repo="videos/UADFV/fake", repo_type="dataset") if f.path.endswith(".mp4")]

# Sort deterministically
real_files.sort()
fake_files.sort()

print(f"Found {len(real_files)} REAL and {len(fake_files)} FAKE UADFV benchmark videos.")

# Generate curl config file for fast parallel download
curl_config_path = TARGET_DIR / "curl_urls.txt"
manifest_videos = []

with open(curl_config_path, "w", encoding="utf-8") as f:
    for idx, p in enumerate(real_files, 1):
        fname = Path(p).name
        dest = real_dir / fname
        url = f"https://huggingface.co/datasets/belkhir-nacim/deepfake-videos/resolve/main/{p}"
        f.write(f'url = "{url}"\n')
        f.write(f'output = "{str(dest).replace(chr(92), "/")}"\n')
        manifest_videos.append({
            "id": f"uadfv_real_{idx:03d}",
            "filename": fname,
            "local_path": str(dest),
            "ground_truth": "REAL",
            "ground_truth_label": 0,
            "dataset_origin": "UADFV (Yang et al., 2019, SUNY Albany)",
            "file_size_bytes": 0
        })

    for idx, p in enumerate(fake_files, 1):
        fname = Path(p).name
        dest = fake_dir / fname
        url = f"https://huggingface.co/datasets/belkhir-nacim/deepfake-videos/resolve/main/{p}"
        f.write(f'url = "{url}"\n')
        f.write(f'output = "{str(dest).replace(chr(92), "/")}"\n')
        manifest_videos.append({
            "id": f"uadfv_fake_{idx:03d}",
            "filename": fname,
            "local_path": str(dest),
            "ground_truth": "FAKE",
            "ground_truth_label": 1,
            "dataset_origin": "UADFV (Yang et al., 2019, SUNY Albany)",
            "file_size_bytes": 0
        })

print(f"Executing curl.exe --parallel download for {len(manifest_videos)} videos...")
cmd = ["curl.exe", "-L", "--parallel", "--parallel-max", "16", "--config", str(curl_config_path)]
try:
    subprocess.run(cmd, check=True)
except Exception as e:
    print(f"Parallel curl finished with code/warning: {e}. Verifying files...")

# Verify and update manifest with actual file sizes
valid_count = 0
for item in manifest_videos:
    p = Path(item["local_path"])
    if p.exists() and p.stat().st_size > 0:
        item["file_size_bytes"] = p.stat().st_size
        valid_count += 1

manifest = {
    "benchmark_name": "UADFV_Standard_Benchmark_98",
    "dataset_source": "UADFV (Yang et al., 2019, SUNY Albany)",
    "official_repository": "belkhir-nacim/deepfake-videos",
    "total_videos": len(manifest_videos),
    "total_real": len(real_files),
    "total_fake": len(fake_files),
    "valid_downloaded_videos": valid_count,
    "storage_directory": str(TARGET_DIR),
    "videos": manifest_videos
}

manifest_path = TARGET_DIR / "manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"\n[DONE] {valid_count}/{len(manifest_videos)} UADFV videos downloaded to {TARGET_DIR}.")
print(f"Manifest written to {manifest_path}.")
