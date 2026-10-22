"""
scripts/download_dfdc_benchmark.py

Acquires a balanced 100-video subset (50 REAL, 50 FAKE) from the DeepFake Detection Challenge (DFDC)
using the official metadata.json to ensure exact label fidelity and pair tracking.
Stores data in D:/benchmark_dataset_dfdc.
"""

import json
import os
import subprocess
import urllib.request
from pathlib import Path

TARGET_DIR = Path("D:/benchmark_dataset_dfdc")
TARGET_DIR.mkdir(parents=True, exist_ok=True)
real_dir = TARGET_DIR / "real"
fake_dir = TARGET_DIR / "fake"
real_dir.mkdir(exist_ok=True)
fake_dir.mkdir(exist_ok=True)

# Fetch official metadata.json
meta_url = "https://huggingface.co/datasets/gonnerthetooner/DFDC-extracted-full/resolve/main/dfdc_train_part_00/dfdc_train_part_0/metadata.json"
print("Fetching official DFDC metadata.json...")
res = urllib.request.urlopen(meta_url)
metadata = json.loads(res.read().decode("utf-8"))
print(f"Loaded {len(metadata)} video entries from official DFDC metadata.")

# Filter and select 50 REAL and 50 FAKE videos
reals = [k for k, v in metadata.items() if v.get("label") == "REAL"][:50]
fakes = [k for k, v in metadata.items() if v.get("label") == "FAKE"][:50]

print(f"Selected {len(reals)} REAL and {len(fakes)} FAKE DFDC videos.")

# Generate curl config for fast parallel downloading
curl_config_path = TARGET_DIR / "curl_urls.txt"
manifest_videos = []

with open(curl_config_path, "w", encoding="utf-8") as f:
    for idx, fname in enumerate(reals, 1):
        dest = real_dir / fname
        url = f"https://huggingface.co/datasets/gonnerthetooner/DFDC-extracted-full/resolve/main/dfdc_train_part_00/dfdc_train_part_0/{fname}"
        f.write(f'url = "{url}"\n')
        f.write(f'output = "{str(dest).replace(chr(92), "/")}"\n')
        manifest_videos.append({
            "id": f"dfdc_real_{idx:03d}",
            "filename": fname,
            "local_path": str(dest),
            "ground_truth": "REAL",
            "ground_truth_label": 0,
            "original_meta": metadata[fname],
            "dataset_origin": "DFDC (DeepFake Detection Challenge, Facebook AI / AWS / Microsoft)",
            "file_size_bytes": 0
        })

    for idx, fname in enumerate(fakes, 1):
        dest = fake_dir / fname
        url = f"https://huggingface.co/datasets/gonnerthetooner/DFDC-extracted-full/resolve/main/dfdc_train_part_00/dfdc_train_part_0/{fname}"
        f.write(f'url = "{url}"\n')
        f.write(f'output = "{str(dest).replace(chr(92), "/")}"\n')
        manifest_videos.append({
            "id": f"dfdc_fake_{idx:03d}",
            "filename": fname,
            "local_path": str(dest),
            "ground_truth": "FAKE",
            "ground_truth_label": 1,
            "original_meta": metadata[fname],
            "dataset_origin": "DFDC (DeepFake Detection Challenge, Facebook AI / AWS / Microsoft)",
            "file_size_bytes": 0
        })

print("Executing parallel curl download...")
cmd = ["curl.exe", "-L", "--parallel", "--parallel-max", "16", "--config", str(curl_config_path)]
try:
    subprocess.run(cmd, check=True)
except Exception as e:
    print(f"Curl finished with note: {e}")

# Verify file sizes
valid_count = 0
for item in manifest_videos:
    p = Path(item["local_path"])
    if p.exists() and p.stat().st_size > 0:
        item["file_size_bytes"] = p.stat().st_size
        valid_count += 1

manifest = {
    "benchmark_name": "DFDC_Balanced_Benchmark_100",
    "dataset_source": "DeepFake Detection Challenge (DFDC, Facebook AI / AWS / Microsoft)",
    "official_repository": "gonnerthetooner/DFDC-extracted-full",
    "total_videos": len(manifest_videos),
    "total_real": len(reals),
    "total_fake": len(fakes),
    "valid_downloaded_videos": valid_count,
    "storage_directory": str(TARGET_DIR),
    "videos": manifest_videos
}

manifest_path = TARGET_DIR / "manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"\n[DONE] {valid_count}/{len(manifest_videos)} DFDC videos downloaded to {TARGET_DIR}.")
print(f"Manifest written to {manifest_path}.")
