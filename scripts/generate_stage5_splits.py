"""
scripts/generate_stage5_splits.py

Generates evaluation/STAGE_5_SPLITS.md with complete tracking of:
- Calibration / Validation Split (25 REAL + 25 FAKE = 50 videos)
- Final Test Split (25 REAL + 25 FAKE = 50 videos)
"""

import json
from pathlib import Path

manifest_path = Path("D:/benchmark_dataset_dfdc/manifest.json")
m = json.load(open(manifest_path, "r", encoding="utf-8"))
reals = [v for v in m["videos"] if v["ground_truth"] == "REAL"]
fakes = [v for v in m["videos"] if v["ground_truth"] == "FAKE"]

cal_videos = reals[:25] + fakes[:25]
test_videos = reals[25:50] + fakes[25:50]

lines = [
    "# STAGE 5 — BENCHMARK SPLITS SPECIFICATION",
    "",
    "> [!IMPORTANT]",
    "> **DISJOINT SPLIT INTEGRITY GUARANTEE:**",
    "> The second benchmark dataset (DeepFake Detection Challenge, DFDC) is partitioned into two strictly disjoint subsets: **Calibration / Validation ($N=50$)** and **Final Test ($N=50$)**.",
    "> Zero video overlap exists between Calibration and Final Test.",
    "> The 100 Google/Jigsaw DFD benchmark videos from Stage 4B remain completely outside both new splits.",
    "",
    "---",
    "",
    "## 1. Dataset Overview",
    "- **Benchmark Name**: DeepFake Detection Challenge (DFDC) Balanced Subset",
    "- **Source Repository**: `gonnerthetooner/DFDC-extracted-full` (`dfdc_train_part_00/dfdc_train_part_0/`)",
    "- **Official Metadata**: Extracted directly from official `metadata.json`",
    "- **Total Videos**: 100 (50 REAL, 50 FAKE)",
    "- **Storage Location**: `D:/benchmark_dataset_dfdc/`",
    "",
    "---",
    "",
    "## 2. Calibration / Validation Subset ($N = 50$ Videos)",
    "Used strictly for fitting temperature scaling parameter $T$ and exploratory threshold characterization.",
    "Never used for final reporting.",
    "",
    "| ID | Filename | Ground Truth | Original Source Pairing |",
    "| :--- | :--- | :---: | :--- |"
]

for v in cal_videos:
    orig = v.get("original_meta", {}).get("original", "N/A (Real source)")
    lines.append(f"| `{v['id']}` | `{v['filename']}` | **{v['ground_truth']}** | `{orig}` |")

lines.extend([
    "",
    "---",
    "",
    "## 3. Final Test Subset ($N = 50$ Videos)",
    "Held out completely until calibration parameter $T$ is frozen.",
    "Used for final unbiased comparative evaluation of Baseline ViT and GenD PE_L (raw vs calibrated).",
    "",
    "| ID | Filename | Ground Truth | Original Source Pairing |",
    "| :--- | :--- | :---: | :--- |"
])

for v in test_videos:
    orig = v.get("original_meta", {}).get("original", "N/A (Real source)")
    lines.append(f"| `{v['id']}` | `{v['filename']}` | **{v['ground_truth']}** | `{orig}` |")

out_path = Path("evaluation/STAGE_5_SPLITS.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Generated {out_path} with {len(cal_videos)} Calibration and {len(test_videos)} Test entries.")
