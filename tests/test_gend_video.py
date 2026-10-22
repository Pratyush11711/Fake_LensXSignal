"""
tests/test_gend_video.py

Stage 2 Standalone Test: Full Video Inference Pipeline with GenD (yermandy/GenD_PE_L).
Discovers diagnostic videos, runs end-to-end frame extraction & GenD inference,
computes temporal aggregations, and compares against read-only baseline snapshot.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gend_video_inference import GenDVideoInferencePipeline


def discover_diagnostic_videos() -> List[str]:
    """
    Dynamically discovers all available diagnostic video files across known workspace paths.
    """
    search_locations = [
        Path("D:/"),
        Path("tests"),
        Path("Uploaded_Files"),
        Path(os.path.expanduser("~/Downloads")),
        Path(os.path.expanduser("~/Videos")),
        Path(os.path.expanduser("~/Desktop")),
    ]

    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    discovered_paths = set()

    for loc in search_locations:
        if loc.exists():
            try:
                for file_path in loc.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                        if not file_path.name.startswith((".", "~$")):
                            discovered_paths.add(str(file_path.resolve()))
            except Exception as e:
                print(f"  [Notice] Could not search location {loc}: {e}", flush=True)

    # Also include any videos mentioned in baseline_snapshot.json if they exist on disk
    snapshot_path = PROJECT_ROOT / "baseline_snapshot.json"
    if snapshot_path.exists():
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                snapshot_data = json.load(f)
            for v in snapshot_data.get("videos", []):
                p = v.get("path")
                if p and os.path.exists(p):
                    discovered_paths.add(str(Path(p).resolve()))
        except Exception:
            pass

    return sorted(list(discovered_paths))


def load_baseline_snapshot() -> Dict[str, Dict]:
    """Loads baseline snapshot data strictly in read-only mode."""
    snapshot_path = PROJECT_ROOT / "baseline_snapshot.json"
    if not snapshot_path.exists():
        return {}

    try:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = {}
        for item in data.get("videos", []):
            fname = item.get("filename")
            if fname:
                mapping[fname] = item
        return mapping
    except Exception as exc:
        print(f"  [Warning] Failed to load baseline_snapshot.json: {exc}", flush=True)
        return {}


def run_stage2_test():
    print("=" * 80, flush=True)
    print("STAGE 2 - GEN D VIDEO INFERENCE PIPELINE VERIFICATION", flush=True)
    print("=" * 80, flush=True)

    # 1. Discover Videos
    discovered_videos = discover_diagnostic_videos()
    print(f"\n[1] Discovered {len(discovered_videos)} Diagnostic Videos:", flush=True)
    for idx, v in enumerate(discovered_videos, 1):
        print(f"  {idx:2d}. {Path(v).name} ({v})", flush=True)

    if not discovered_videos:
        print("  [ERROR] No diagnostic videos found for verification.", flush=True)
        return

    # 2. Initialize Pipeline (Singleton weights loading)
    print("\n[2] Initializing GenD Video Pipeline (yermandy/GenD_PE_L)...", flush=True)
    init_start = time.perf_counter()
    pipeline = GenDVideoInferencePipeline()
    init_time = time.perf_counter() - init_start
    print(f"  [OK] Pipeline initialized in {init_time:.2f}s (Device: {pipeline.device})", flush=True)

    # 3. Process Each Video Sequentially
    print("\n[3] Executing GenD Video Analysis Across All Diagnostic Videos:", flush=True)
    results: List[Dict] = []
    total_frames_inferred = 0
    total_inference_time = 0.0

    for idx, video_path in enumerate(discovered_videos, 1):
        fname = Path(video_path).name
        print(f"\n  --- [{idx}/{len(discovered_videos)}] Processing: {fname} ---", flush=True)

        try:
            res = pipeline.analyze_video(video_path, frames_to_sample=15)
            res["filename"] = fname
            res["path"] = video_path
            results.append(res)

            status = res.get("status")
            sampled = res.get("sampled_frames", 0)
            faces = res.get("face_detected_frames", 0)
            inferred = res.get("frames_inferred_by_gend", 0)
            mean_p = res.get("gend_mean_fake_probability")
            median_p = res.get("gend_median_fake_probability")
            trimmed_p = res.get("gend_trimmed_mean_fake_probability")
            p_std = res.get("probability_std")
            quality = res.get("evidence_quality", "UNKNOWN")
            elapsed = res.get("elapsed_seconds", 0.0)

            total_frames_inferred += inferred
            total_inference_time += elapsed

            if status == "ANALYZED":
                print(f"      Status: {status} | Quality: {quality} | Elapsed: {elapsed:.2f}s", flush=True)
                print(f"      Frames: {sampled} sampled -> {faces} faces -> {inferred} inferred", flush=True)
                print(f"      GenD Probabilities: Mean={mean_p:.4f}, Median={median_p:.4f}, Trimmed={trimmed_p:.4f}, Std={p_std if p_std is not None else 0.0:.4f}", flush=True)
            elif status == "INSUFFICIENT_EVIDENCE":
                print(f"      Status: {status} | Quality: {quality} | Elapsed: {elapsed:.2f}s", flush=True)
                print(f"      Frames: {sampled} sampled -> {faces} faces (Face coverage too low)", flush=True)
                print(f"      GenD Probabilities: Mean={mean_p}, Median={median_p}, Trimmed={trimmed_p}", flush=True)
            elif status == "NO_FACE_DETECTED":
                print(f"      Status: {status} | Quality: {quality} | Elapsed: {elapsed:.2f}s", flush=True)
                print(f"      Frames: {sampled} sampled -> 0 faces detected", flush=True)
            else:
                print(f"      Status: {status} | Error: {res.get('error')} | Detail: {res.get('detail')}", flush=True)

        except Exception as exc:
            print(f"      [EXCEPTION] Failed processing {fname}: {exc}", flush=True)
            results.append({
                "filename": fname,
                "path": video_path,
                "status": "ERROR",
                "error": "EXCEPTION",
                "detail": str(exc),
                "elapsed_seconds": 0.0,
            })

    # 4. Load Baseline Snapshot for Read-Only Comparison
    print("\n" + "=" * 80, flush=True)
    print("[4] READ-ONLY COMPARISON: BASELINE vs GEND", flush=True)
    print("=" * 80, flush=True)

    baseline_data = load_baseline_snapshot()
    header = f"{'Filename':<35} | {'Baseline Score':<14} | {'GenD Mean':<10} | {'GenD Med':<10} | {'GenD Trim':<10} | {'Diff':<8} | {'Quality':<12} | {'Status':<15}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for r in results:
        fname = r["filename"]
        b_info = baseline_data.get(fname, {})
        b_score = b_info.get("existing_video_level_score")
        b_score_str = f"{b_score:.4f}" if b_score is not None else "N/A"

        g_mean = r.get("gend_mean_fake_probability")
        g_med = r.get("gend_median_fake_probability")
        g_trim = r.get("gend_trimmed_mean_fake_probability")

        g_mean_str = f"{g_mean:.4f}" if g_mean is not None else "N/A"
        g_med_str = f"{g_med:.4f}" if g_med is not None else "N/A"
        g_trim_str = f"{g_trim:.4f}" if g_trim is not None else "N/A"

        if b_score is not None and g_trim is not None:
            diff_str = f"{g_trim - b_score:+.4f}"
        else:
            diff_str = "N/A"

        quality = r.get("evidence_quality", "N/A")
        status = r.get("status", "N/A")

        # Truncate fname if needed
        disp_name = fname if len(fname) <= 35 else fname[:32] + "..."
        print(f"{disp_name:<35} | {b_score_str:<14} | {g_mean_str:<10} | {g_med_str:<10} | {g_trim_str:<10} | {diff_str:<8} | {quality:<12} | {status:<15}", flush=True)

    # 5. Aggregate Performance Statistics
    analyzed_count = sum(1 for r in results if r.get("status") == "ANALYZED")
    insufficient_count = sum(1 for r in results if r.get("status") == "INSUFFICIENT_EVIDENCE")
    no_face_count = sum(1 for r in results if r.get("status") == "NO_FACE_DETECTED")
    error_count = sum(1 for r in results if r.get("status") == "ERROR")

    avg_time_per_video = total_inference_time / max(1, len(results))
    avg_time_per_frame = (total_inference_time / max(1, total_frames_inferred)) if total_frames_inferred > 0 else 0.0

    print("\n" + "=" * 80, flush=True)
    print("STAGE 2 VERIFICATION SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"  * Total Videos Discovered:          {len(discovered_videos)}", flush=True)
    print(f"  * Successfully Analyzed (High/Med): {analyzed_count}", flush=True)
    print(f"  * Insufficient Evidence (Low):      {insufficient_count}", flush=True)
    print(f"  * No Faces Detected:                {no_face_count}", flush=True)
    print(f"  * Pipeline Errors:                  {error_count}", flush=True)
    print(f"  * Total Inferred Frames:            {total_frames_inferred}", flush=True)
    print(f"  * Avg Processing Time / Video:      {avg_time_per_video:.2f}s", flush=True)
    print(f"  * Avg Inference Time / Face:        {avg_time_per_frame:.4f}s", flush=True)
    print("=" * 80, flush=True)

    # Assert integrity
    assert error_count == 0, f"Found {error_count} pipeline execution errors!"
    assert analyzed_count > 0, "No videos were successfully analyzed!"
    print("\n[ALL CHECKS PASSED] Stage 2 video inference pipeline verified successfully.", flush=True)


if __name__ == "__main__":
    run_stage2_test()
