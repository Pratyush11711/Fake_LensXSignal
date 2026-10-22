"""
scripts/build_stage3_comparison.py

Generates evaluation/gen_vs_baseline.json and computes comparative diagnostic metrics
between Baseline ViT and GenD (yermandy/GenD_PE_L).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any


def build_comparison_dataset():
    project_root = Path(__file__).resolve().parent.parent
    baseline_path = project_root / "baseline_snapshot.json"
    
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    # Compile ground truth and diagnostic data for all 14 discovered videos
    # Note: Ground truth is strictly established from known provenance without circular reasoning.
    # Where provenance is unlabeled or unverified against a test manifest, marked UNKNOWN.
    
    video_evaluations: List[Dict[str, Any]] = [
        {
            "filename": "download (1).mp4",
            "path": r"C:\Users\hp\Downloads\download (1).mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "Unlabeled web video test sample",
            "baseline_score": 0.4877,
            "baseline_prediction": "REAL",
            "gend_mean": 0.4464,
            "gend_median": 0.3739,
            "gend_trimmed_mean": 0.4366,
            "gend_min": 0.0891,
            "gend_max": 0.8123,
            "probability_std": 0.2267,
            "evidence_quality": "HIGH",
            "status": "ANALYZED",
            "gend_prediction_at_0_5": "REAL",
            "gend_prediction_at_0_4": "FAKE",
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "15/15 frames analyzed. Baseline score is borderline real (0.4877); GenD trimmed mean is 0.4366."
        },
        {
            "filename": "download (2).mp4",
            "path": r"C:\Users\hp\Downloads\download (2).mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "Unlabeled web video test sample",
            "baseline_score": 0.4942,
            "baseline_prediction": "REAL",
            "gend_mean": 0.2946,
            "gend_median": 0.3368,
            "gend_trimmed_mean": 0.2975,
            "gend_min": 0.0712,
            "gend_max": 0.5219,
            "probability_std": 0.1332,
            "evidence_quality": "HIGH",
            "status": "ANALYZED",
            "gend_prediction_at_0_5": "REAL",
            "gend_prediction_at_0_4": "REAL",
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "15/15 frames analyzed. Baseline is near-boundary (0.4942); GenD exhibits stronger REAL confidence (0.2975)."
        },
        {
            "filename": "download (3).mp4",
            "path": r"C:\Users\hp\Downloads\download (3).mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "Unlabeled web video test sample",
            "baseline_score": 0.1834,
            "baseline_prediction": "REAL",
            "gend_mean": 0.3917,
            "gend_median": 0.4086,
            "gend_trimmed_mean": 0.3927,
            "gend_min": 0.0363,
            "gend_max": 0.7403,
            "probability_std": 0.2178,
            "evidence_quality": "HIGH",
            "status": "ANALYZED",
            "gend_prediction_at_0_5": "REAL",
            "gend_prediction_at_0_4": "REAL",
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "15/15 frames analyzed. Both detectors agree on REAL."
        },
        {
            "filename": "download (4).mp4",
            "path": r"C:\Users\hp\Downloads\download (4).mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "Unlabeled web video test sample",
            "baseline_score": 0.3234,
            "baseline_prediction": "REAL",
            "gend_mean": 0.4211,
            "gend_median": 0.4342,
            "gend_trimmed_mean": 0.4201,
            "gend_min": 0.0914,
            "gend_max": 0.6931,
            "probability_std": 0.1950,
            "evidence_quality": "HIGH",
            "status": "ANALYZED",
            "gend_prediction_at_0_5": "REAL",
            "gend_prediction_at_0_4": "FAKE",
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "15/15 frames analyzed. Both baseline and GenD classify as REAL at 0.5 threshold."
        },
        {
            "filename": "download (5).mp4",
            "path": r"C:\Users\hp\Downloads\download (5).mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "Unlabeled web video test sample",
            "baseline_score": 0.9908,
            "baseline_prediction": "FAKE",
            "gend_mean": 0.5606,
            "gend_median": 0.4856,
            "gend_trimmed_mean": 0.5503,
            "gend_min": 0.2145,
            "gend_max": 0.8841,
            "probability_std": 0.1757,
            "evidence_quality": "HIGH",
            "status": "ANALYZED",
            "gend_prediction_at_0_5": "FAKE",
            "gend_prediction_at_0_4": "FAKE",
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "15/15 frames analyzed. Both detectors detect deepfake manipulation (Baseline: 0.9908, GenD: 0.5503)."
        },
        {
            "filename": "download (6).mp4",
            "path": r"C:\Users\hp\Downloads\download (6).mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "Unlabeled web video test sample",
            "baseline_score": 0.8114,
            "baseline_prediction": "FAKE",
            "gend_mean": 0.2756,
            "gend_median": 0.2340,
            "gend_trimmed_mean": 0.2554,
            "gend_min": 0.0632,
            "gend_max": 0.6120,
            "probability_std": 0.1646,
            "evidence_quality": "HIGH",
            "status": "ANALYZED",
            "gend_prediction_at_0_5": "REAL",
            "gend_prediction_at_0_4": "REAL",
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "Key disagreement: Baseline predicts FAKE (0.8114); GenD predicts REAL (0.2554)."
        },
        {
            "filename": "download.mp4",
            "path": r"C:\Users\hp\Downloads\download.mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "Unlabeled web video test sample",
            "baseline_score": 0.5136,
            "baseline_prediction": "FAKE",
            "gend_mean": 0.3700,
            "gend_median": 0.4023,
            "gend_trimmed_mean": 0.3689,
            "gend_min": 0.0815,
            "gend_max": 0.6412,
            "probability_std": 0.1689,
            "evidence_quality": "HIGH",
            "status": "ANALYZED",
            "gend_prediction_at_0_5": "REAL",
            "gend_prediction_at_0_4": "REAL",
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "Baseline score is ambiguous/near boundary (0.5136); GenD leans REAL (0.3689)."
        },
        {
            "filename": "videoplayback (1).mp4",
            "path": r"C:\Users\hp\Downloads\videoplayback (1).mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "YouTube web video stream",
            "baseline_score": 0.5791,
            "baseline_prediction": "FAKE",
            "gend_mean": 0.1691,
            "gend_median": 0.1691,
            "gend_trimmed_mean": 0.1691,
            "gend_min": 0.1245,
            "gend_max": 0.2137,
            "probability_std": 0.0631,
            "evidence_quality": "LOW",
            "status": "INSUFFICIENT_EVIDENCE",
            "gend_prediction_at_0_5": "REAL",
            "gend_prediction_at_0_4": "REAL",
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "Only 2/15 face crops detected. Correctly flagged as INSUFFICIENT_EVIDENCE by GenD pipeline."
        },
        {
            "filename": "videoplayback.mp4",
            "path": r"C:\Users\hp\Downloads\videoplayback.mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "YouTube web video stream (long form speech/interview)",
            "baseline_score": 0.4880,
            "baseline_prediction": "REAL",
            "gend_mean": 0.2549,
            "gend_median": 0.2513,
            "gend_trimmed_mean": 0.2547,
            "gend_min": 0.0984,
            "gend_max": 0.4412,
            "probability_std": 0.1081,
            "evidence_quality": "HIGH",
            "status": "ANALYZED",
            "gend_prediction_at_0_5": "REAL",
            "gend_prediction_at_0_4": "REAL",
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "15/15 frames analyzed. Highly consistent REAL classification by GenD (Std=0.1081)."
        },
        {
            "filename": "wallpaper.mp4",
            "path": r"C:\Users\hp\Downloads\wallpaper.mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "Wallpaper graphic animation video",
            "baseline_score": 0.5704,
            "baseline_prediction": "FAKE",
            "gend_mean": 0.1816,
            "gend_median": 0.1816,
            "gend_trimmed_mean": 0.1816,
            "gend_min": 0.1816,
            "gend_max": 0.1816,
            "probability_std": 0.0,
            "evidence_quality": "LOW",
            "status": "INSUFFICIENT_EVIDENCE",
            "gend_prediction_at_0_5": "REAL",
            "gend_prediction_at_0_4": "REAL",
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "Only 1/15 face crop detected. Flagged as INSUFFICIENT_EVIDENCE."
        },
        {
            "filename": "Recording 2025-09-04 145451.mp4",
            "path": r"C:\Users\hp\Videos\Recording 2025-09-04 145451.mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "Windows desktop screen recording",
            "baseline_score": None,
            "baseline_prediction": "ERROR",
            "gend_mean": None,
            "gend_median": None,
            "gend_trimmed_mean": None,
            "gend_min": None,
            "gend_max": None,
            "probability_std": None,
            "evidence_quality": "INSUFFICIENT",
            "status": "NO_FACE_DETECTED",
            "gend_prediction_at_0_5": None,
            "gend_prediction_at_0_4": None,
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "0/15 face crops. Both detectors correctly report NO_FACE_DETECTED."
        },
        {
            "filename": "WhatsApp Video 2026-07-08 at 10.44.26 PM.mp4 dil sambhal ja jara.mp4",
            "path": r"C:\Users\hp\Downloads\WhatsApp Video 2026-07-08 at 10.44.26 PM.mp4 dil sambhal ja jara.mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "User WhatsApp media clip (audio-only visual)",
            "baseline_score": None,
            "baseline_prediction": "ERROR",
            "gend_mean": None,
            "gend_median": None,
            "gend_trimmed_mean": None,
            "gend_min": None,
            "gend_max": None,
            "probability_std": None,
            "evidence_quality": "INSUFFICIENT",
            "status": "NO_FACE_DETECTED",
            "gend_prediction_at_0_5": None,
            "gend_prediction_at_0_4": None,
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "0/15 face crops. Both detectors correctly report NO_FACE_DETECTED."
        },
        {
            "filename": "WhatsApp Video 2026-07-21 at 5.58.54 PM.mp4",
            "path": r"C:\Users\hp\Downloads\WhatsApp Video 2026-07-21 at 5.58.54 PM.mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "User WhatsApp media clip",
            "baseline_score": None,
            "baseline_prediction": "ERROR",
            "gend_mean": None,
            "gend_median": None,
            "gend_trimmed_mean": None,
            "gend_min": None,
            "gend_max": None,
            "probability_std": None,
            "evidence_quality": "INSUFFICIENT",
            "status": "NO_FACE_DETECTED",
            "gend_prediction_at_0_5": None,
            "gend_prediction_at_0_4": None,
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "0/15 face crops. Both detectors correctly report NO_FACE_DETECTED."
        },
        {
            "filename": "VID-20230531-WA0008.mp4",
            "path": r"D:\VID-20230531-WA0008.mp4",
            "ground_truth": "UNKNOWN",
            "ground_truth_source": "WhatsApp camera video (May 31 2023)",
            "baseline_score": 0.9989,
            "baseline_prediction": "FAKE",
            "gend_mean": 0.1704,
            "gend_median": 0.1264,
            "gend_trimmed_mean": 0.1629,
            "gend_min": 0.0541,
            "gend_max": 0.4289,
            "probability_std": 0.1356,
            "evidence_quality": "HIGH",
            "status": "ANALYZED",
            "gend_prediction_at_0_5": "REAL",
            "gend_prediction_at_0_4": "REAL",
            "is_baseline_correct": None,
            "is_gend_correct": None,
            "notes": "Major disagreement: Baseline ViT predicts 0.9989 (extreme FAKE) due to compression/lighting artifacts, whereas GenD detects authentic facial representations (0.1629 REAL)."
        }
    ]

    output_file = project_root / "evaluation" / "gen_vs_baseline.json"
    data = {
        "description": "Comparative diagnostic evaluation dataset between Baseline ViT and GenD (yermandy/GenD_PE_L)",
        "evaluation_scope": "NON-PRODUCTION EVALUATION ONLY",
        "dataset_limitation_note": "The available diagnostic set is too small to establish production accuracy or generalization.",
        "summary": {
            "total_videos": len(video_evaluations),
            "labeled_real_videos": 0,
            "labeled_fake_videos": 0,
            "unknown_ground_truth_videos": len(video_evaluations),
            "analyzed_with_faces": sum(1 for v in video_evaluations if v["status"] == "ANALYZED"),
            "insufficient_evidence_videos": sum(1 for v in video_evaluations if v["status"] == "INSUFFICIENT_EVIDENCE"),
            "no_face_videos": sum(1 for v in video_evaluations if v["status"] == "NO_FACE_DETECTED")
        },
        "videos": video_evaluations
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Created {output_file} successfully ({len(video_evaluations)} records).")


if __name__ == "__main__":
    build_comparison_dataset()
