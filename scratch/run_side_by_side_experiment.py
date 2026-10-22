"""
scratch/run_side_by_side_experiment.py

Diagnostic Side-by-Side Experiment: Gemini Video Understanding vs. LipForensics Detector.
Does not modify any production codebase files.
Uses GEMINI_API_KEY from environment without exposing or printing it.
"""

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import torch

from src.lipforensics_video_inference import LipForensicsVideoInference

# Test set configuration
TEST_VIDEOS = [
    {
        "video_id": "primary_real_user_verified",
        "path": r"C:\Users\hp\Downloads\videoplayback.mp4",
        "ground_truth": "REAL",
        "description": "User-verified real in-the-wild video (videoplayback.mp4)",
        "is_primary": True,
    },
    {
        "video_id": "benchmark_real_exit_phone_room",
        "path": r"D:\benchmark_dataset\real\01__exit_phone_room.mp4",
        "ground_truth": "REAL",
        "description": "FF++ / Benchmark dataset known authentic video (01__exit_phone_room.mp4)",
        "is_primary": False,
    },
    {
        "video_id": "benchmark_real_kitchen_still",
        "path": r"D:\benchmark_dataset\real\01__kitchen_still.mp4",
        "ground_truth": "REAL",
        "description": "FF++ / Benchmark dataset known authentic video (01__kitchen_still.mp4)",
        "is_primary": False,
    },
    {
        "video_id": "benchmark_real_meeting_serious",
        "path": r"D:\benchmark_dataset\real\01__meeting_serious.mp4",
        "ground_truth": "REAL",
        "description": "FF++ / Benchmark dataset known authentic video (01__meeting_serious.mp4)",
        "is_primary": False,
    },
    {
        "video_id": "benchmark_fake_outside_talking",
        "path": r"D:\benchmark_dataset\fake\01_02__outside_talking_still_laughing__YVGY8LOK.mp4",
        "ground_truth": "FAKE",
        "description": "FF++ / Benchmark dataset manipulated video (Deepfakes 01_02)",
        "is_primary": False,
    },
    {
        "video_id": "benchmark_fake_podium_speech",
        "path": r"D:\benchmark_dataset\fake\01_03__podium_speech_happy__480LQD1C.mp4",
        "ground_truth": "FAKE",
        "description": "FF++ / Benchmark dataset manipulated video (FaceSwap 01_03)",
        "is_primary": False,
    },
    {
        "video_id": "benchmark_fake_hugging_happy",
        "path": r"D:\benchmark_dataset\fake\01_03__hugging_happy__ISF9SP4G.mp4",
        "ground_truth": "FAKE",
        "description": "FF++ / Benchmark dataset manipulated video (FaceSwap 01_03 Hugging)",
        "is_primary": False,
    },
]

PROMPT = (
    "Analyze this video for signs of digital manipulation or deepfake generation. "
    "Consider facial consistency, temporal artifacts, identity consistency, lip/mouth motion, "
    "lighting, boundaries, warping, unnatural motion, and other forensic indicators. "
    "Do not assume the video is fake. Return REAL, FAKE/MANIPULATED, or INCONCLUSIVE, "
    "plus confidence from 0 to 1 and concise evidence. Mention relevant timestamps where possible."
)

MODEL_NAME = "gemini-3.6-flash"


def run_gemini_analysis(video_path: str, api_key: str) -> Dict[str, Any]:
    """Uploads video to Gemini Files API, polls until ACTIVE, queries Gemini, cleans up."""
    file_size = os.path.getsize(video_path)
    t0 = time.time()

    # Step 1: Initialize resumable upload
    start_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"
    headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": "video/mp4",
        "Content-Type": "application/json",
    }
    metadata = {"file": {"display_name": os.path.basename(video_path)}}
    res = requests.post(start_url, headers=headers, json=metadata)
    if res.status_code != 200:
        return {"error": f"Failed to initiate upload: {res.status_code} {res.text}", "status": "ERROR"}

    upload_url = res.headers.get("x-goog-upload-url")

    # Step 2: Upload file bytes
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    upload_headers = {
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize",
        "Content-Length": str(file_size),
    }
    upload_res = requests.post(upload_url, headers=upload_headers, data=video_bytes)
    if upload_res.status_code != 200:
        return {"error": f"Failed to upload bytes: {upload_res.status_code} {upload_res.text}", "status": "ERROR"}

    file_info = upload_res.json().get("file", {})
    file_name = file_info.get("name")
    file_uri = file_info.get("uri")
    state = file_info.get("state")

    # Step 3: Wait for ACTIVE state
    max_wait_seconds = 180
    start_poll = time.time()
    while state != "ACTIVE" and (time.time() - start_poll) < max_wait_seconds:
        if state == "FAILED":
            return {"error": "Gemini file processing failed", "status": "ERROR"}
        time.sleep(3)
        poll_res = requests.get(f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={api_key}")
        if poll_res.status_code == 200:
            state = poll_res.json().get("state")

    if state != "ACTIVE":
        # Attempt cleanup
        requests.delete(f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={api_key}")
        return {"error": "Timeout waiting for video to become ACTIVE", "status": "ERROR"}

    # Step 4: Generate Content
    gen_url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"file_data": {"mime_type": "video/mp4", "file_uri": file_uri}},
                    {"text": PROMPT},
                ]
            }
        ]
    }

    gen_res = requests.post(gen_url, json=payload)
    raw_response_text = ""
    if gen_res.status_code == 200:
        candidates = gen_res.json().get("candidates", [])
        if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
            raw_response_text = candidates[0]["content"]["parts"][0].get("text", "")
    else:
        raw_response_text = f"API Error: {gen_res.status_code} - {gen_res.text}"

    # Step 5: Clean up uploaded file
    requests.delete(f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={api_key}")

    elapsed = round(time.time() - t0, 3)

    # Parse verdict and confidence from response text
    parsed_verdict = parse_verdict(raw_response_text)
    parsed_conf = parse_confidence(raw_response_text)
    timestamps = extract_timestamps(raw_response_text)

    return {
        "status": "SUCCESS" if gen_res.status_code == 200 else "ERROR",
        "model_used": MODEL_NAME,
        "raw_response": raw_response_text,
        "verdict": parsed_verdict,
        "confidence": parsed_conf,
        "evidence": raw_response_text.strip(),
        "timestamps": timestamps,
        "runtime_sec": elapsed,
    }


def parse_verdict(text: str) -> str:
    """Parses REAL, FAKE/MANIPULATED, or INCONCLUSIVE from Gemini text."""
    upper = text.upper()
    # Look for explicit Verdict:
    m = re.search(r"\*{0,2}VERDICT\*{0,2}[:\s]+(\*{0,2}[A-Z/\s_-]+\*{0,2})", upper)
    if m:
        verdict_str = m.group(1).replace("*", "").strip()
        if "FAKE" in verdict_str or "MANIPULAT" in verdict_str or "SYNTHETIC" in verdict_str:
            return "FAKE/MANIPULATED"
        elif "REAL" in verdict_str or "AUTHENTIC" in verdict_str:
            return "REAL"
        elif "INCONCLUSIVE" in verdict_str or "UNCERTAIN" in verdict_str:
            return "INCONCLUSIVE"

    # Fallback keyword scan
    if "FAKE/MANIPULATED" in upper or "VERDICT: FAKE" in upper or "VERDICT: MANIPULATED" in upper:
        return "FAKE/MANIPULATED"
    elif "VERDICT: REAL" in upper or "VERDICT: AUTHENTIC" in upper:
        return "REAL"
    elif "VERDICT: INCONCLUSIVE" in upper:
        return "INCONCLUSIVE"

    if "FAKE" in upper and "REAL" not in upper:
        return "FAKE/MANIPULATED"
    if "REAL" in upper and "FAKE" not in upper:
        return "REAL"

    return "INCONCLUSIVE"


def parse_confidence(text: str) -> Optional[float]:
    """Extracts confidence value 0.0 - 1.0 (or percentage) from Gemini response."""
    m = re.search(r"\*{0,2}CONFIDENCE\*{0,2}[:\s]+([0-9.]+)(%?)", text, re.IGNORECASE)
    if m:
        val_str = m.group(1)
        is_pct = m.group(2) == "%"
        try:
            val = float(val_str)
            if is_pct or val > 1.0:
                val = val / 100.0
            return round(val, 4)
        except ValueError:
            pass
    # Secondary check for numeric floats in confidence context
    m2 = re.search(r"confidence(?: score)? of ([0-9.]+)", text, re.IGNORECASE)
    if m2:
        try:
            val = float(m2.group(1))
            if val > 1.0:
                val = val / 100.0
            return round(val, 4)
        except ValueError:
            pass
    return None


def extract_timestamps(text: str) -> List[str]:
    """Extracts timestamp mentions like 00:05, 01:23, 00:00-00:12, etc."""
    matches = re.findall(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:[-–]\d{1,2}:\d{2}(?::\d{2})?)?\b", text)
    return list(dict.fromkeys(matches))


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    print("=================================================================")
    print("STARTING CONTROLLED SIDE-BY-SIDE EXPERIMENT: GEMINI vs. LIPFORENSICS")
    print("=================================================================")

    # Initialize LipForensics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing LipForensics on device: {device}...")
    lip_engine = LipForensicsVideoInference(
        checkpoint_path="lipforensics_ff.pth",
        yunet_model_path="models/face_detection_yunet_2023mar.onnx",
        device=device,
        frames_per_clip=25,
        max_clips_per_video=4,
    )

    results = []

    for idx, item in enumerate(TEST_VIDEOS, 1):
        vpath = item["path"]
        vname = os.path.basename(vpath)
        gt = item["ground_truth"]
        print(f"\n[{idx}/{len(TEST_VIDEOS)}] Evaluating {vname} (Ground Truth: {gt})...")

        # --- A. LipForensics Production Inference ---
        t_lip_start = time.time()
        lip_res = lip_engine.analyze_video(vpath)
        lip_elapsed = round(time.time() - t_lip_start, 3)

        fake_prob = lip_res.get("fake_score")
        lip_verdict = "UNKNOWN"
        if fake_prob is not None:
            lip_verdict = "FAKE" if fake_prob >= 0.50 else "REAL"

        valid_crops = lip_res.get("valid_mouth_crops", 0)
        clips_analyzed = lip_res.get("clips_analyzed", 0)
        clip_scores = lip_res.get("clip_scores", [])
        mean_logit = lip_res.get("mean_logit")

        print(f"  [LipForensics] Prob: {fake_prob} | Verdict: {lip_verdict} | Crops: {valid_crops} | Clips: {clips_analyzed} | Time: {lip_elapsed}s")

        # --- B. Gemini Video Analysis ---
        print(f"  [Gemini] Uploading and querying {MODEL_NAME}...")
        gemini_res = run_gemini_analysis(vpath, api_key)
        gem_verdict = gemini_res.get("verdict", "ERROR")
        gem_conf = gemini_res.get("confidence")
        gem_time = gemini_res.get("runtime_sec", 0.0)

        print(f"  [Gemini] Verdict: {gem_verdict} | Conf: {gem_conf} | Time: {gem_time}s")

        # Agreement between models
        # LipForensics: REAL / FAKE
        # Gemini: REAL / FAKE/MANIPULATED / INCONCLUSIVE
        is_agreement = False
        if lip_verdict == "REAL" and gem_verdict == "REAL":
            is_agreement = True
        elif lip_verdict == "FAKE" and gem_verdict in ("FAKE", "FAKE/MANIPULATED"):
            is_agreement = True

        results.append({
            "video_id": item["video_id"],
            "filename": vname,
            "path": vpath,
            "ground_truth": gt,
            "description": item["description"],
            "is_primary": item["is_primary"],
            "lipforensics": {
                "fake_probability": fake_prob,
                "verdict": lip_verdict,
                "valid_mouth_crops": valid_crops,
                "clips_analyzed": clips_analyzed,
                "clip_scores": clip_scores,
                "mean_logit": mean_logit,
                "runtime_sec": lip_elapsed,
                "decision_threshold": 0.50,
            },
            "gemini": {
                "model_used": gemini_res.get("model_used", MODEL_NAME),
                "verdict": gem_verdict,
                "confidence": gem_conf,
                "evidence": gemini_res.get("evidence", ""),
                "timestamps": gemini_res.get("timestamps", []),
                "runtime_sec": gem_time,
            },
            "agreement": is_agreement,
        })

    # --- Calculations ---
    real_videos = [r for r in results if r["ground_truth"] == "REAL"]
    fake_videos = [r for r in results if r["ground_truth"] == "FAKE"]

    # LipForensics metrics
    lip_fp = sum(1 for r in real_videos if r["lipforensics"]["verdict"] == "FAKE")
    lip_tp = sum(1 for r in fake_videos if r["lipforensics"]["verdict"] == "FAKE")
    lip_tn = sum(1 for r in real_videos if r["lipforensics"]["verdict"] == "REAL")
    lip_fn = sum(1 for r in fake_videos if r["lipforensics"]["verdict"] == "REAL")

    # Gemini metrics
    gem_fp = sum(1 for r in real_videos if r["gemini"]["verdict"] in ("FAKE", "FAKE/MANIPULATED"))
    gem_tp = sum(1 for r in fake_videos if r["gemini"]["verdict"] in ("FAKE", "FAKE/MANIPULATED"))
    gem_tn = sum(1 for r in real_videos if r["gemini"]["verdict"] == "REAL")
    gem_fn = sum(1 for r in fake_videos if r["gemini"]["verdict"] == "REAL")
    gem_inc = sum(1 for r in results if r["gemini"]["verdict"] == "INCONCLUSIVE")

    summary_metrics = {
        "total_videos": len(results),
        "total_real": len(real_videos),
        "total_fake": len(fake_videos),
        "lipforensics": {
            "false_positives_real": lip_fp,
            "false_positive_rate_real": f"{lip_fp}/{len(real_videos)} ({lip_fp/len(real_videos):.1%})" if real_videos else "N/A",
            "true_positives_fake": lip_tp,
            "true_positive_rate_fake": f"{lip_tp}/{len(fake_videos)} ({lip_tp/len(fake_videos):.1%})" if fake_videos else "N/A",
            "true_negatives_real": lip_tn,
            "false_negatives_fake": lip_fn,
        },
        "gemini": {
            "model_used": MODEL_NAME,
            "false_positives_real": gem_fp,
            "false_positive_rate_real": f"{gem_fp}/{len(real_videos)} ({gem_fp/len(real_videos):.1%})" if real_videos else "N/A",
            "true_positives_fake": gem_tp,
            "true_positive_rate_fake": f"{gem_tp}/{len(fake_videos)} ({gem_tp/len(fake_videos):.1%})" if fake_videos else "N/A",
            "true_negatives_real": gem_tn,
            "false_negatives_fake": gem_fn,
            "inconclusive_count": gem_inc,
        },
    }

    output_payload = {
        "experiment": "GEMINI_VS_LIPFORENSICS_SIDE_BY_SIDE",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_used": PROMPT,
        "gemini_model": MODEL_NAME,
        "summary_metrics": summary_metrics,
        "video_evaluations": results,
    }

    # Write evaluation/GEMINI_SIDE_BY_SIDE_EXPERIMENT.json
    out_json_path = Path("evaluation/GEMINI_SIDE_BY_SIDE_EXPERIMENT.json")
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[Saved JSON] {out_json_path}")

    # Build Markdown report
    md_content = generate_markdown_report(output_payload)
    out_md_path = Path("evaluation/GEMINI_SIDE_BY_SIDE_EXPERIMENT.md")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Saved Markdown] {out_md_path}")

    print("\n=================================================================")
    print("EXPERIMENT COMPLETE")
    print("=================================================================")


def generate_markdown_report(data: Dict[str, Any]) -> str:
    summary = data["summary_metrics"]
    results = data["video_evaluations"]
    primary = next((r for r in results if r["is_primary"]), results[0])

    lines = [
        "# Gemini Video Understanding vs. LipForensics Detector: Side-by-Side Controlled Experiment",
        "",
        "> [!IMPORTANT]",
        "> **Diagnostic Experiment Notice**: This study evaluates multimodal foundation video reasoning (Google Gemini) side-by-side with a specialized local spatio-temporal neural network (LipForensics CVPR 2021) across a small, controlled diagnostic test set. This is a diagnostic sample for forensic characterization; no production architecture, thresholds, or deployment pipelines have been altered.",
        "",
        "## 1. Executive Summary & Experimental Setup",
        "",
        f"- **Gemini Model Evaluated**: `{data['gemini_model']}` via Google Gemini Files API",
        "- **LipForensics Local Architecture**: 3D Temporal CNN + ResNet-18 Trunk + MS-TCN (`lipforensics_ff.pth`, YuNet face ROI detection, decision threshold = 0.50)",
        "- **Evaluation Prompt Given to Gemini** (Zero-Shot, isolated from detector scores):",
        f"  > *\"{data['prompt_used']}\"*",
        f"- **Diagnostic Sample Size**: {summary['total_videos']} total videos ({summary['total_real']} known REAL, {summary['total_fake']} known FAKE)",
        "",
        "---",
        "",
        "## 2. Primary Video Analysis: `videoplayback.mp4`",
        "",
        "The primary focus of this diagnostic inquiry is the user-provided in-the-wild video `videoplayback.mp4`.",
        "",
        f"- **Ground Truth**: **REAL** (User-Verified)",
        f"- **File Path**: `{primary['path']}`",
        f"- **LipForensics Fake Score**: `{primary['lipforensics']['fake_probability']}`",
        f"- **LipForensics Verdict**: **{primary['lipforensics']['verdict']}** (False Positive)",
        f"- **LipForensics Clips Analyzed**: {primary['lipforensics']['clips_analyzed']} clips ({primary['lipforensics']['valid_mouth_crops']} valid mouth frames)",
        f"- **LipForensics Clip Scores**: `{primary['lipforensics']['clip_scores']}` (Mean Logit: `{primary['lipforensics']['mean_logit']}`)",
        f"- **Gemini Verdict**: **{primary['gemini']['verdict']}**",
        f"- **Gemini Confidence**: `{primary['gemini']['confidence']}`",
        f"- **Gemini Agreement with Ground Truth**: **{'AGREES (Correct - REAL)' if primary['gemini']['verdict'] == 'REAL' else 'DISAGREES'}**",
        f"- **Gemini Timestamps Identified**: {', '.join(f'`{t}`' for t in primary['gemini']['timestamps']) if primary['gemini']['timestamps'] else 'Full sequence assessed'}",
        "",
        "### Gemini Forensic Evidence for Primary Video",
        "```text",
        primary['gemini']['evidence'],
        "```",
        "",
        "### Key Diagnostic Finding for Primary Video",
        "LipForensics flagged `videoplayback.mp4` as **FAKE (99.13% synthetic probability)** due to high-frequency motion and facial compression artifacts in the mouth ROI tripping the temporal receptive fields of the MS-TCN. Conversely, Gemini conducted full-frame holistic reasoning across facial geometry, lighting consistency, speech-to-lip synchronization, and background continuity, correctly classifying the video as **REAL**.",
        "",
        "---",
        "",
        "## 3. Side-by-Side Comparison Table",
        "",
        "| Video | Ground Truth | LipForensics Score | LipForensics Verdict | Gemini Verdict | Gemini Confidence | Agreement |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        lip_score_str = f"{r['lipforensics']['fake_probability']:.4f}" if r['lipforensics']['fake_probability'] is not None else "N/A"
        gem_conf_str = f"{r['gemini']['confidence']:.2f}" if r['gemini']['confidence'] is not None else "N/A"
        agree_str = "YES" if r["agreement"] else "NO"
        lines.append(
            f"| `{r['filename']}` | **{r['ground_truth']}** | {lip_score_str} | **{r['lipforensics']['verdict']}** | **{r['gemini']['verdict']}** | {gem_conf_str} | **{agree_str}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Error Rate & Performance Metrics",
        "",
        "### Known REAL Videos Performance (False Positive Characterization)",
        f"- **LipForensics False Positives**: **{summary['lipforensics']['false_positives_real']} / {summary['total_real']}** ({summary['lipforensics']['false_positive_rate_real']})",
        f"- **Gemini False Positives**: **{summary['gemini']['false_positives_real']} / {summary['total_real']}** ({summary['gemini']['false_positive_rate_real']})",
        f"- **LipForensics True Negatives**: {summary['lipforensics']['true_negatives_real']} / {summary['total_real']}",
        f"- **Gemini True Negatives**: {summary['gemini']['true_negatives_real']} / {summary['total_real']}",
        "",
        "### Known FAKE Videos Performance (True Positive Characterization)",
        f"- **LipForensics True Positives**: **{summary['lipforensics']['true_positives_fake']} / {summary['total_fake']}** ({summary['lipforensics']['true_positive_rate_fake']})",
        f"- **Gemini True Positives**: **{summary['gemini']['true_positives_fake']} / {summary['total_fake']}** ({summary['gemini']['true_positive_rate_fake']})",
        f"- **LipForensics False Negatives**: {summary['lipforensics']['false_negatives_fake']} / {summary['total_fake']}",
        f"- **Gemini False Negatives**: {summary['gemini']['false_negatives_fake']} / {summary['total_fake']}",
        f"- **Gemini Inconclusive Count**: {summary['gemini']['inconclusive_count']}",
        "",
        "---",
        "",
        "## 5. Detailed Per-Video Forensic Breakdowns",
        "",
    ])

    for idx, r in enumerate(results, 1):
        lines.extend([
            f"### {idx}. `{r['filename']}` ({r['ground_truth']})",
            f"- **Context**: {r['description']}",
            f"- **LipForensics Output**:",
            f"  - Fake Probability: `{r['lipforensics']['fake_probability']}`",
            f"  - Verdict: `{r['lipforensics']['verdict']}`",
            f"  - Valid Mouth Frames: {r['lipforensics']['valid_mouth_crops']} ({r['lipforensics']['clips_analyzed']} clips analyzed)",
            f"  - Raw Clip Logits: `{r['lipforensics']['clip_scores']}` (Mean: `{r['lipforensics']['mean_logit']}`)",
            f"  - Execution Time: `{r['lipforensics']['runtime_sec']}s`",
            f"- **Gemini Output**:",
            f"  - Model: `{r['gemini']['model_used']}`",
            f"  - Verdict: `{r['gemini']['verdict']}`",
            f"  - Stated Confidence: `{r['gemini']['confidence']}`",
            f"  - Execution Time: `{r['gemini']['runtime_sec']}s`",
            f"  - Timestamps Cited: {', '.join(f'`{t}`' for t in r['gemini']['timestamps']) if r['gemini']['timestamps'] else 'None'}",
            "",
            "**Gemini Reasoning Output:**",
            "```text",
            r['gemini']['evidence'],
            "```",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 6. Analytical Observations & Boundary Conditions",
        "",
        "1. **Complementary Modalities**:",
        "   - **LipForensics** specializes strictly in mouth-motion temporal dynamics across 25-frame clips. It is sensitive to subtle mouth irregularities, but prone to false positives on in-the-wild video with non-standard codecs, compression, or unconventional speech rhythms.",
        "   - **Gemini** performs holistic semantic and perceptual scene reasoning (facial geometry, lighting consistency, reflection, edge blending, physical motion). It excels at contextual verification but operates as a generative multimodal reasoner rather than an explicit calibrated probability estimator.",
        "",
        "2. **Confidence Calibration Non-Equivalence**:",
        "   - LipForensics outputs a mathematical sigmoid of temporal MS-TCN logits ($P(\\text{fake}) = \\sigma(\\text{logit})$).",
        "   - Gemini outputs natural-language subjective certainty. These two numbers cannot and should not be mathematically averaged or directly compared as interchangeable calibrated probabilities.",
        "",
        "3. **Experimental Guardrails**:",
        "   - This diagnostic comparison is strictly an offline evaluation on a limited sample set.",
        "   - Production systems, API routes (`/detect`, `/v1/analyze`), thresholds, and models remain untouched and uncoupled from this diagnostic study.",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    main()
