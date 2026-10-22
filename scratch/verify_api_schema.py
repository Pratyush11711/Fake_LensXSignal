import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
import main

def test_api_alignment():
    client = TestClient(main.app)

    # 1. Health check verification
    health_resp = client.get("/health")
    print("=== /health STATUS ===", health_resp.status_code)
    health_json = health_resp.json()
    print("=== /health RESPONSE ===")
    print(json.dumps(health_json, indent=2))

    # 2. Detect endpoint verification with a real video
    test_video_path = Path("D:/benchmark_dataset/real/01__kitchen_still.mp4")
    if not test_video_path.exists():
        candidates = list(Path("D:/benchmark_dataset").rglob("*.mp4"))
        if candidates:
            test_video_path = candidates[0]

    print("\n=== USING TEST VIDEO ===", test_video_path)
    
    with open(test_video_path, "rb") as f:
        video_bytes = f.read()

    print(f"Uploaded video size: {len(video_bytes)} bytes")
    t0 = time.time()
    response = client.post(
        "/detect",
        files={"video": ("01__kitchen_still.mp4", video_bytes, "video/mp4")}
    )
    elapsed = time.time() - t0
    print(f"Request duration: {elapsed:.2f}s, HTTP Status: {response.status_code}")

    if response.status_code != 200:
        print("ERROR RESPONSE:", response.text)
        return

    data = response.json()
    print("\n=== /detect REAL RESPONSE PAYLOAD ===")
    print(json.dumps(data, indent=2))

    # Verification checklist
    required_fields = [
        "forensic_score",
        "assessment_tier",
        "assessment_message",
        "frames_analyzed",
        "face_coverage",
        "execution_time",
        "model_backend",
        "face_scan",
        "frame_scores",
        "decision_threshold",
    ]

    print("\n=== FRONTEND SCHEMA COMPLIANCE AUDIT ===")
    all_matched = True
    for field in required_fields:
        present = field in data
        val = data.get(field)
        print(f"Field '{field}': {'[OK]' if present else '[MISSING]'} -> {repr(val)[:60]}")
        if not present:
            all_matched = False

    if "face_scan" in data and data["face_scan"]:
        has_box = "box" in data["face_scan"]
        print(f"Field 'face_scan.box': {'[OK]' if has_box else '[MISSING]'} -> {data['face_scan'].get('box')}")
        if not has_box:
            all_matched = False

    print(f"\nSCHEMA AUDIT RESULT: {'ALL FIELDS VALIDATED MATCHING FRONTEND' if all_matched else 'SCHEMA MISMATCH FOUND'}")

    Path("scratch").mkdir(exist_ok=True)
    with open("scratch/verified_real_detect_payload.json", "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, indent=2)

if __name__ == "__main__":
    test_api_alignment()
