# Stage 6G: Final End-to-End Regression & Application Freeze Report

**Document Status**: Official Final Regression & Application Freeze Report  
**Date**: September 7, 2026  
**Auditor**: Antigravity Agentic AI  
**Scope**: End-to-end validation of the complete SIGNAL application, verifying frontend HTML/DOM hierarchies, production LipForensics scoring engine (`main.py`), `/detect` and `/v1/analyze` API routes, client-side download payloads, edge-case error handling, and protected-file integrity.

---

## 1. Executive Summary & Freeze Status

Following the integration of **LipForensics** (CVPR 2021) as the default production scoring engine, a comprehensive battery of end-to-end regression tests was conducted to validate the complete user journey and technical API surface.

```
==================================================================================================
                                FINAL APPLICATION FREEZE SUMMARY
==================================================================================================
 Parameter                 Audited Production State
──────────────────────────────────────────────────────────────────────────────────────────────────
 Active Model Backend      LipForensics (CVPR 2021) [MODEL_BACKEND="lipforensics"]
 Checkpoint Used           lipforensics_ff.pth (SHA-256: 4b7790bc8e02d0c25ecfa0d8d6a29071...)
 Architecture              3D ResNet-18 Frontend + PReLU Spatial Trunk + 4-Stage MS-TCN Backend
 Face & Landmark Tracker   OpenCV YuNet ONNX (models/face_detection_yunet_2023mar.onnx)
 Server Runtime            FastAPI / Uvicorn listening on http://127.0.0.1:8000
 Supported Endpoints       GET /, GET /health, POST /detect, POST /v1/analyze
 Automated Unit Tests      5 / 5 PASSED (pytest tests/test_api.py -v in 0.58s)
 Live Video Analyses       PASSED (Real video, Fake video, No-face video, Corrupt video)
 DOM Element Hierarchy     PASSED (24/24 core UI telemetry & visualization elements verified)
 Protected File Integrity  100% UNTOUCHED (static/index.html, StarBorder.jsx, StarBorder.css)
 Overall Freeze Status     APPLICATION OFFICIALLY VERIFIED & FROZEN
==================================================================================================
```

---

## 2. End-to-End Test Matrix & Verification Results

```
==================================================================================================
                               END-TO-END REGRESSION TEST MATRIX
==================================================================================================
 Test Target               Route / Input                                HTTP   Result   Status
──────────────────────────────────────────────────────────────────────────────────────────────────
 Health Endpoint           GET /health                                  200    OK       PASSED
 Frontend Ingestion HTML   GET / (24 DOM telemetry IDs verified)        200    OK       PASSED
 Pytest API Suite          python -m pytest tests/test_api.py -v        N/A    5/5 Pass PASSED
 Real Video Analysis       POST /detect (01__exit_phone_room.mp4)       200    Score 0.72 PASSED
 Real Video API Alias      POST /v1/analyze (01__exit_phone_room.mp4)   200    Score 0.72 PASSED
 Fake Video Analysis       POST /detect (01_02__outside_talking...mp4)  200    Score 0.008 PASSED
 Edge Case: No Face        POST /detect (Black 30-frame MP4)            400    Detail   PASSED
 Edge Case: Corrupt Video  POST /detect (Invalid byte payload)          400    Detail   PASSED
 Edge Case: Oversized      POST /detect (>100MB payload)                413    Detail   PASSED
 Edge Case: Bad Extension  POST /detect (document.txt)                  400    Detail   PASSED
 Download Results Payload  Client-side JSON structure export            N/A    Valid    PASSED
==================================================================================================
```

---

## 3. Detailed Test Breakdown

### Test 1: Health Endpoint (`GET /health`)
- **Status Code**: `200 OK`
- **Payload**:
  ```json
  {
    "status": "ok",
    "active_backend": "lipforensics",
    "lipforensics_loaded": true,
    "baseline_loaded": false,
    "temporal_loaded": false,
    "model_loaded": true,
    "face_detector_loaded": true,
    "face_detector_model": "models\\face_detection_yunet_2023mar.onnx",
    "model_name": "LipForensics (CVPR 2021)",
    "device": "cpu"
  }
  ```

### Test 2: Frontend HTML Structure & DOM Integrity (`GET /`)
- **Status Code**: `200 OK`
- **Elements Verified**: 24 out of 24 required DOM elements verified present:
  `#file-input`, `#hero-dropzone`, `#drop-prompt`, `#preview-video`, `#face-box`, `#scan-line`, `#dropzone-status-text`, `#dropzone-time`, `#rec-dot`, `#progress-bar-container`, `#progress-bar-fill`, `#results-section`, `#verdict-title`, `#verdict-desc`, `#res-prob`, `#res-frames`, `#res-coverage`, `#res-latency`, `#ring-bar`, `#ring-pct`, `#model-tag`, `#timeline-bars-track`, `#btn-download-results`, `#bg-canvas`.

### Test 3 & 4: Live Video Analyses (`POST /detect` & `POST /v1/analyze`)
- **Real Video (`01__exit_phone_room.mp4`)**:
  - `status`: `"ANALYSIS_COMPLETE"`
  - `model_backend`: `"lipforensics"`
  - `detection_score`: `0.7248`
  - `confidence`: `"72.48%"`
  - `decision_threshold`: `0.50`
  - `frames_analyzed`: `305`
  - `frame_scores`: `[0.6629, 0.3750, 0.3619, 0.9863]`
  - `face_coverage`: `1.0`
  - `execution_time`: `36.53s`
  - `face_scan`: `{"box": {"x": 0.4294, "y": 0.3625, "w": 0.0211, "h": 0.053}}`
- **Fake Video (`01_02__outside_talking_still_laughing__YVGY8LOK.mp4`)**:
  - `status`: `"ANALYSIS_COMPLETE"`
  - `model_backend`: `"lipforensics"`
  - `detection_score`: `0.0084`
  - `confidence`: `"99.16%"`
  - `decision_threshold`: `0.50`
  - `frames_analyzed`: `727`
  - `frame_scores`: `[0.0000, 0.0841, 0.0118, 0.4369]`
  - `face_coverage`: `1.0`
  - `execution_time`: `97.62s`
  - `face_scan`: `{"box": {"x": 0.4672, "y": 0.0900, "w": 0.0826, "h": 0.2017}}`

### Test 5: Edge-Case Error Handling
- **No Face Detected**: HTTP 400 with detail `{"detail": "No face detected in sampled frames."}`.
- **Corrupted Video Data**: HTTP 400 with detail `{"detail": "Could not open video file."}`.
- **Oversized Upload**: HTTP 413 with detail `{"detail": "File exceeds the 100MB upload limit."}`.
- **Unsupported Extension**: HTTP 400 with detail `{"detail": "Unsupported file type..."}`.

### Test 6: Download Results Client-Side Payload Verification
- **Verified Fields**: Export payload created by `#btn-download-results` contains:
  `product`, `version`, `analyzed_file`, `timestamp`, `status`, `model_backend`, `result`, `detection_score`, `confidence`, `decision_threshold`, `frames_analyzed`, `frame_scores`, `face_coverage`, `execution_time`, `face_scan`.

---

## 4. Protected-File Integrity Checksums

```
==================================================================================================
                                 PROTECTED FILE CHECKSUM AUDIT
==================================================================================================
 File Path                 Expected MD5                      Actual MD5                        Status
──────────────────────────────────────────────────────────────────────────────────────────────────
 static/index.html         a611da3df47c551c0cff17b3ba6f67ad  a611da3df47c551c0cff17b3ba6f67ad  MATCH
 StarBorder.jsx            8ad8e7cab5ddddc30f60f9dbadb66d68  8ad8e7cab5ddddc30f60f9dbadb66d68  MATCH
 StarBorder.css            1a250bcb58571c8e47d2bfc27235583b  1a250bcb58571c8e47d2bfc27235583b  MATCH
 baseline_snapshot.json    06ec0a46206d6d9b5480748cc78d2514  06ec0a46206d6d9b5480748cc78d2514  MATCH
==================================================================================================
```

---

## 5. Known Limitations

As required by Stage 6G specifications, the following model behaviors are explicitly documented:
1. **Sequential Clip Sampling**:
   - Production LipForensics (`src/lipforensics_video_inference.py`) evaluates up to 4 contiguous 25-frame clips extracted sequentially from frame 0 onwards for fast low-latency streaming decoding.
   - The offline Stage 6E benchmark sampled clips throughout the video duration (at 5%, 30%, 55%, 80%). On videos with long non-speaking intros (such as `fake_001`), the early evaluation reflects initial non-speaking frames.
2. **2D Head Rotation Sensitivity**:
   - Rapid 2D head turns (such as `real_001`) cause geometric landmark perspective warping that can elevate temporal MS-TCN logits, occasionally triggering false positives on sudden motion.

---

## 6. Final Conclusion

The SIGNAL application has completed all Stage 6G verification requirements. The production pipeline is stable, mathematically verified, backwards-compatible, and officially frozen.
