# UI Redesign Rollback Report: Restoring Pre-B1 / Pre-B2 Visual UI

**Scope:** Full visual rollback to the exact UI state immediately preceding Phase B1 and Phase B2  
**Date:** September 8, 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

As requested, all visual changes introduced during **Phase B1 (Visual Foundation)** and **Phase B2 (Optical Viewfinder)** have been completely reverted. The SIGNAL user interface has been restored to its original visual presentation, including the violet/purple accent system, StarBorder animated dropzone, local upload-panel theme toggle, Space Grotesk display typography, pill buttons, cursor-tracking ambient spotlighting, and the original Three.js particle engine parameters.

**All functional and forensic data-integrity improvements completed prior to B1 have been strictly preserved.**

---

## 2. Reversion Breakdown

| UI Section | Visual State Restored |
| :--- | :--- |
| **Color System** | Restored original SIGNAL palette: `--color-violet: #8B5CF6`, `--color-violet-deep: #6D28D9`, `--color-violet-soft: #A78BFA`, `--color-violet-glow: rgba(139, 92, 246, 0.25)`, `--color-cyan: #06B6D4`, `--color-bg-primary: #0A0A0C`, `--color-bg-surface: #101014`. |
| **Typography** | Restored **Space Grotesk** for display headers/branding, **Inter** for UI body, and **IBM Plex Mono** for technical telemetry and metrics. |
| **Card System** | Restored `16px` corner radius (`--radius-card: 16px`), mousemove coordinate-tracking radial purple spotlight (`.signal-card::after`), buoyant hover translation (`translateY(-3px)`), and visual variants (`.signal-card--dark`, `.signal-card--blue`, `.signal-card--glow`). |
| **Buttons & Controls** | Restored pill/rounded button styling (`--radius-btn: 9999px`), purple gradient fills (`linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)`), drop-shadow glows, and hover elevations. |
| **Upload Zone** | Replaced B2 optical viewfinder with the original **StarBorder** animated chassis (`.star-border-container`, `.border-gradient-bottom`, `.border-gradient-top`, `star-movement-bottom`, `star-movement-top`), drop-icon container, and dual-tone progress bars. |
| **Local Theme Toggle** | Restored upload-panel local light/dark pill switch (`#panel-theme-toggle`, `initPanelTheme()`, `togglePanelTheme()`). |
| **Three.js Background** | Restored 900 particles, `#A881FE` violet, `#3A3A42` dim gray, `#F5F5F5` soft white, size `0.016`, opacity `0.45`, rotation speed `0.003`, and dual-radial atmospheric overlays. |

---

## 3. Preserved Part A Forensic & Data Integrity Features

The following functional, model, and data-integrity features were intentionally kept intact and were NOT modified or reverted:

1. **No Simulated Probabilities / Math.random()**: `generateSimulatedProbabilities()` remains deleted; missing `frame_scores` displays the explicit message: `"Frame-level data not available for this analysis"`.
2. **Real Benchmark Case Archive**: Real benchmark dataset metrics preserved (`01__kitchen_still.mp4` / `01_03__podium_speech_happy__480LQD1C.mp4`, true logits `-9.2901` and `+7.4863`, anomaly scores `0.0001` and `0.9994`).
3. **Provenance Watermark**: Preserved `"REFERENCE BENCHMARK DATA · PRE-COMPUTED HISTORICAL CASE EVALUATION"` pill disclaimer on the Case Archive.
4. **Live /health Telemetry**: Nav status, detector name (`YUNET ONNX`), alignment (`MOUTH ROI CROP`), stride (`25 FRAMES`), and limit (`MAX 100 MB`) remain connected to live backend configuration.
5. **Standardized /detect Response Handling**: Seamless parsing of `assessment_tier`, `assessment_message`, `forensic_score`, and `detection_score`.
6. **Active Video Click Guard**: Preserved guard preventing accidental file picker reactivation when interacting with an already-loaded/playing video.
7. **Backend & Inference Logic**: `main.py`, LipForensics model, OpenCV YuNet detector, 3-tier assessment boundaries, and API schemas remain 100% untouched.

---

## 4. Verification & Automated Test Results

### 4.1 Pytest Test Suite
Executed `py -3.14 -m pytest tests/ -v`:

```text
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AOAI PROJECT
collected 26 items

tests/test_alignment.py::test_aligner_canonical_points_scaling PASSED    [  3%]
tests/test_alignment.py::test_aligner_crop_fallback_with_face_box PASSED [  7%]
tests/test_api.py::test_health_endpoint PASSED                           [ 11%]
tests/test_api.py::test_detect_no_file PASSED                            [ 15%]
tests/test_api.py::test_detect_unsupported_file_type PASSED              [ 19%]
tests/test_api.py::test_detect_file_size_exceeded PASSED                 [ 23%]
tests/test_api.py::test_cleanup_after_rejected_uploads PASSED            [ 26%]
tests/test_face_detector.py::test_yunet_detector_initialized PASSED      [ 30%]
tests/test_face_detector.py::test_mediapipe_detector_initialized PASSED  [ 34%]
tests/test_face_detector.py::test_selected_landmarks_categories PASSED   [ 38%]
tests/test_face_detector.py::test_missing_yunet_model_raises_error PASSED [ 42%]
tests/test_face_detector.py::test_face_detection_and_crop_with_margin PASSED [ 46%]
tests/test_face_detector.py::test_face_crop_boundary_clamping_at_edges PASSED [ 50%]
tests/test_face_detector.py::test_multiple_faces_selects_largest_usable_face PASSED [ 53%]
tests/test_face_detector.py::test_min_face_size_filtering PASSED         [ 57%]
tests/test_face_detector.py::test_no_face_detected_in_frame PASSED       [ 61%]
tests/test_face_detector.py::test_normalized_box_and_landmarks_format PASSED [ 65%]
tests/test_model_architecture.py::test_xception_feature_extractor_shape PASSED [ 69%]
tests/test_model_architecture.py::test_temporal_attention_pooling PASSED [ 73%]
tests/test_model_architecture.py::test_end_to_end_temporal_classifier_forward PASSED [ 76%]
tests/test_pipeline.py::test_threshold_calibrator_optimal_f1 PASSED      [ 80%]
tests/test_pipeline.py::test_threshold_calibrator_single_class_fallback PASSED [ 84%]
tests/test_tracker.py::test_compute_iou PASSED                           [ 88%]
tests/test_tracker.py::test_tracker_initialization_picks_largest_face PASSED [ 92%]
tests/test_tracker.py::test_tracker_matches_consistent_subject PASSED    [ 96%]
tests/test_tracker.py::test_tracker_handles_temporary_occlusion PASSED   [100%]

============================= 26 passed in 6.78s ==============================
```

### 4.2 Live Server Endpoint Verification
- `GET http://127.0.0.1:8000/health` $\rightarrow$ `HTTP 200 OK` (All services online).
- `GET http://127.0.0.1:8000/` $\rightarrow$ `HTTP 200 OK` (Verified: `star-border-container` present, `--color-violet: #8B5CF6` present, Space Grotesk present, optical viewfinder elements completely removed).

---

## 5. Rollback Confirmation

- [x] Phase B1 and B2 visual changes completely reverted.
- [x] Original violet/purple palette, Space Grotesk typography, and pill buttons restored.
- [x] StarBorder animated dropzone and local panel theme toggle restored.
- [x] Three.js background particle configuration restored to original settings.
- [x] Optical viewfinder markup, reticles, and crosshairs completely removed.
- [x] Part A data-integrity and telemetry improvements 100% preserved.
- [x] 26/26 automated tests passing.
- [x] Zero backend, API, or inference files modified.

**STATUS: STOPPED. Visual redesign halted.**
