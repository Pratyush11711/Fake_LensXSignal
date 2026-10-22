# System Architecture Redesign — "What Runs Under the Hood"

## 1. Executive Summary

This document records the replacement of the previous System Architecture section with the reference **"What runs under the hood"** architecture section in [`static/index.html`](file:///d:/AOAI%20PROJECT/static/index.html). The new implementation features a 6-node horizontal signal-flow pathway, a continuous travelling mint signal dot, and sequential staggered card pulse animations while preserving navigation anchors, responsive wrapping, forensic accuracy, and 100% test passing.

---

## 2. Heading & Subtitle Updates

- **Section Anchor**: Retained `<section class="architecture-section" id="architecture">` to ensure full compatibility with the top navigation (`href="#architecture"`).
- **Heading**: `What runs under the hood`
- **Subtitle**: `A high-throughput inference pathway — ONNX for detection, PyTorch for temporal feature pooling.`

---

## 3. Six Architecture Nodes

The 6-stage technical pathway represents the exact end-to-end inference stack:

| Node | Role Label | Indicator | Title | Description |
| :--- | :--- | :---: | :--- | :--- |
| **01** | `INGEST` | `●` (#10B981) | Video decoder | Hardware-accelerated stream decoding. |
| **02** | `SAMPLE` | `●` (#10B981) | Temporal stride | Uniform multi-window frame extraction. |
| **03** | `DETECT` | `●` (#10B981) | YuNet ONNX | 5-point landmark detection & tracking. |
| **04** | `ALIGN` | `●` (#10B981) | Affine warp | Mouth ROI normalized cropping. |
| **05** | `INFER` | `●` (#10B981) | LipForensics | ResNet-18 + MS-TCN spatio-temporal net. |
| **06** | `OUTPUT` | `●` (#10B981) | Policy engine | Calibrated three-tier assessment. |

---

## 4. Signal Flow & Travelling Dot Animation

- **Track Line**: 1px subtle graphite connector line (`rgba(255, 255, 255, 0.08)`) spanning behind the cards.
- **Signal Dot**: Glowing 8px mint/emerald dot (`#10B981`, `box-shadow: 0 0 10px rgba(16, 185, 129, 0.9), 0 0 20px rgba(16, 185, 129, 0.5)`).
- **Movement**: Continuous 5.0s linear sweep from left to right along the connector track (`signalTravel` keyframe animation).

---

## 5. Staggered Card Pulse Animation

Each node card executes a subtle 5.0s pulse cycle timed with the traveling signal:
- **Node 01 (`INGEST`)**: Delay `0.0s`
- **Node 02 (`SAMPLE`)**: Delay `0.8s`
- **Node 03 (`DETECT`)**: Delay `1.6s`
- **Node 04 (`ALIGN`)**: Delay `2.4s`
- **Node 05 (`INFER`)**: Delay `3.2s`
- **Node 06 (`OUTPUT`)**: Delay `4.0s`

**Pulse Effect**:
- Subtle border brightening to `rgba(168, 129, 254, 0.4)`.
- Internal ambient highlight (`rgba(168, 129, 254, 0.04)`).
- Zero bounce, rotation, or vertical displacement.

---

## 6. Responsive & Layout Behavior

- **Desktop (>= 1100px)**: Clean 6-column grid pathway spanning the container width.
- **Tablet / Mobile (< 1100px)**: Contained horizontal scroll wrapper (`.arch-flow-wrapper`) with custom thin scrollbar, preserving minimum card widths (160px) without causing page-wide horizontal overflow.
- **Scroll Reveal**: Integrated with `.reveal-item` intersection observer for clean entry.

---

## 7. Automated Test & Verification Results

```text
============================= test session starts =============================
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AOAI PROJECT
plugins: anyio-4.13.0
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

======================== 26 passed in 66.73s (0:01:06) ========================
```
