# Methodology Cards Visual Restyling & Ambient Animation Evaluation

## 1. Executive Summary

This document records the visual restyling of the three methodology cards in [`static/index.html`](file:///d:/AOAI%20PROJECT/static/index.html). The update introduces a deliberate **purple-to-black gradient progression**, smooth **ambient radial light animations**, and **restrained fixed-position hover illumination** while keeping all copy, layouts, responsive behavior, backend inference, thresholds, and tests 100% intact.

---

## 2. Visual Gradient Progression

The three cards now feature an atmospheric progression from structured violet illumination to an almost-pure obsidian void:

| Card | Heading | Background Treatment | Border Styling |
| :--- | :--- | :--- | :--- |
| **Card 01** | `METHOD // 01`<br>**Boundary & Artifact Localization** | `radial-gradient(circle at 20% 15%, rgba(168, 129, 254, 0.18), rgba(168, 129, 254, 0.06) 45%, #0B0B10 100%)` | `border-color: rgba(168, 129, 254, 0.22)` |
| **Card 02** | `METHOD // 02`<br>**Spatio-Temporal Coherence** | `radial-gradient(circle at 25% 15%, rgba(168, 129, 254, 0.10), rgba(168, 129, 254, 0.035) 45%, #0B0B10 100%)` | `border-color: rgba(168, 129, 254, 0.16)` |
| **Card 03** | `METHOD // 03`<br>**Calibrated Three-Tier Policy** | `radial-gradient(circle at 30% 15%, rgba(168, 129, 254, 0.055), rgba(168, 129, 254, 0.015) 45%, #0B0B10 100%)` | `border-color: rgba(168, 129, 254, 0.12)` |

* **Palette Base**: Void surface `#0B0B10`.
* **Hue Integrity**: Restrained lavender/purple `#A881FE` without any oversaturated neon or artificial glow.

---

## 3. Ambient Light Shift Animations

Each card features a subtle, slow-moving radial light highlight drifting across its surface on asynchronous timelines to create an organic, breathing depth:

```css
@keyframes methodLightShift01 {
  0%   { background-position: 0% 0%; }
  50%  { background-position: 30% 25%; }
  100% { background-position: 15% 45%; }
}

@keyframes methodLightShift02 {
  0%   { background-position: 10% 0%; }
  50%  { background-position: 40% 30%; }
  100% { background-position: 20% 50%; }
}

@keyframes methodLightShift03 {
  0%   { background-position: 15% 5%; }
  50%  { background-position: 35% 35%; }
  100% { background-position: 25% 55%; }
}
```

* **Timing & Delays**:
  - **Card 01**: `8s ease-in-out infinite alternate;`
  - **Card 02**: `9s ease-in-out infinite alternate 1.5s;`
  - **Card 03**: `10s ease-in-out infinite alternate 3s;`
* **Constraints Met**: Zero physical translation, zero bouncing, zero scaling, zero rotation, zero neon strobe.

---

## 4. Fixed Hover Illumination

Hovering over a methodology card gently illuminates its internal radial gradient and sharpens its graphite/lavender border without moving the element:

* **Translation**: `transform: none;` (overriding default card `translateY(-3px)`).
* **Shadow**: Fixed `box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);` without large glowing blooms.
* **Illumination**:
  - **Card 01 Hover**: `rgba(168, 129, 254, 0.24)` peak, border `rgba(168, 129, 254, 0.38)`.
  - **Card 02 Hover**: `rgba(168, 129, 254, 0.15)` peak, border `rgba(168, 129, 254, 0.30)`.
  - **Card 03 Hover**: `rgba(168, 129, 254, 0.09)` peak, border `rgba(168, 129, 254, 0.24)`.

---

## 5. Automated Test & Verification Results

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

============================= 26 passed in 15.85s =============================
```
