# SIGNAL UI Cursor & Re-Upload Interaction Audit & Fixes

## 1. Executive Summary

This evaluation document details the interaction, cursor behavior fixes, and the final secondary CTA design of the `"INSERT NEW VIDEO"` control in [`static/index.html`](file:///d:/AOAI%20PROJECT/static/index.html). No visual styling, fonts, colors, layouts, animations, backend API contracts, ML inference pipelines, thresholds, or Case Archive data were modified.

---

## 2. Cursor Rules Fixed

### 2.1 Global Text & Surface Reset
* **Problem**: Browsers defaulted standard UI text elements (headings, paragraphs, labels, telemetry indicators, metadata cells) to `cursor: auto`, which turns into an I-beam text cursor (`cursor: text`) upon hovering over interface copy.
* **Solution**:
  - Implemented explicit non-editable interface text reset: `html, body, h1..h6, p, span, div, li, ul, ol, label, table, th, td, code, em, strong, .signal-card, .verdict-card, .gauge-card, .timeline-card, .methodology-panel, .schematic-node, .case-container-card, .meta-label, .meta-val, .panel-label, .dropzone-timestamp, .cell-label, .cell-value, .ring-number, .benchmark-watermark-pill, .section-eyebrow, .section-title, .section-subtitle, .hero-headline, .hero-subtext, .hero-system-telemetry-strip, .tag-mono` set to `cursor: default;`.
  - Normal text selection is fully preserved (no global `user-select: none` workaround).
  - Explicit `cursor: text;` is strictly constrained to editable input fields (`input[type="text"]`, `input[type="search"]`, `input[type="email"]`, `input[type="number"]`, `textarea`, `[contenteditable="true"]`).

### 2.2 Pre-Upload Dropzone Behavior
* **Pre-Upload State**:
  - Upload/dropzone target and prompt region (`.inner-dropzone:not(.has-video)`, `.dropzone-media-viewport`, `.dropzone-prompt-content`, `.drop-icon-box`, `.dropzone-main-text`, `.dropzone-sub-text`) use `cursor: pointer;`.
  - Header controls maintain their respective scoped cursors (theme toggle: `pointer`, timestamp/labels: `default`).
  - `"INSERT NEW VIDEO"` control is completely hidden (`display: none;`) with zero reserved blank space.

### 2.3 Post-Upload Video Viewport Guard
* **Selected-Video State**:
  - Video viewport and all associated elements (`.inner-dropzone.has-video`, `.dropzone-media-viewport`, `.preview-video`, `.face-tracker-box`, `.scan-line`, `.dropzone-header`, `.dropzone-footer-meta`) strictly use `cursor: default;`.
  - Hovering over the video viewport frame does not show a hand cursor and does not suggest the video frame is a dropzone trigger.
  - Active-video click guard prevents any clicks on the dropzone or video viewport from triggering the file picker.

### 2.4 Cursor Scope Audit Table

| UI Component | Element / Selector | Cursor State | Action / Justification |
| :--- | :--- | :--- | :--- |
| **Brand Link** | `.brand-group` | `pointer` | Navigates to top / home |
| **Navigation Links** | `.nav-link` | `pointer` | Smooth-scrolls to section |
| **Hero Headline & Copy** | `.hero-headline, .hero-subtext` | `default` | Informational text |
| **Hero Telemetry** | `.hero-system-telemetry-strip` | `default` | Live status indicators |
| **Hero Hardware Metadata** | `.meta-label, .meta-val` | `default` | Read-only hardware telemetry |
| **Pre-Upload Dropzone** | `.inner-dropzone:not(.has-video)` | `pointer` | Click opens system file picker |
| **Post-Upload Dropzone** | `.inner-dropzone.has-video` | `default` | Media playback frame |
| **Post-Upload Viewport** | `.dropzone-media-viewport, video` | `default` | Video render surface |
| **Re-Upload Trigger** | `.btn-insert-new-video` | `pointer` | Opens file picker to replace video |
| **Theme Toggle** | `.panel-theme-toggle, .panel-toggle-btn` | `pointer` | Switches dark/light dropzone theme |
| **Workflow Stages** | `.workflow-step-node` | `default` | Informational pipeline schematic |
| **Verdict Cards & Telemetry** | `.verdict-card, .telemetry-cell` | `default` | Model output display |
| **Download Report** | `.btn-download-results` | `pointer` | Downloads JSON forensic report |
| **Score Ring Gauge** | `.gauge-card, .ring-number` | `default` | Anomaly score visualizer |
| **Timeline Waveform** | `.timeline-card, .timeline-bar-column` | `default` | Frame anomaly visualizer |
| **Methodology Cards** | `.methodology-panel` | `default` | Technical documentation |
| **Architecture Nodes** | `.schematic-node` | `default` | Hardware/software flow diagram |
| **Case Archive Buttons** | `.case-btn` | `pointer` | Switches benchmark reference case |
| **Case Telemetry Box** | `.case-container-card, .case-telemetry-box`| `default` | Pre-computed benchmark reference |
| **Footer Links** | `.footer-link-item a` | `pointer` | Navigation links |
| **Text Inputs** | `input[type="text"], textarea` | `text` | User text entry |

---

## 3. "INSERT NEW VIDEO" Final Secondary CTA Design

1. **Placement & Centering**:
   - Centered horizontally below the video preview and metadata row (`CANONICAL CROPS 256×256` / `FILENAME.MP4`).
   - Occupies its own dedicated line within `.dropzone-action-row`.
2. **Visual Hierarchy & Styling**:
   - Styled as an intentional secondary CTA: `height: 44px;`, `border-radius: 10px;`, `border: 1px solid rgba(168, 129, 254, 0.45);`, `padding: 0 24px;`.
   - Uses the signal lavender accent (`#A881FE`) with dark background and medium-weight IBM Plex Mono typography.
   - Includes a clean SVG rotate/refresh icon (`↻`).
   - Subtle hover transition (`background: rgba(168, 129, 254, 0.12); border-color: #A881FE; color: #FFFFFF;`).
3. **Visibility Lifecycle**:
   - **STATE A (No Video)**: `btn-insert-new-video` and `dropzone-action-row` remain `display: none;` with zero reserved empty space.
   - **STATE B (Video Selected)**: When footage is ingested, `heroDropzone` receives `.has-video`, revealing the centered secondary action button.
   - **STATE C (Click Re-Upload)**: Clicking `"INSERT NEW VIDEO"` triggers `fileInput.click()` via `e.stopPropagation()`. Selecting replacement footage runs the existing ingestion, face tracking, and `/detect` inference workflow.
   - **STATE D (Hover Video)**: Video preview displays `cursor: default;` and clicking the video does not trigger the file picker.

---

## 4. Automated Test & Verification Results

### 4.1 Test Suite Run (`pytest tests/ -v`)
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
tests/test_no_face_detected_in_frame PASSED                               [ 61%]
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

============================= 26 passed in 13.34s =============================
```

### 4.2 HTTP Endpoint Verification
* `GET /`: Status 200 OK
* `GET /health`: Status 200 OK (`status: ok`, `model_loaded: true`)
