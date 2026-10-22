# SIGNAL Human-Readable Forensic Report Download (PDF)

## Overview & Objective
This update adds a user-friendly **"DOWNLOAD REPORT"** capability to the SIGNAL Forensic Intelligence System. Users can now export a formatted, human-readable forensic report in PDF format (`SIGNAL-forensic-report.pdf`) directly from their browser, alongside the existing technical JSON export (`SIGNAL-analysis-results.json`) and print workflow (`PRINT REPORT`).

---

## Button Layout & Action Hierarchy

The footer of the forensic report modal sits together with the three unified controls:

```
[ PRINT REPORT ]   [ DOWNLOAD REPORT ]   [ DOWNLOAD RAW JSON ]
```

### Visual & Functional Hierarchy
| Control | Action Type | Visual Hierarchy | Target Output / Action |
| :--- | :--- | :--- | :--- |
| **DOWNLOAD REPORT** | Primary Action | Highest prominence (`.btn-modal-primary`, glowing cyan `#33d6e8`) | Downloads `SIGNAL-forensic-report.pdf` without opening/navigating away |
| **PRINT REPORT** | Secondary Action | Balanced secondary (`.btn-modal-secondary`, translucent border) | Opens browser print dialog with dedicated `@media print` styling |
| **DOWNLOAD RAW JSON** | Tertiary / Technical | Muted tertiary (`.btn-modal-tertiary`, low-contrast outline) | Downloads `SIGNAL-analysis-results.json` directly via blob/data URI |

---

## PDF Generation Details

### 1. Generation Method
- **Client-Side Generation**: Uses `jsPDF` (v2.5.1 loaded via CDN).
- **Format**: Standard A4 (`210mm x 297mm`, portrait).
- **Visual Aesthetic**: High-contrast dark forensic dossier style matching SIGNAL's design system (`#0a0c0e` dark background, `#0d0f12` cards, `#33d6e8` cyan accents, and calibrated tier indicators).
- **Direct Download**: Triggers standard browser download (`doc.save("SIGNAL-forensic-report.pdf")`) without navigating away or automatically launching preview viewers.

### 2. Export Filenames
- **Forensic PDF Report**: `SIGNAL-forensic-report.pdf`
- **Technical Analysis JSON**: `SIGNAL-analysis-results.json`

### 3. Data Integrity & Data Source
Both the PDF report and the raw JSON export originate exclusively from the **same actual `/detect` API response**:
- **Branding**: `SIGNAL // FORENSIC INTELLIGENCE REPORT` with unique generated Case Report ID (`SIG-2026-XXXXXX`).
- **Source Sequence**: Actual uploaded filename (e.g. `evidence.mp4`).
- **Analysis Timestamp**: Exact local client timestamp of the audit.
- **Assessment Tier**: Preserves the exact 3-tier policy:
  - `LIKELY AUTHENTIC` (Green badge / border)
  - `INCONCLUSIVE — NEEDS REVIEW` (Amber badge / border)
  - `SIGNS OF POSSIBLE MANIPULATION` (Red badge / border)
- **Forensic Score**: Exact calibrated probability score from the model (`0.0000 - 1.0000`).
- **Frames Audited**: Verified sample frames audited (e.g. `15 / 15`).
- **Face Track Coverage**: Percentage of frame sequence where subject was detected and tracked.
- **Analysis Latency**: Actual execution time in seconds.
- **Landmark Detector**: Detector model name (`YuNet ONNX`, 5-point affine mouth ROI).
- **Model Backend**: Inference neural network backend (`LipForensics`, ResNet-18 + MS-TCN spatio-temporal network).
- **Assessment Message**: Backend forensic evaluation message.
- **Temporal Frame-Level Summary**: Number of sample windows, peak window score, mean window score, and anomaly flag count.
- **Plain-Language Interpretation**: Contextual explanation of the findings and operational guidance for non-technical users.
- **Evidentiary Limitations & Notice**: Explicit legal and operational disclaimer noting that automated screening results do not constitute absolute proof and require human verification.

---

## Verification & Test Results

### 1. Test Suite Execution
Executed backend and pipeline regression tests via Python 3.14:
```bash
py -3.14 -m pytest tests/ -v
```

### 2. Test Output
```
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

============================= 26 passed in 6.08s ==============================
```

### 3. Verification Checklist
- [x] `[ PRINT REPORT ]`, `[ DOWNLOAD REPORT ]`, and `[ DOWNLOAD RAW JSON ]` sit together in the report footer.
- [x] Primary action `DOWNLOAD REPORT` triggers `SIGNAL-forensic-report.pdf` download directly.
- [x] Secondary action `PRINT REPORT` opens clean print preview with `@media print` styles.
- [x] Tertiary action `DOWNLOAD RAW JSON` triggers `SIGNAL-analysis-results.json` download.
- [x] Neither action navigates away from the SIGNAL application.
- [x] 3-tier language (`LIKELY AUTHENTIC`, `INCONCLUSIVE — NEEDS REVIEW`, `SIGNS OF POSSIBLE MANIPULATION`) preserved.
- [x] Zero mock or fabricated scores; strictly uses actual `/detect` inference telemetry.
- [x] 26/26 automated unit and integration tests passing.
