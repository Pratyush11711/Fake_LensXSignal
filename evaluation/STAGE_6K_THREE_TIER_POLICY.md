# Stage 6K: Controlled Three-Tier Forensic Output Policy

> **Product-Policy & Presentation Change Report**
> This document details the validated implementation of the provisional 3-tier forensic assessment policy for the SIGNAL intelligence platform. Binary `REAL / FAKE` assertions and misleading 'Confidence %' metrics have been retired. All underlying LipForensics model weights, architectures, preprocessing, and temporal sampling pipelines remain **100% untouched**.

---

## 1. Executive Summary & Policy Motivation

Following the empirical findings of **Stage 6J (Real-World False-Positive Audit)**, where 44.0% of authentic consumer videos crossed the nominal $T=0.50$ threshold (with 16.0% scoring $\ge 0.99$ due to mobile compression and conversational dynamics), binary classification was determined to be fundamentally unsafe for open-world deployment.

Stage 6K introduces an **Uncertainty & Risk-Communication Policy**:
1. **Retires Binary Assertions**: Discontinues definitive `REAL` and `FAKE` declarations.
2. **Establishes 3-Tier Categorization**:
   - **`LIKELY AUTHENTIC`** ($\text{Score} < 0.40$)
   - **`INCONCLUSIVE — NEEDS REVIEW`** ($0.40 \le \text{Score} < 0.90$)
   - **`SIGNS OF POSSIBLE MANIPULATION`** ($\text{Score} \ge 0.90$)
3. **Decouples Continuous Measurement from UX**: Replaces `Confidence %` with `Forensic Anomaly Score` ($0.0000$ to $1.0000$) accompanied by mandatory environmental context caveats.
4. **Preserves Complete Backward Compatibility**: The `/detect` and `/v1/analyze` endpoints retain all existing fields while adding structured tier semantics.

---

## 2. Exact Boundary Test Results

| Anomaly Score | Expected Tier | Evaluated Tier | Advisory Message Summary | Status |
| :--- | :--- | :--- | :--- | :---: |
| `0.00` | **LIKELY AUTHENTIC** | LIKELY AUTHENTIC | Low forensic anomaly signal detected. This do... | ✅ PASS |
| `0.39` | **LIKELY AUTHENTIC** | LIKELY AUTHENTIC | Low forensic anomaly signal detected. This do... | ✅ PASS |
| `0.40` | **INCONCLUSIVE — NEEDS REVIEW** | INCONCLUSIVE — NEEDS REVIEW | Ambiguous forensic signal. Independent review... | ✅ PASS |
| `0.50` | **INCONCLUSIVE — NEEDS REVIEW** | INCONCLUSIVE — NEEDS REVIEW | Ambiguous forensic signal. Independent review... | ✅ PASS |
| `0.76` | **INCONCLUSIVE — NEEDS REVIEW** | INCONCLUSIVE — NEEDS REVIEW | Ambiguous forensic signal. Independent review... | ✅ PASS |
| `0.89` | **INCONCLUSIVE — NEEDS REVIEW** | INCONCLUSIVE — NEEDS REVIEW | Ambiguous forensic signal. Independent review... | ✅ PASS |
| `0.90` | **SIGNS OF POSSIBLE MANIPULATION** | SIGNS OF POSSIBLE MANIPULATION | High forensic anomaly signal detected. This d... | ✅ PASS |
| `0.99` | **SIGNS OF POSSIBLE MANIPULATION** | SIGNS OF POSSIBLE MANIPULATION | High forensic anomaly signal detected. This d... | ✅ PASS |
| `1.00` | **SIGNS OF POSSIBLE MANIPULATION** | SIGNS OF POSSIBLE MANIPULATION | High forensic anomaly signal detected. This d... | ✅ PASS |

---

## 3. End-to-End Video API Validations

Four diverse video cases were evaluated through the full FastAPI `/detect` and `/v1/analyze` pipelines:

| Test Case | Filename | Ground Truth | Raw Score | Legacy Binary Verdict | 3-Tier Assessment | API Response Code |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **Primary Real Case (High Anomaly)** | `videoplayback.mp4` | REAL | `0.9913` | ~~FAKE~~ | **SIGNS OF POSSIBLE MANIPULATION** | `HTTP 200` |
| **Low-Score Known-Real** | `download.mp4` | REAL | `0.0025` | ~~REAL~~ | **LIKELY AUTHENTIC** | `HTTP 200` |
| **High-Score Known-Real** | `download (5).mp4` | REAL | `1.0000` | ~~FAKE~~ | **SIGNS OF POSSIBLE MANIPULATION** | `HTTP 200` |
| **Known Fake Video** | `01_03__hugging_happy__ISF9SP4G.mp4` | FAKE | `0.9999` | ~~FAKE~~ | **SIGNS OF POSSIBLE MANIPULATION** | `HTTP 200` |

### In-Depth Case Analysis:

1. **Primary Known Real Case (`videoplayback.mp4`)**:
   - **Raw Forensic Score**: `0.9913` (MS-TCN temporal anomaly logit $+4.73$)
   - **Legacy UI Display**: `FAKE (99.13% Confidence)` $\rightarrow$ *Severe False Accusation*
   - **New 3-Tier UI Display**: `SIGNS OF POSSIBLE MANIPULATION` with `Forensic Anomaly Score: 0.9913`
   - **Displayed Advisory**: *"High forensic anomaly signal detected. This does not by itself establish manipulation. Compression, motion, lighting, and camera conditions can elevate the score. Independent verification is recommended."*
   - **Outcome**: The product no longer declares the video definitively fraudulent; it transparently communicates high signal anomaly alongside required environmental caveats.

2. **Known Authentic Low-Anomaly Video**:
   - **Raw Forensic Score**: `< 0.40`
   - **New 3-Tier UI Display**: `LIKELY AUTHENTIC`
   - **Displayed Advisory**: *"Low forensic anomaly signal detected. This does not guarantee authenticity."*

3. **Known Synthetic Deepfake Video**:
   - **Raw Forensic Score**: `> 0.90`
   - **New 3-Tier UI Display**: `SIGNS OF POSSIBLE MANIPULATION`
   - **Displayed Advisory**: Full environmental caveat and verification notice.

---

## 4. Modified Files & System Components

| Component | File Path | Scope of Change | Model Weights / Inference Impact |
| :--- | :--- | :--- | :--- |
| **Backend Core** | `main.py` | Added `get_forensic_assessment_tier()`; enriched `/detect` and `/v1/analyze` response schemas with `forensic_score`, `assessment_tier`, `assessment_message`. | **None (100% Untouched)** |
| **Web Interface** | `static/index.html` | Updated verdict tags, typography, CSS color tokens (`--color-inconclusive`), telemetry labels (`FORENSIC ANOMALY SCORE`), gauge animations, and download schemas. | **None (Presentation Only)** |
| **Evaluation Suite** | `scratch/test_stage6k_policy.py` | Automated unit boundary tests and end-to-end multi-video HTTP verification. | **None** |

---

## 5. Statistical & Operational Limitations

1. **Provisional Operational Bands**: The boundaries ($0.40$ and $0.90$) are risk-management buffers established from Stage 6E/6I/6J diagnostic data. They are **not** universal physical constants.
2. **Model Accuracy Invariant**: This policy change modifies risk communication and decision presentation; it does **not** alter the underlying ROC curve, feature embeddings, or AUC of the LipForensics neural network.
3. **In-the-Wild Recall Requirement**: A balanced dataset of 25 authentic and 25 synthetic mobile/web videos is required before finalizing permanent calibration parameters.

---

## 6. Verification & Completion Status

- [x] Exact boundary tests ($0.00$ to $1.00$) passed.
- [x] End-to-end API evaluation on `videoplayback.mp4` verified.
- [x] End-to-end API evaluation on authentic low-score, authentic high-score, and benchmark fake verified.
- [x] Pytest suite (26 tests) passed with 100% success.
- [x] Model weights, checkpoint, and inference untouched.
- [x] Frontend JSON download schema verified.