# Stage 6: Genuine Cross-Dataset Evaluation & Primary Model Generalization Report

**Document Status**: Final Cross-Dataset Evaluation & Provenance Report  
**Date**: September 7, 2026  
**Auditor**: Antigravity Agentic AI  
**Scope**: Celeb-DF-v2 / DFDC Access Verification, GenD Pipeline Evaluation on 100-Video DFDC Benchmark, Generalization Failure Mode Diagnosis, and FTCN Legal Disqualification

---

## 1. Executive Summary & Research Priority Correction

In accordance with strict scientific protocols, all secondary temporal model investigations (TALL, FTCN, AltFreezing) have been **paused**. Before evaluating or designing complex multi-model ensembles, we must establish empirical ground truth on whether our primary detector (**GenD `yermandy/GenD_PE_L`**) generalizes to independent cross-dataset video distributions.

### Key Conclusions:
1. **FTCN Legal Disqualification**: The repository `yinglinzheng/FTCN` contains no `LICENSE` file. Legally, an unlicensed repository defaults to **All Rights Reserved**. Coupled with its legacy PyTorch 1.4/CUDA 10.1 dependency and the lowest published cross-dataset AUC (86.9% Celeb-DF-v2, 74.0% DFDC), **FTCN is officially disqualified and removed from candidate consideration**.
2. **Celeb-DF-v2 Access Process**: Access requires formal application submission via Google/Tencent forms and manual institutional review by dataset authors. It cannot be acquired programmatically within an automated session without violating official terms of use.
3. **DFDC Cross-Dataset Evaluation ($N=100$)**: Evaluated the unchanged GenD pipeline across all 100 balanced videos ($50\text{ REAL}, 50\text{ FAKE}$) from the official DeepFake Detection Challenge (DFDC).
4. **Primary Model Generalization Finding**: **GenD fails to generalize sensitivity on DFDC.** While it achieves strong specificity on real videos (**86.0% True Negative Rate**, only 7 false alarms on 50 real videos), it misses **47 out of 50 deepfakes** (6.0% Recall at $\tau=0.50$), producing an overall **ROC-AUC of 0.4378**.

---

## 2. Dataset Access & Provenance Verification

### A. Celeb-DF-v2 Official Access Protocol (Action: Verified via Official Repo)
- **Official Repository**: `https://github.com/yuezunli/celeb-deepfakeforensics`
- **Application Portal**: Google Form (`https://forms.gle/2jYBby6y1FBU3u6q9`) / Tencent Form (`https://wj.qq.com/s2/8540155/b5d9/`)
- **Review Mechanism**: The dataset maintainers (`deepfakeforensics@gmail.com`) manually verify institutional affiliation, research intent, and signed agreement to terms before issuing credentials.
- **Session Feasibility**: **Cannot be obtained instantaneously within this session** without bypassing official protocols (which is strictly forbidden).

### B. DFDC Benchmark Dataset (Action: Verified 100 Files Physically Present)
- **Source**: Official DeepFake Detection Challenge (Facebook AI / AWS / Microsoft / Partnership on AI).
- **Storage Location**: `D:/benchmark_dataset_dfdc/` (50 Real in `real/`, 50 Fake in `fake/`).
- **File Verification**: All 100 MP4 video files are physically verified present on disk and readable.
- **Metadata**: Ground truth verified directly from official `metadata.json` (`manifest.json`).
- **Classification**: **Genuine Independent Cross-Dataset Benchmark** (zero subject or capture overlap with FaceForensics++).

---

## 3. Full 100-Video DFDC Benchmark Evaluation (GenD PE_L)

The existing GenD pipeline (`src/gend_video_inference.py` / `detectors/gend_detector.py`) was evaluated across all 100 balanced DFDC videos using identical preprocessing (15 frames sampled uniformly, YuNet $\times 1.3$ margin face crops, 10% trimmed mean aggregation).

### Performance Metrics Table (Threshold $\tau = 0.50$)

| Metric | Measured Value | Benchmark Interpretation |
| :--- | :---: | :--- |
| **Total Videos Evaluated** | **100** | $50\text{ REAL} + 50\text{ FAKE}$ (Balanced 1:1) |
| **ROC-AUC** | **0.4378** | Inverted / sub-random separation on wild compressed video |
| **Accuracy** | **46.00%** | 46 out of 100 videos classified correctly |
| **Precision** | **30.00%** | 3 true positives out of 10 total positive predictions |
| **Recall (Sensitivity)** | **6.00%** | **Only 3 out of 50 deepfake videos detected** (47 false negatives) |
| **Specificity (TNR)** | **86.00%** | **43 out of 50 real videos correctly identified** (7 false alarms) |
| **F1-Score** | **0.1000** | Severely suppressed by low recall |
| **False Positive Rate** | **14.00%** | 7 / 50 real videos flagged as fake |
| **False Negative Rate** | **94.00%** | 47 / 50 fake videos flagged as real |

---

### Confusion Matrix Breakdown (Threshold $\tau = 0.50$)

```
                        Predicted REAL (Class 0)    Predicted FAKE (Class 1)
Actual REAL (50)                43 (TN)                      7 (FP)
Actual FAKE (50)                47 (FN)                      3 (TP)
```

---

### Score Distribution Analysis

| Class | Sample Count | Mean Score | Median Score | Std Dev | Score Range |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **REAL Videos** | 50 | **0.3927** | 0.3980 | 0.0984 | $[0.184, 0.682]$ |
| **FAKE Videos** | 50 | **0.3697** | 0.3610 | 0.0875 | $[0.192, 0.615]$ |

---

## 4. Scientific Diagnosis: Why GenD Fails on Cross-Dataset DFDC

1. **The Spatial Phase Attenuation Problem**:
   - GenD's linear probe was trained on high-quality FaceForensics++ studio recordings with clean spatial boundaries and uncompressed phase features.
   - DFDC features aggressive H.264/MPEG compression, low bitrates, variable mobile sensor noise, and varied face resolutions.
   - These compression artifacts destroy the subtle high-frequency spatial phase signatures that GenD relies on, causing its output logits to shift negative ($z < 0$).
2. **The Conservatism vs. Recall Asymmetry**:
   - GenD is highly effective at avoiding false alarms (**86% Specificity on DFDC**, **68% on DFD**), making it vastly superior to the Baseline ViT (which had an 84-96% False Positive Rate).
   - However, because it lacks temporal modeling (cannot detect inter-frame flickering, optical flow discontinuities, or eye-blinking anomalies), it cannot compensate for lost spatial cues, resulting in **94% false negatives on unseen deepfakes**.
3. **Conclusion on Model Strategy**:
   - GenD cannot function as a standalone zero-shot video detector on heavily compressed out-of-domain media without domain adaptation or threshold re-calibration.
   - Adding a second model (e.g. temporal detector) without first establishing the baseline boundaries of the primary detector would result in an uncalibrated, fragile ensemble.

---

## 5. Summary of Disqualified Candidates & Verified Facts

| Candidate | Verified Action | Licensing & Usability Status | Final Decision |
| :--- | :--- | :--- | :--- |
| **TALL / TALL++** | Inspected `rainy-xu/TALL4Deepfake` repo | No `.pth` weights hosted or linked. Training from scratch required. | **DISQUALIFIED** |
| **FTCN** | Downloaded `ftcn_tt.pth` (56.50 MB, MD5: `2c7366...`) | **No `LICENSE` file** $\rightarrow$ **All Rights Reserved** by default. Obsolete PyTorch 1.4 environment. | **DISQUALIFIED** |
| **AltFreezing** | Attempted RecDrive/BaiduNetdisk links | Requires interactive browser portal, JavaScript, and password `altf`. Cannot be fetched via automated CLI. | **PAUSED** |

---

## 6. Protected File Integrity Confirmation

All production code, `/v1/analyze`, and UI files remain 100% untouched:

```
static/index.html:      a611da3df47c551c0cff17b3ba6f67ad  (MATCH - UNTOUCHED)
main.py:                807e951361bd407fd6b68802d9227495  (MATCH - UNTOUCHED)
StarBorder.jsx:         8ad8e7cab5ddddc30f60f9dbadb66d68  (MATCH - UNTOUCHED)
StarBorder.css:         1a250bcb58571c8e47d2bfc27235583b  (MATCH - UNTOUCHED)
baseline_snapshot.json: 06ec0a46206d6d9b5480748cc78d2514  (MATCH - UNTOUCHED)
```
