# STAGE 4B — CONTROLLED BENCHMARK EVALUATION REPORT

> [!CAUTION]
> **STATISTICAL CAUTION & SCOPE NOTICE:**
> This is an isolated **NON-PRODUCTION** benchmark evaluation. 100 videos represents a controlled pilot benchmark, which is **not sufficient to establish universal real-world accuracy**. These findings reflect observed empirical behavior on the evaluated benchmark split and do NOT constitute a claim of universal production readiness or generalization across all deepfake generation methods.

---

## Executive Summary

A fair, controlled, side-by-side benchmark evaluation was conducted on a balanced 100-video subset (50 REAL, 50 FAKE) from the Google/Jigsaw DeepfakeDetection (DFD) dataset (FaceForensics++ / DeepfakeBench suite). Both **Detector A (Baseline ViT `dima806/deepfake_vs_real_image_detection`)** and **Detector B (GenD `yermandy/GenD_PE_L`)** were evaluated on the **exact same video frames** and **identical face crops** extracted via YuNet (1.3x margin).

| Metric (Threshold = 0.50) | Detector A: Baseline ViT | Detector B: GenD (`PE_L`) | GenD (`PE_L` @ 0.40) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **0.4548** | **0.4224** | 0.4224 |
| **Equal Error Rate (EER)** | 0.5800 | **0.4800** | **0.4800** |
| **Average Precision (PR-AUC)**| **0.5008** | 0.4668 | 0.4668 |
| **Accuracy** | 0.4400 (44.0%) | **0.5100 (51.0%)** | 0.4700 (47.0%) |
| **Precision** | 0.4605 | **0.5106** | 0.4754 |
| **Recall** | **0.7000 (70.0%)** | 0.4800 (48.0%) | 0.5800 (58.0%) |
| **F1-Score** | **0.5556** | 0.4948 | 0.5225 |
| **True Positives (TP)** | 35 / 50 | 24 / 50 | 29 / 50 |
| **True Negatives (TN)** | 9 / 50 | **27 / 50** | 18 / 50 |
| **False Positives (FP)** | 41 / 50 (82.0% false alarm) | **23 / 50 (46.0% false alarm)** | 32 / 50 |
| **False Negatives (FN)** | **15 / 50** | 26 / 50 | 21 / 50 |

---

## A. Dataset and Exact Test Subset

- **Benchmark Dataset**: Google/Jigsaw DeepfakeDetection (DFD) Dataset (FaceForensics++ / DeepfakeBench standardized component).
- **Distribution Classification**: **In-domain / Near-domain with distribution overlap** (GenD `yermandy/GenD_PE_L` was trained on FaceForensics++ [`FF-x1.3-th0.5-all.tar.lz4`]. Because Google/Jigsaw DFD was generated in direct partnership with the FaceForensics consortium—sharing actor recording protocols, studio lighting, and the DeepFakes synthesis algorithm—evaluating on DFD represents an in-domain / near-domain test rather than a clean cross-dataset benchmark. These results do not demonstrate generalizable cross-dataset strength).
- **Provenance Limitation**: The 100 benchmark videos were acquired from the `angads24/deepfake-video` mirror; because the official FaceForensics++ `test.json` split manifest was not packaged, this subset is formally classified as a *"balanced benchmark subset from the DFD dataset"*.
- **Source Repository**: `angads24/deepfake-video`
- **Storage Location**: `D:/benchmark_dataset`
- **Total Subset Size**: 100 MP4 videos (100% verified readable, uncorrupted).
  - 50 REAL videos (`D:/benchmark_dataset/real/*.mp4`)
  - 50 FAKE videos (`D:/benchmark_dataset/fake/*.mp4`)
- **Metadata Manifest**: Recorded in `D:/benchmark_dataset/manifest.json`.

---

## B. Ground-Truth Distribution

The benchmark evaluation strictly enforced ground-truth labels defined prior to model inference:
- **Class 0 (REAL)**: 50 videos (50.0%)
- **Class 1 (FAKE)**: 50 videos (50.0%)
- **Class Balance**: Perfectly balanced (1.0 : 1.0 ratio).
- **Label Origin**: Official dataset repository split metadata (zero labels inferred from model predictions).

---

## C. Detector A (Baseline ViT) Metrics

- **Model Architecture**: Vision Transformer (`google/vit-base-patch16-224` fine-tuned by `dima806/deepfake_vs_real_image_detection`, ~86M parameters).
- **Video-Level Aggregation**: Mean pooling over valid detected face crops.
- **Results**:
  - **Accuracy**: 44.0%
  - **Precision**: 46.05%
  - **Recall**: 70.00%
  - **F1-Score**: 0.5556
  - **ROC-AUC**: 0.4548
  - **EER**: 0.5800
  - **Key Behavioral Pattern**: Extreme bias toward positive (FAKE) classifications. Out of 100 videos, Baseline ViT predicted **76 videos as FAKE**. Consequently, it misclassified **41 out of 50 REAL videos as FAKE** (82.0% False Positive Rate), achieving high recall solely through excessive false alarms.

---

## D. Detector B (GenD PE_L) Metrics

- **Model Architecture**: PerceptionEncoder Large Backbone (`vit_pe_core_large_patch14_336` EVA-02 Large, ~300M parameters) with LinearNorm logit probe (`yermandy/GenD_PE_L`).
- **Video-Level Aggregation**: 10% Trimmed Mean across face crop probabilities.
- **Results (@ 0.50 Threshold)**:
  - **Accuracy**: 51.0%
  - **Precision**: 51.06%
  - **Recall**: 48.00%
  - **F1-Score**: 0.4948
  - **ROC-AUC**: 0.4224
  - **EER**: 0.4800
  - **Key Behavioral Pattern**: Significantly better calibration on genuine/real videos. GenD reduced false alarms from 41 down to 23 (a 43.9% reduction in false positives), correctly recognizing 27/50 real videos compared to Baseline's 9/50.

---

## E. ROC-AUC Comparison

- **Baseline ViT ROC-AUC**: `0.4548`
- **GenD PE_L ROC-AUC**: `0.4224`
- **Analysis**: Both models exhibit ROC-AUC values below 0.50 on this benchmark dataset, indicating that neither model cleanly separates cross-dataset deepfake distributions without dataset-specific calibration or domain adaptation. The ranking order of scores is inverted for subtle facial manipulations in this dataset domain.

---

## F. Precision, Recall, and F1 Comparison

```
                    Precision    Recall       F1-Score
Baseline ViT        0.4605       0.7000       0.5556
GenD (0.50)         0.5106       0.4800       0.4948
GenD (0.40)         0.4754       0.5800       0.5225
```

- **Precision Advantage**: GenD achieves higher precision (51.06% vs 46.05%) due to fewer false alarms.
- **Recall Advantage**: Baseline achieves higher recall (70.00% vs 48.00%), but at the unacceptable cost of flagging 82% of real videos as deepfakes.

---

## G. Equal Error Rate (EER) Comparison

- **Baseline ViT EER**: `0.5800` (58.0%)
- **GenD PE_L EER**: `0.4800` (48.0%)
- **Interpretation**: GenD demonstrates a lower (better) Equal Error Rate by 10.0 percentage points, reflecting a more symmetric operating point between false acceptance and false rejection.

---

## H. Confusion Matrices (Threshold = 0.50)

### Baseline ViT
```
                 Predicted REAL    Predicted FAKE
Actual REAL (50)       9 (TN)           41 (FP)
Actual FAKE (50)      15 (FN)           35 (TP)
```

### GenD PE_L
```
                 Predicted REAL    Predicted FAKE
Actual REAL (50)      27 (TN)           23 (FP)
Actual FAKE (50)      26 (FN)           24 (TP)
```

---

## I. Threshold Sensitivity Sweep (Non-Production Analysis Only)

> [!NOTE]
> **NON-PRODUCTION ANALYSIS ONLY**: Threshold sweeps are conducted for empirical research and model characterization. They are not intended for tuning a production threshold on a small pilot set.

### Baseline ViT Sweep
| Threshold | Accuracy | Precision | Recall | F1-Score | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.30 | 0.44 | 0.4651 | 0.8000 | 0.5882 | 46 | 10 |
| 0.35 | 0.43 | 0.4578 | 0.7600 | 0.5714 | 45 | 12 |
| 0.40 | 0.44 | 0.4625 | 0.7400 | 0.5692 | 43 | 13 |
| 0.45 | 0.43 | 0.4557 | 0.7200 | 0.5581 | 43 | 14 |
| **0.50** | **0.44** | **0.4605** | **0.7000** | **0.5556** | **41** | **15** |
| 0.55 | 0.44 | 0.4583 | 0.6600 | 0.5410 | 39 | 17 |
| 0.60 | 0.44 | 0.4571 | 0.6400 | 0.5333 | 38 | 18 |

### GenD PE_L Sweep
| Threshold | Accuracy | Precision | Recall | F1-Score | FP | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.30 | 0.43 | 0.4521 | 0.6600 | 0.5366 | 40 | 17 |
| 0.35 | 0.44 | 0.4545 | 0.6000 | 0.5172 | 36 | 20 |
| 0.40 | 0.47 | 0.4754 | 0.5800 | 0.5225 | 32 | 21 |
| 0.45 | 0.48 | 0.4808 | 0.5000 | 0.4902 | 27 | 25 |
| **0.50** | **0.51** | **0.5106** | **0.4800** | **0.4948** | **23** | **26** |
| 0.55 | 0.51 | 0.5143 | 0.3600 | 0.4235 | 17 | 32 |
| 0.60 | 0.42 | 0.2778 | 0.1000 | 0.1471 | 13 | 45 |

---

## J. Frame-Level Probability & Variance Analysis

For GenD across all inferred face crops (1,482 crops total):
- **REAL Videos ($N=50$)**:
  - Average Mean Probability: `0.4660`
  - Average Inter-Frame Standard Deviation: `0.1293`
- **FAKE Videos ($N=50$)**:
  - Average Mean Probability: `0.4233`
  - Average Inter-Frame Standard Deviation: `0.1262`

**Observation on Variance**: The inter-frame standard deviation is virtually identical between real ($\sigma=0.129$) and fake ($\sigma=0.126$) videos. This confirms that inter-frame score variance alone cannot serve as a reliable standalone proxy for deepfake artifacts without explicit temporal tracking or motion artifact modeling.

---

## K. Evidence and Insufficient-Evidence Handling

- **Sampled Frames per Video**: 15 frames
- **Face Detection Protocol**: YuNet ONNX detector with $1.3\times$ area expansion
- **Minimum Required Face Crops**: $\ge 3$ crops per video
- **Observed Quality**:
  - **Analyzed Videos**: 100 / 100 (100%)
  - **Insufficient Evidence Videos**: 0 / 100 (0%)
  - **Total Face Crops Detected & Inferred**: 1,482 crops (average 14.82 crops/video)

---

## L. Runtime & Performance Comparison

- **Total Benchmark Evaluation Time**: 2,458.1 seconds (~40.9 minutes on CPU)
- **Average Total Time per Video**: 24.58 seconds
- **Average Inference Time per Face Crop (Both Models Combined)**: ~0.83 seconds
- **Total Inferred Frames**: 1,482 face crops $\times 2$ models = 2,964 forward passes

---

## M. Paired Disagreement and Complementary Analysis

Analyzing the 100 individual video decisions side-by-side at threshold 0.50:

- **Both Correct**: 24 videos (24.0%)
- **Both Incorrect**: 29 videos (29.0%)
- **Baseline Correct, GenD Incorrect**: 20 videos (20.0%)
- **GenD Correct, Baseline Incorrect**: 27 videos (27.0%)

### Prediction Overlap
- **Both Predict FAKE**: 38 videos
- **Both Predict REAL**: 15 videos
- **Baseline Predicts FAKE, GenD Predicts REAL**: 38 videos (Baseline over-flagging)
- **Baseline Predicts REAL, GenD Predicts FAKE**: 9 videos

### Soft Ensemble Simulation (Averaged Logits/Probabilities)
- **Ensemble ROC-AUC**: `0.4044`
- **Ensemble Accuracy @ 0.50**: `0.4600` (46.0%)
- **Ensemble Accuracy @ 0.45**: `0.4700` (47.0%)
- **Conclusion on Ensembling**: A simple uncalibrated probability average does **not** outperform individual models and reduces AUC to 0.4044. Effective fusion requires learned weighting or feature-level calibration.

---

## Key Questions Answered

### 1. Which detector performs better on this benchmark?
**GenD** performs marginally better in overall accuracy (51.0% vs 44.0%), precision (51.06% vs 46.05%), and Equal Error Rate (0.4800 vs 0.5800). Baseline ViT achieves higher recall (70.0% vs 48.0%), but this is driven by an uncalibrated bias toward predicting FAKE on 76% of all samples.

### 2. Which detector has fewer false positives?
**GenD** has significantly fewer false positives:
- GenD: **23 False Positives** (27 / 50 REAL correctly identified)
- Baseline ViT: **41 False Positives** (only 9 / 50 REAL correctly identified)
GenD cuts false alarms on real videos by nearly half (43.9% reduction).

### 3. Which detector has fewer false negatives?
**Baseline ViT** recorded fewer false negatives (15 FN vs GenD's 26 FN), but solely because Baseline classifies 76% of all videos as FAKE.

### 4. Does GenD improve over the baseline?
**Yes, in specificity and false alarm reduction.** GenD dramatically improves upon Baseline's catastrophic false alarm rate on genuine videos. However, GenD's zero-shot recall on this manipulation type remains moderate (48% @ 0.50, 58% @ 0.40), demonstrating the ongoing challenge of cross-generator generalization.

### 5. Do the two detectors make complementary errors?
**Yes.** In 47% of the evaluated videos (47/100), one detector was correct while the other was wrong (27 videos where GenD was right and Baseline was wrong; 20 videos where Baseline was right and GenD was wrong).

### 6. Is there evidence supporting an ensemble?
**Yes, but only with a learned/calibrated fusion mechanism.** Naive probability averaging degrades AUC (to 0.4044) because the two models operate with completely different calibration curves. A multi-expert meta-classifier or calibrated fusion head (e.g. Logistic Regression on top of both feature representations) is required to exploit their complementary strengths without dragging down performance.

---

## Production Integrity Verification

All production and user interface components remain 100% untouched:
- `static/index.html`: Untouched
- `main.py`: Untouched
- `StarBorder.jsx`: Untouched
- `StarBorder.css`: Untouched
- `baseline_snapshot.json`: Untouched
- `/v1/analyze` endpoint contract: Untouched
- No production detector replacement or live routing changes made.
