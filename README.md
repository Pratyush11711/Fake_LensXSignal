# FakeLens × SIGNAL: Multimodal Misinformation & Deepfake Video Detection

> **Unified Forensic Intelligence Platform**  
> **Stream A (NLP Text Classification)**: TF-IDF Feature Extraction + Logistic Regression (ISOT Benchmark, 99.14% held-out accuracy)  
> **Stream B (Computer Vision Deepfake Detection)**: OpenCV YuNet Face Tracking + Pretrained Vision Transformer / LipForensics (CVPR 2021)  
> **Policy**: Calibrated Provisional 3-Tier Forensic Verification Policy  
> **Tagline**: *"Detect misinformation in text. Uncover deepfakes in video. Understand the signal."*

---

## Table of Contents
1. [Platform Overview](#1-platform-overview)
2. [Dual-Stream Architecture](#2-dual-stream-architecture)
3. [NLP Misinformation Pipeline (FakeLens Core)](#3-nlp-misinformation-pipeline-fakelens-core)
4. [Deepfake Video Detection Pipeline (SIGNAL Integration)](#4-deepfake-video-detection-pipeline-signal-integration)
5. [Provisional 3-Tier Forensic Policy](#5-provisional-3-tier-forensic-policy)
6. [REST API Reference](#6-rest-api-reference)
7. [Installation & Setup](#7-installation--setup)
8. [Running the Application](#8-running-the-application)
9. [Running Automated Tests](#9-running-automated-tests)
10. [Directory Structure](#10-project-directory-structure)
11. [Scientific & Forensic Disclaimer](#11-scientific--forensic-disclaimer)

---

## 1. Platform Overview

**FakeLens × SIGNAL** is an end-to-end multimodal verification and forensic analysis platform. Originally built as an academic NLP fake news classification suite, FakeLens now integrates the complete facial and temporal forensic capabilities of **SIGNAL** (a deepfake video detector).

The platform equips researchers, journalists, and forensic investigators with:
- **Article & Headline NLP Verification**: Instant classification of unstructured text as **FAKE** or **REAL**, displaying exact directional n-gram coefficients driving the decision.
- **Deepfake Video Analysis**: Frame-by-frame spatio-temporal facial analysis of uploaded video evidence (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`) using OpenCV YuNet face detection and deep Vision Transformer / LipForensics architectures.
- **Interactive Forensic Viewfinder HUD**: Live video preview showing animated tracking bounding boxes, 5-point facial landmark crosshairs, and scanlines.
- **Frame-by-Frame Anomaly Waveform**: Interactive temporal bar chart illustrating exactly which sampled frames triggered forensic anomaly spikes.
- **Forensic PDF Dossier Export**: Automatic generation of court- and review-ready PDF reports with case IDs, quantitative telemetry, per-frame anomaly breakdowns, and timestamps.
- **Dataset Leakage & Academic Rigor Audit**: Isolation of stylistic publisher artifacts (e.g., Reuters datelines) and strict deduplication benchmarks.

---

## 2. Dual-Stream Architecture

```
                            MULTIMODAL EVIDENCE INTAKE
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
[ STREAM A: NLP TEXT PIPELINE ]                           [ STREAM B: SIGNAL VIDEO PIPELINE ]
  │                                                         │
  ├─ 1. Text Normalization & Cleaning                       ├─ 1. Video Ingestion & Format Validation
  │    (HTML/URL removal, lowercase, datelines)             │    (MP4/MOV/AVI/WEBM, <= 100MB)
  │                                                         │
  ├─ 2. TF-IDF Feature Extraction                           ├─ 2. YuNet ONNX Face Tracking (320x320)
  │    (Unigrams + Bigrams, 5000 max features)              │    (Spatio-temporal single-subject tracking)
  │                                                         │
  ├─ 3. Logistic Regression Classifier                      ├─ 3. Affine Alignment & Mouth ROI Sequence
  │    (Linear decision boundary: z = w·x + b)              │    (Normalized sequence: 88x88, mean 0.421, std 0.165)
  │                                                         │
  └─ 4. Calibrated Sigmoid Class Probability                ├─ 4. Spatio-Temporal Neural Inference
       ├─ P(y=1) -> FAKE vs REAL                            │    (Vision Transformer / ResNet-18 + MS-TCN)
       └─ Feature Contribution Coefficients                 │
                                                            └─ 5. Provisional 3-Tier Policy Calibration
                                                                 ├─ < 0.40  -> LIKELY AUTHENTIC
                                                                 ├─ 0.40–0.90 -> INCONCLUSIVE (NEEDS REVIEW)
                                                                 └─ >= 0.90 -> SIGNS OF MANIPULATION
```

---

## 3. NLP Misinformation Pipeline (FakeLens Core)

- **Dataset**: Kaggle / ISOT Fake and Real News Benchmark (39,105 cleaned articles: 21,197 Real, 17,908 Fake).
- **Preprocessing (`model/preprocess.py`)**: Strips HTML tags, email formats, and URLs. Normalizes whitespace, hyphens, and quotes. Provides configurable dateline removal (`strip_source=True`).
- **Feature Extraction**: Scikit-Learn `TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)`.
- **Classification**: `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)`.
- **Evaluation Performance on 20% Held-Out Test Set (7,821 articles)**:
  - **Accuracy**: 99.14%
  - **Precision (Fake)**: 99.35%
  - **Recall (Fake)**: 98.77%
  - **F1 Score**: 99.06%
  - **Confusion Matrix**: True Negatives: 4,216 | False Positives: 23 | False Negatives: 44 | True Positives: 3,538.

---

## 4. Deepfake Video Detection Pipeline (SIGNAL Integration)

- **Face Detection (`models/face_detection_yunet_2023mar.onnx`)**: High-performance OpenCV DNN FaceDetectorYN operating on sampled video keyframes.
- **Facial Alignment (`src/alignment/face_aligner.py`)**: Computes 5-point affine transformation matrix to align eyes and mouth to canonical coordinates.
- **Model Backends**:
  1. **Baseline ViT (`src/baseline/baseline_vit.py`)**: Pretrained Vision Transformer (`dima806/deepfake_vs_real_image_detection`) evaluating frame-level facial artifact probability.
  2. **LipForensics (`src/lipforensics_video_inference.py`)**: CVPR 2021 architecture composed of a 3D convolutional frontend, ResNet-18 spatial trunk, and Multi-Scale Temporal Convolutional Network (MS-TCN) tracking phonetic and mouth motion anomalies.
  3. **Temporal Xception (`src/pipeline/inference_engine.py`)**: Learned temporal attention forward pass across 16-frame sequences.
- **Upload Protection**: Enforces strict `MAX_UPLOAD_MB` limits (default 100MB) with streaming writes and guaranteed cleanup in a `finally` block to prevent orphaned files.

---

## 5. Provisional 3-Tier Forensic Policy

Rather than forcing brittle binary decisions on noisy media, SIGNAL applies calibrated forensic confidence bands:

| Anomaly Score Band | Assessment Tier | Forensic Interpretation |
| :--- | :--- | :--- |
| **`Score < 0.40`** | **`LIKELY AUTHENTIC`** | Low forensic anomaly signal detected. Spatio-temporal consistency metrics align with natural recording parameters. |
| **`0.40 <= Score < 0.90`** | **`INCONCLUSIVE — NEEDS REVIEW`** | Ambiguous forensic signal. Compression artifacts, extreme motion blur, or poor lighting can elevate scores. Human review recommended. |
| **`Score >= 0.90`** | **`SIGNS OF POSSIBLE MANIPULATION`** | High anomaly signal detected across consecutive sampled facial frames. Independent corroboration is recommended. |

---

## 6. REST API Reference

### 1. NLP Text Classification
- **Endpoint**: `POST /predict` (or `/api/predict`)
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  { "text": "Washington (Reuters) - Senate approves budget bill..." }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "prediction": "REAL",
    "confidence": 98.42,
    "fake_probability": 0.0158,
    "real_probability": 0.9842,
    "features": [
      { "term": "reuters", "coefficient": -21.32, "weight": "-0.45", "influence": "REAL" }
    ]
  }
  ```

### 2. Deepfake Video Detection
- **Endpoint**: `POST /detect` (or `/v1/analyze`, `/api/deepfake/detect`)
- **Headers**: `multipart/form-data`
- **Form Parameter**: `video` (file: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`)
- **Response**:
  ```json
  {
    "status": "ANALYSIS_COMPLETE",
    "result": "REAL",
    "detection_score": 0.0382,
    "assessment_tier": "LIKELY AUTHENTIC",
    "assessment_message": "Low forensic anomaly signal detected...",
    "confidence": "96.18%",
    "frames_analyzed": 16,
    "frame_scores": [0.03, 0.04, 0.02, 0.05, 0.03, 0.04],
    "execution_time": 1.28
  }
  ```

### 3. System Health & Telemetry
- **Endpoint**: `GET /health` (or `/api/health`, `/api/status`)
- **Response**:
  ```json
  {
    "status": "ok",
    "active_backend": "baseline",
    "model_loaded": true,
    "nlp_loaded": true,
    "face_detector_loaded": true,
    "max_upload_mb": 100
  }
  ```

---

## 7. Installation & Setup

1. **Clone repository & open directory**:
   ```bash
   cd d:\NLP\FakeLens-NLP-main
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 8. Running the Application

### Option A: Flask Backend (Default)
```bash
python backend/app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

### Option B: FastAPI Backend (High-Concurrency Async)
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 9. Running Automated Tests

Run the full unified test suite (both NLP pipeline and Deepfake Video detection):

```bash
# Run all pytest suites (Deepfake API, Face tracking, ViT inference)
pytest

# Run Flask NLP pipeline unit tests
python -m unittest tests/test_pipeline.py

# Run Flask Deepfake API unit tests
python -m unittest tests/test_flask_deepfake.py
```

---

## 10. Project Directory Structure

```
FakeLens-NLP-main/
├── backend/
│   └── app.py                      # Unified Flask backend application (NLP + Deepfake endpoints)
├── main.py                         # Unified FastAPI async application
├── wsgi.py                         # Production WSGI entrypoint
├── requirements.txt                # Unified dependencies (Flask, PyTorch, Transformers, OpenCV, Scikit-Learn)
├── .env.example                    # Environment variable configurations
│
├── models/                         # Computer vision & deepfake weights
│   ├── face_detection_yunet_2023mar.onnx  # OpenCV YuNet CNN face detector
│   └── face_landmarker.task        # MediaPipe facial landmarker asset
│
├── detectors/                      # Isolated research detector wrappers
│   ├── gend_detector.py            # GenD PerceptionEncoder wrapper
│   └── lipforensics_detector.py    # LipForensics 3D Conv + MS-TCN model
│
├── src/                            # Deepfake computer vision pipelines
│   ├── baseline/baseline_vit.py    # 2D Vision Transformer pipeline
│   ├── alignment/face_aligner.py   # 5-point affine transformation
│   ├── detection/yunet_detector.py # YuNet inference wrapper
│   ├── pipeline/inference_engine.py# Temporal Xception inference engine
│   └── lipforensics_video_inference.py # CVPR 2021 LipForensics video runner
│
├── model/                          # NLP fake news classification assets
│   ├── preprocess.py               # Text cleaning & feature contribution explainability
│   ├── logistic_regression.pkl     # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl        # Trained TF-IDF vectorizer
│   ├── metrics.json                # Evaluated test metrics
│   └── dataset_stats.json          # ISOT corpus distributions
│
├── static/                         # Web assets
│   ├── script.js                   # Client application logic (Text & Deepfake Video UI, jsPDF)
│   └── style.css                   # Unified editorial forensic styling
│
├── templates/
│   └── index.html                  # Single-page multimodal interface with Optical Viewfinder HUD
│
├── tests/                          # Automated test suites
│   ├── conftest.py                 # Pytest mocks & testclient fixtures
│   ├── test_pipeline.py            # NLP unit tests
│   ├── test_flask_deepfake.py      # Flask deepfake API unit tests
│   ├── test_deepfake_api.py        # FastAPI deepfake unit tests
│   └── test_face_detector.py       # YuNet face tracking tests
│
└── vendor/                         # External research references (DeepfakeBench, GenD)
```

---

## 11. Scientific & Forensic Disclaimer

- **NLP Classification Scope**: FakeLens evaluates text stylistic and lexical resemblance to articles in the ISOT benchmark dataset. It does not replace independent fact-checking against live investigative sources.
- **Deepfake Video Boundaries**: Video compression, motion blur, and poor camera lighting can elevate forensic anomaly scores. A high score flags media for expert human inspection rather than proving malicious alteration in isolation.
#   F a k e _ L e n s X S i g n a l  
 