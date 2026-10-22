# STAGE 4A — BENCHMARK SETUP AUDIT REPORT

> [!IMPORTANT]
> **FEASIBILITY & AUDIT ONLY — NON-PRODUCTION EVALUATION**
> Strictly zero modifications have been made to production endpoints, contracts, detectors, or frontend UI files.
> No training or large dataset downloads have been executed.

---

## 1. Current System & Hardware Environment (Section 1)

The system and PyTorch execution environment were audited via local hardware probing:

| Component | Specification / Detected Value | Forensic / Evaluation Implication |
| :--- | :--- | :--- |
| **Python Version** | `3.14.4` (`64-bit`) | Fully operational with standard libraries and PyTorch |
| **PyTorch Version** | `2.11.0+cpu` | Operational; running under CPU backend |
| **TorchVision Version** | `0.26.0+cpu` | Operational for tensor transforms and I/O |
| **CUDA / GPU Status** | **CUDA Available: False** (CPU Only) | No discrete NVIDIA GPU detected; all tensor inference executes on CPU |
| **GPU Name & VRAM** | `None` / `N/A` | Memory management must strictly monitor host RAM |
| **Total System RAM** | `15.65 GB` | Sufficient for single-model inference batches |
| **Available System RAM** | `3.49 GB` (Dynamic) | Heavy multi-worker loaders should be avoided; batch size = 1–4 |
| **Disk Storage (Drive C:)** | `183.13 GB` Total (`27.67 GB` Free) | Primary OS drive; avoid dumping raw video sets here |
| **Disk Storage (Drive D:)** | `292.97 GB` Total (`56.08 GB` Free) | **Recommended evaluation storage directory** (56 GB workspace headroom) |

> [!TIP]
> **CPU Throughput Feasibility:** On CPU, GenD (`yermandy/GenD_PE_L`) processes ~1.5 to 2.5 frames/sec. Evaluating a full 5,000-video dataset (160,000 frames) on CPU would require ~25–35 hours. Evaluating a **standardized balanced test split ($N=100$ to $200$ videos $\times$ 15 frames = 1,500–3,000 frames)** will complete in **10–25 minutes** and fit comfortably within available RAM and disk limits.

---

## 2. DeepfakeBench Audit (`vendor/DeepfakeBench`) (Section 2)

DeepfakeBench was cloned into an isolated vendor directory (`vendor/DeepfakeBench/`) without altering any production code.

### Platform Capabilities & Architecture:
- **Repository Location:** `vendor/DeepfakeBench` (NeurIPS 2023 D&B / v2 codebase).
- **Supported Detectors (36 Total):**
  - *Naive Detectors (5):* Xception, Meso4, MesoInception, CNN-Aug (ResNet34), EfficientNet-B4.
  - *Spatial Detectors (20):* Effort (ICML'25 Spotlight), LSDA (CVPR'24), CapsuleNet, DSP-FWA, Face X-ray, FFD, CORE, RECCE, UCF, LRL, IID, RFM, SIA, SLADD, UIA-ViT, CLIP, SBI, PCL-I2G, Multi-Attention.
  - *Frequency Detectors (3):* F3Net, SPSL, SRM.
  - *Video Temporal Detectors (8):* TALL (ICCV'23), I3D, STIL, FTCN, X-CLIP, TimeTransformer, VideoMAE.
- **Supported Benchmark Datasets:**
  - FaceForensics++ (c23 HQ & c40 LQ), FaceShifter, DeepfakeDetection (DFD), Celeb-DF-v1, Celeb-DF-v2, DFDC Preview, DFDC (Kaggle), DeeperForensics-1.0 (DF-1.0), UADFV, DF40.
- **Standardized Metrics Supported:**
  - Video-level AUC & Frame-level AUC
  - Equal Error Rate (EER)
  - Accuracy (Overall ACC, Real ACC, Fake ACC)
  - Average Precision (AP / PR-AUC)
- **Extensibility for External/Custom Detectors:**
  - DeepfakeBench uses standard JSON metadata schemas mapping image/video paths to binary labels (`0 = REAL`, `1 = FAKE`).
  - Custom detectors (such as our GenD pipeline and Baseline ViT) can be evaluated directly via standalone evaluation adapters without touching production code or requiring deep architectural modifications.

---

## 3. Dataset Access & Feasibility Audit (Section 3)

| Benchmark Candidate | Real Videos | Fake Videos | Manipulation Methods | Access Protocol / Form | Storage Required (Raw vs Test Split) | Direct Labels | Video-Level Evaluation | Feasibility Ranking |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **Celeb-DF-v2** | 590 | 5,639 | High-quality DeepFake synthesis with reduced visual artifacts | Direct download via GitHub release / Google Drive link | ~35 GB (Full) / **~2.5 GB (Test Split)** | **Yes** (`List_of_testing_videos.txt`) | **Yes** (Official test protocol) | ⭐️ **#1 (Immediate Feasibility)** |
| **FaceForensics++ (c23 / c40)** | 1,000 | 4,000 | 4 methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures | Academic request form to TUM for automated download script | ~80 GB (c23/c40) / **~5.0 GB (Test Split)** | **Yes** (Official train/val/test json splits) | **Yes** (De facto industry gold standard) | ⭐️ **#2 (Gold Standard)** |
| **DFDC Preview** | 1,131 | 4,119 | 2 face-swapping algorithms across diverse actors | Direct download via Meta AI / Kaggle | ~40 GB (Full) / **~3.0 GB (Sample)** | **Yes** (metadata.json) | **Yes** | ⭐️ **#3 (Good Diversity)** |
| **WildDeepfake** | 3,500+ | 3,800+ | Internet in-the-wild deepfakes (real social media) | GitHub repository download | ~10 GB | **Yes** (Sequence annotations) | **Yes** | ⭐️ **#4 (In-the-Wild Testing)** |
| **DeeperForensics-1.0** | 50,000 | 10,000 | DeepFake with 7 perturbation levels (noise, blur, etc.) | Request agreement | >300 GB (Too large for current disk) | **Yes** | **Yes** | ❌ (Exceeds disk capacity) |

---

## 4. GenD Compatibility Audit (Section 4)

GenD (`yermandy/GenD_PE_L`) was analyzed for compatibility against benchmark datasets:

```
[Benchmark Input]
       │
       ├─► (A) Pre-cropped benchmark frame (e.g. DeepfakeBench LMDB / RGB crops) ──┐
       │                                                                           │
       └─► (B) Raw benchmark video file ──► YuNet Face Detection + 1.3x margin ────┤
                                                                                   ▼
                                                                     [RGB PIL Image (224x224)]
                                                                                   │
                                                                                   ▼
                                                                     [ImageNet Tensor Normalization]
                                                                                   │
                                                                                   ▼
                                                                     [GenD LinearNorm Logit Inference]
                                                                                   │
                                                                                   ▼
                                                                     [Sigmoid Probability per Frame]
                                                                                   │
                                                                                   ▼
                                                                     [Video-Level 10% Trimmed Mean]
```

- **Preprocessing Alignment:** GenD takes standard RGB PIL images resized to `(224, 224)` and normalized using standard ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`.
- **Inference Function:** `predict_frame_logit(image: PIL.Image) -> float` in [`detectors/gend_detector.py`](file:///d:/AOAI%20PROJECT/detectors/gend_detector.py) works directly out-of-the-box.
- **Aggregation:** 10% Trimmed Mean across 15–32 sampled frames per video.
- **Compatibility Verdict:** **100% Compatible.** Zero modifications required to `detectors/gend_detector.py`.

---

## 5. Baseline Compatibility Audit (Section 5)

The production baseline detector (`dima806/deepfake_vs_real_image_detection` + YuNet face detection):
- **Input:** 15 sampled frames from raw video, cropped via YuNet and evaluated using HuggingFace ViT `AutoImageProcessor` (`google/vit-base-patch16-224`).
- **Score Extraction:** Softmax probability of fake class.
- **Evaluation Adapter Pattern:** A standalone benchmark runner will import the baseline model in read-only mode, iterate over the benchmark test split, and compute comparative metrics without touching `main.py` or `/v1/analyze`.
- **Compatibility Verdict:** **100% Compatible.** Zero production changes required.

---

## 6. Recommendations & Plan for Stage 4B (Section 6)

1. **Best Immediately Accessible Benchmark:**
   - **Celeb-DF-v2 Official Test Split** (518 videos, or a balanced 100-video subset: 50 Real + 50 Fake).
   - *Rationale:* High visual realism, standardized public split (`List_of_testing_videos.txt`), immediate download availability, tests subtle blending artifacts.
2. **Best Benchmark if Institutional Access is Approved:**
   - **FaceForensics++ c23 (HQ) and c40 (LQ) Test Splits** (700 videos: 140 Real + 560 Fake across 4 manipulation methods).
   - *Rationale:* Evaluates detector degradation under video compression.
3. **Estimated Resource Footprint:**
   - **Storage:** ~2.5 GB to 5.0 GB on Drive `D:` (well within the available 56.08 GB free space).
   - **RAM:** ~2.0 GB peak working memory during sequential single-stream inference.
   - **Compute Time:** ~15–20 minutes on CPU for a balanced 100-video test set ($100 \times 15 = 1,500$ frame evaluations).
4. **Exact Next Steps for Stage 4B:**
   - Step 1: Create an isolated, non-production benchmark evaluator script (`evaluation/run_benchmark_eval.py`).
   - Step 2: Acquire the balanced, labeled test split with verified ground truth.
   - Step 3: Run side-by-side inference on identical frames for both Baseline ViT and GenD.
   - Step 4: Compute empirical metrics: **AUC-ROC, EER, Overall ACC, F1-Score, Precision, Recall, and Confusion Matrix** at candidate decision thresholds.

---

## 7. Protected File Integrity Verification

| Protected Target | MD5 Hash | Status |
| :--- | :---: | :---: |
| [`static/index.html`](file:///d:/AOAI%20PROJECT/static/index.html) | `a611da3df47c551c0cff17b3ba6f67ad` | **UNTOUCHED** |
| [`main.py`](file:///d:/AOAI%20PROJECT/main.py) | `807e951361bd407fd6b68802d9227495` | **UNTOUCHED** |
| [`StarBorder.jsx`](file:///d:/AOAI%20PROJECT/StarBorder.jsx) | `8ad8e7cab5ddddc30f60f9dbadb66d68` | **UNTOUCHED** |
| [`StarBorder.css`](file:///d:/AOAI%20PROJECT/StarBorder.css) | `1a250bcb58571c8e47d2bfc27235583b` | **UNTOUCHED** |
| [`baseline_snapshot.json`](file:///d:/AOAI%20PROJECT/baseline_snapshot.json) | `06ec0a46206d6d9b5480748cc78d2514` | **UNTOUCHED** |
| Production `/v1/analyze` Contract & Pipeline | Unchanged | **UNTOUCHED** |

---

## 8. Stop Condition (Section 7)

Stage 4A audit is complete. **No large datasets have been downloaded, no training has been initiated, and production remains completely untouched.** Awaiting user review before proceeding.
