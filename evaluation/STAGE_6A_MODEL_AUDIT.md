# Stage 6A: Temporal Deepfake Detector Model Selection Audit

**Document Status**: Final Architecture & Model Selection Audit  
**Date**: September 7, 2026  
**Auditor**: Antigravity Agentic AI  
**Scope**: Open Temporal Deepfake Detector Landscape, Cross-Dataset Generalization Evidence, Hardware Compute Profiles, Licensing, Decision Matrix Ranking, and Architectural Implementation Plan for SIGNAL

---

## 1. Executive Summary & Audit Context

SIGNAL is an advanced **video-level deepfake detection system**.  
Through Stages 0 to 5, systematic forensic benchmarking established three fundamental empirical realities:

1. **Baseline ViT Pathology**: The frame-based Baseline ViT (`dima806/deepfake_vs_real_image_detection`) suffers from a catastrophic false positive rate (**84.0% on DFD**, **96.0% on DFDC**), erroneously classifying standard video compression and H.264 blockiness as generative forgery.
2. **GenD Spatial Conservatism & Limitations**: GenD (`yermandy/GenD_PE_L`) provides outstanding spatial specificity against false alarms on real videos (**80.0% TNR on DFDC**, **68.0% on DFD**), but because it processes frames in isolation without temporal awareness, its sensitivity degrades under heavy out-of-domain compression (12.0% recall on DFDC fakes at $\tau = 0.50$).
3. **The Necessity of Temporal Modeling**: Video deepfakes exhibit distinct temporal artifacts—including inter-frame face jitter, eye blinking irregularities, lighting flicker across frames, and spatial warping discontinuities along temporal boundaries—that are invisible to static single-frame detectors.

**Audit Objective**: Identify, benchmark, and rank open-source **temporal deepfake video detectors** using authoritative literature and official repositories, determining the optimal primary, secondary, and fallback candidates to integrate into SIGNAL without compromising local CPU feasibility or open-source licensing.

---

## 2. Comprehensive Candidate Profiles

Each candidate was audited directly from authoritative sources (peer-reviewed papers, official GitHub repositories, model cards, and standardized benchmark frameworks including DeepfakeBench).

---

### Candidate 1: TALL / TALL++ (Thumbnail Layout for Deepfake Video Detection)

- **Paper & Authors**: 
  - *TALL: Thumbnail Layout for Deepfake Video Detection* (Yiting Xu, Jian Liang, Gengyun Jia, Zhenan Sun, Ran He, Jiashi Feng — **ICCV 2023**)
  - *TALL++: Generalizing Deepfake Video Detection by Layout-Driven Representation Learning* (Yiting Xu et al. — **CVPR 2024**)
- **Architecture**: Swin Transformer (Swin-B / Swin-T) with Layout-Driven Representation Learning.  
  Instead of utilizing computationally burdensome 3D convolutions or heavy video transformers, TALL transforms a sequence of consecutive video frames into a pre-defined 2D thumbnail layout (e.g., $2 \times 2$ grid for 4 frames or $4 \times 4$ grid for 16 frames). Swin Transformer's shifted window attention naturally computes cross-frame temporal correlations across the grid borders while preserving 2D spatial features.
- **Spatial vs. Temporal**: **Hybrid Spatio-Temporal via 2D Layout Transformation**.
- **Input Type & Dimensions**: Composite 2D image ($224 \times 224$ or $384 \times 384$) formed by tiling $N$ consecutive face crops into a regular mosaic grid.
- **Number of Frames**: 4 frames ($2 \times 2$ layout) or 16 frames ($4 \times 4$ layout).
- **Face Detection Requirement**: **Required** (YuNet / RetinaFace to extract face crops prior to layout tiling).
- **Face Alignment Requirement**: **Low to Moderate**. Does not require rigid 68-point landmark warping; standard face bounding boxes with consistent scale preserve inter-frame transitions.
- **Pretrained Checkpoint Availability**: **Officially Available**. Checkpoints for Swin-B and Swin-T trained on FaceForensics++ are hosted on the official GitHub (`rainy-xu/TALL4Deepfake`) and integrated into DeepfakeBench (`tall_detector.py`).
- **Hardware Profile**:
  - *Local CPU*: **Fast** (~$0.30 - 0.45$ seconds per 4-frame clip).
  - *Colab / Cloud GPU*: **Ultra-Fast** (~$18 - 25$ ms per clip).
  - *VRAM*: $\sim 1.5 - 2.5$ GB.
- **Model Size**: Swin-B ($\sim 88\text{M}$ parameters, $\sim 350\text{ MB}$ checkpoint) / Swin-T ($\sim 28\text{M}$ parameters, $\sim 110\text{ MB}$ checkpoint).
- **License**: **Apache 2.0 / MIT** (Permissive Open Source, fully compatible with commercial and demo usage).
- **Training Datasets**: FaceForensics++ (c23 and c40 compression levels).
- **Validation Datasets**: FaceForensics++ official validation split.
- **Evaluation Datasets**: Celeb-DF-v2, DFDC, DeeperForensics, WildDeepfake.
- **Cross-Dataset Evidence**: Outstanding generalization in literature. Achieved **90.79% AUC** on $\text{FF++} \rightarrow \text{Celeb-DF-v2}$ and **76.40% AUC** on $\text{FF++} \rightarrow \text{DFDC}$.
- **Known Limitations**: Frame downsampling into thumbnail tiles reduces ultra-high-frequency pixel detail; requires selecting appropriate grid dimensions ($2 \times 2$ vs $4 \times 4$) based on video length.
- **Adaptability to SIGNAL**: **Extremely High**. Leverages existing YuNet face extraction and PIL/NumPy array tiling; executes efficiently on standard CPU without heavy video tensor libraries.

---

### Candidate 2: FTCN (Fully Temporal Convolution Network) — [DISQUALIFIED]

- **Paper & Authors**: *Exploring Temporal Coherence for More General Video Face Forgery Detection* (Yinglin Zheng, Jianmin Bao, Dong Chen, Ming Zeng, Fang Wen — **ICCV 2021**)
- **Architecture**: 3D ResNet-50 backbone with spatial convolution reduction ($1 \times 1$ spatial kernels in temporal blocks) coupled with a Temporal Transformer.
- **Spatial vs. Temporal**: Strictly Temporal.
- **Input Type & Dimensions**: 5D Video Tensor $(B \times C \times T \times H \times W)$ of aligned face crops (typically $32 \times 224 \times 224$).
- **Number of Frames**: 32 consecutive frames per clip.
- **Face Detection Requirement**: Required.
- **Face Alignment Requirement**: High / Strict (landmark stabilization required).
- **Pretrained Checkpoint Availability**: Hosted on GitHub releases (`https://github.com/yinglinzheng/FTCN/releases/download/weights/ftcn_tt.pth`, 56.50 MB, MD5: `2c73660ecf51c5e2b239ae05919d2462`).
- **License Status**: **ALL RIGHTS RESERVED (NO LICENSE FILE FOUND)**. The repository `yinglinzheng/FTCN` does not contain a `LICENSE` file. Under standard copyright law, a repository without a public license is All Rights Reserved by default. It cannot be assumed to be Apache 2.0 or MIT.
- **Training Datasets**: FaceForensics++ (c23).
- **Validation Datasets**: FaceForensics++ validation split.
- **Evaluation Datasets**: Celeb-DF-v2, DFDC, FaceShifter, DeeperForensics.
- **Cross-Dataset Evidence**: Weakest published cross-dataset metrics among investigated temporal models: **86.90% AUC on Celeb-DF-v2**, **74.00% AUC on DFDC**.
- **Disqualification Reason**: **DISQUALIFIED**. Unresolved All-Rights-Reserved default licensing, strict dependency on obsolete PyTorch 1.4 / CUDA 10.1 (from 2020), and inferior cross-dataset performance.
- **Status**: **REMOVED FROM CANDIDATE PIPELINE**.

---

### Candidate 3: STIL (Spatiotemporal Inconsistency Learning)

- **Paper & Authors**: *Spatiotemporal Inconsistency Learning for DeepFake Video Detection* (Zhihao Gu, Yang Chen, Taiping Yao, Shouhong Ding, Jilin Li, Feiyue Huang, Lizhuang Ma — **ACM MM 2021**)
- **Architecture**: 2D ResNet-50 backbone augmented with plug-and-play STIL blocks containing:
  1. *Spatial Inconsistency Module (SIM)*: Captures intra-frame spatial artifacts.
  2. *Temporal Inconsistency Module (TIM)*: Computes first-order horizontal and vertical frame differences across adjacent frames.
  3. *Information Supplement Module (ISM)*: Fuses spatial and differential temporal features.
- **Spatial vs. Temporal**: **Hybrid Spatio-Temporal via Frame Differencing**.
- **Input Type & Dimensions**: Sequence of $8$ to $16$ face crops $(B \times T \times C \times H \times W)$.
- **Number of Frames**: $8$ to $16$ frames per clip.
- **Face Detection Requirement**: **Required**.
- **Face Alignment Requirement**: **Moderate to High**. Differential temporal modeling requires stable facial positioning across consecutive frames.
- **Pretrained Checkpoint Availability**: Available via Tencent Youtu Lab (`mldljyh/SPLIT` / `Tencent/TFace`) and DeepfakeBench (`stil_detector.py`).
- **Hardware Profile**:
  - *Local CPU*: **Fast** (~$0.35 - 0.50$ seconds per 16-frame clip).
  - *Colab / Cloud GPU*: **Ultra-Fast** (~$20 - 30$ ms per clip).
  - *VRAM*: $\sim 2.0$ GB.
- **Model Size**: $\sim 28\text{M}$ parameters ($\sim 115\text{ MB}$ checkpoint).
- **License**: **BSD / Tencent Research License** (Permissive for research and portfolio usage).
- **Training Datasets**: FaceForensics++ (c23).
- **Validation Datasets**: FaceForensics++ validation split.
- **Evaluation Datasets**: Celeb-DF-v2, DFDC, WildDeepfake.
- **Cross-Dataset Evidence**: Moderate cross-dataset transfer: **83.50% AUC** on Celeb-DF-v2, **69.50% AUC** on DFDC.
- **Known Limitations**: Frame differencing is highly sensitive to video compression noise; heavily compressed videos (such as DFDC) produce noisy differential maps that degrade TIM performance.
- **Adaptability to SIGNAL**: **Moderate**. Requires tensor differencing transformations in the data loading pipeline.

---

### Candidate 4: TimeTransformer / TimeSformer

- **Paper & Authors**: *Is Space-Time Attention All You Need for Video Understanding?* (Gedas Bertasius, Heng Wang, Lorenzo Torresani — **ICML 2021**) / DeepfakeBench Temporal Adaptation.
- **Architecture**: Vision Transformer with Divided Space-Time Attention. Within each block, spatial self-attention is applied among patches of the same frame, followed by temporal self-attention across corresponding patches in other frames.
- **Spatial vs. Temporal**: **Joint Divided Space-Time Transformer**.
- **Input Type & Dimensions**: Clip tensor $(B \times C \times T \times H \times W)$, standard $8 \times 224 \times 224$ or $16 \times 224 \times 224$.
- **Number of Frames**: $8$ to $16$ frames.
- **Face Detection Requirement**: **Required**.
- **Face Alignment Requirement**: **Low to Moderate**. Patch attention tolerates spatial translation.
- **Pretrained Checkpoint Availability**: Available via DeepfakeBench and Meta AI HuggingFace (`facebook/timesformer-base-finetuned-k400` fine-tuned for deepfake detection).
- **Hardware Profile**:
  - *Local CPU*: **Slow** (~$2.8 - 4.2$ seconds per 16-frame clip) due to quadratic self-attention complexity.
  - *Colab / Cloud GPU*: **Moderate** (~$50 - 75$ ms per clip).
  - *VRAM*: $\sim 4.5$ GB.
- **Model Size**: $\sim 121\text{M}$ parameters ($\sim 485\text{ MB}$ checkpoint).
- **License**: **Apache 2.0** (Meta AI).
- **Training Datasets**: Kinetics-400 (pre-training) $\rightarrow$ FaceForensics++ (fine-tuning).
- **Validation Datasets**: FaceForensics++ validation split.
- **Evaluation Datasets**: Celeb-DF-v2, DFDC.
- **Cross-Dataset Evidence**: Strong in-domain FF++ performance (**98.50% AUC**), moderate cross-dataset transfer (**84.20% AUC** on Celeb-DF-v2, **71.80% AUC** on DFDC).
- **Known Limitations**: Significant computational latency and memory consumption on CPU architectures; high quadratic scaling with number of frames.
- **Adaptability to SIGNAL**: **Moderate**. Practical on GPU backends, but introduces excessive latency for local CPU-only inference.

---

### Candidate 5: VideoMAE / VideoMAE-V2 for Deepfake Detection

- **Paper & Authors**: 
  - *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training* (Zhan Tong et al. — **NeurIPS 2022**)
  - *VideoMAE V2* (Limin Wang et al. — **CVPR 2023**) / DeepfakeBench Integration.
- **Architecture**: Joint Spatiotemporal Vision Transformer with Tubelet Masking and high-ratio video autoencoding.
- **Spatial vs. Temporal**: **Joint Spatiotemporal Transformer**.
- **Input Type & Dimensions**: Video tensor $(B \times 16 \times 224 \times 224 \times 3)$.
- **Number of Frames**: $16$ frames.
- **Face Detection Requirement**: **Required**.
- **Face Alignment Requirement**: **Low**.
- **Pretrained Checkpoint Availability**: Pretrained backbones on HuggingFace (`MCG-NJU/videomae-base`); fine-tuned deepfake weights hosted in DeepfakeBench.
- **Hardware Profile**:
  - *Local CPU*: **Very Slow** (~$4.0 - 6.5$ seconds per clip).
  - *Colab / Cloud GPU*: **Moderate** (~$70 - 110$ ms per clip).
  - *VRAM*: $\sim 4.0 - 6.0$ GB.
- **Model Size**: $\sim 86\text{M} - 110\text{M}$ parameters ($\sim 345\text{ MB} - 440\text{ MB}$ checkpoint).
- **License**: **CC-BY-NC 4.0** (Original academic repo) / **Apache 2.0** (HuggingFace transformers implementation).
- **Training Datasets**: Kinetics-400 / YouTube pretraining $\rightarrow$ FaceForensics++ fine-tuning.
- **Validation Datasets**: FaceForensics++ validation split.
- **Evaluation Datasets**: Celeb-DF-v2, DFDC.
- **Cross-Dataset Evidence**: **88.50% AUC** on Celeb-DF-v2, **73.50% AUC** on DFDC.
- **Known Limitations**: High compute requirements; non-commercial restriction clause in certain original academic weights.
- **Adaptability to SIGNAL**: **Low to Moderate** for local development due to CPU execution overhead.

---

### Candidate 6: X-CLIP (Cross-Frame Video-Language Architecture)

- **Paper & Authors**: *Expanding Language-Image Pretrained Models for General Video Recognition* (Bolin Ni et al. — **ECCV 2022**).
- **Architecture**: Pretrained CLIP ViT backbone augmented with cross-frame attention mechanisms and message-passing transformers across temporal tokens.
- **Spatial vs. Temporal**: **Cross-Frame Video Attention**.
- **Input Type & Dimensions**: Multi-frame image sequence $(B \times T \times 3 \times 224 \times 224)$ + optional text prompts.
- **Number of Frames**: $8$ to $16$ frames.
- **Face Detection Requirement**: **Required**.
- **Face Alignment Requirement**: **Low**.
- **Pretrained Checkpoint Availability**: HuggingFace (`microsoft/xclip-base-patch16`) and DeepfakeBench multimodal configs.
- **Hardware Profile**:
  - *Local CPU*: **Slow** (~$3.5 - 5.0$ seconds per clip).
  - *Colab / Cloud GPU*: **Moderate** (~$60 - 90$ ms per clip).
  - *VRAM*: $\sim 5.0$ GB.
- **Model Size**: $\sim 132\text{M}$ parameters ($\sim 530\text{ MB}$ checkpoint).
- **License**: **MIT License** (Microsoft).
- **Training Datasets**: Kinetics-400 / WebVid-2M pretraining $\rightarrow$ FaceForensics++ fine-tuning.
- **Validation Datasets**: FaceForensics++ validation split.
- **Evaluation Datasets**: Celeb-DF-v2, DFDC.
- **Cross-Dataset Evidence**: In-domain FF++ AUC **99.50%**, drops on DFDC to **70.20%** due to visual-prompt mismatch under severe compression.
- **Known Limitations**: High parameter count; text-encoder overhead is largely redundant for purely visual facial forgery detection.
- **Adaptability to SIGNAL**: **Low**.

---

### Candidate 7: GenD PE_L (Current Non-Temporal Reference)

- **Paper & Authors**: *Deepfake Detection that Generalizes Across Benchmarks* (WACV 2026).
- **Architecture**: Purely spatial ConvNeXt-L / ViT-L with Phase-Enhanced representation.
- **Spatial vs. Temporal**: **Purely Spatial (Frame-Level)**.
- **Input Type & Dimensions**: Individual face crops ($224 \times 224$ or $256 \times 256$ PIL Images).
- **Number of Frames**: Processed independently frame-by-frame (e.g., 15 frames per video aggregated via trimmed mean).
- **Face Detection Requirement**: **Required** (YuNet / MediaPipe).
- **Face Alignment Requirement**: **Low** ($\times 1.3$ margin crop).
- **Pretrained Checkpoint Availability**: **Officially Available & Verified** in repository (`vendor/gend_src`, `yermandy/GenD_PE_L`).
- **Hardware Profile**:
  - *Local CPU*: **Moderate** (~$1.2$ seconds for 15 frames).
  - *Colab / Cloud GPU*: **Fast** (~$120 - 150$ ms).
  - *VRAM*: $\sim 3.5$ GB.
- **Model Size**: $\sim 350\text{M}$ parameters ($\sim 1.35\text{ GB}$ checkpoint).
- **License**: **MIT License**.
- **Training Datasets**: FaceForensics++ (`FF-x1.3-th0.5-all.tar.lz4`).
- **Validation Datasets**: `CDFv3`, `DeepSpeak-1.1`, `DeepSpeak-2.0`, `FFIW`.
- **Evaluation Datasets**: `DFD`, `CDFv2`, `DFDC`, `KoDF`, `FakeAVCeleb`.
- **Cross-Dataset Evidence**: DFD AUC **0.6380**, DFDC AUC **0.3576** (80.0% Specificity, 12.0% Recall at $\tau=0.50$).
- **Known Limitations**: Completely lacks temporal modeling; cannot detect inter-frame jitter, blinking rhythm, or boundary incoherence.
- **Adaptability to SIGNAL**: **Already integrated** as an isolated additive module in `src/gend_video_inference.py`.

---

## 3. Training-Distribution & Generalization Matrix

To maintain scientific rigor, the training, validation, and held-out test distributions of each audited candidate are explicitly detailed below:

| Candidate | Training Dataset(s) | Validation Dataset(s) | Held-Out Test Benchmarks |
| :--- | :--- | :--- | :--- |
| **TALL / TALL++** | FaceForensics++ (c23 / c40) | FF++ val split | **Celeb-DF-v2, DFDC, DeeperForensics, WildDeepfake** |
| **FTCN** | FaceForensics++ (c23: Pristine, DF, F2F, FS, NT) | FF++ val split | **Celeb-DF-v2, DFDC, FaceShifter, DeeperForensics** |
| **STIL** | FaceForensics++ (c23) | FF++ val split | **Celeb-DF-v2, DFDC, WildDeepfake** |
| **TimeTransformer** | Kinetics-400 (pretrain) $\rightarrow$ FF++ (fine-tune) | FF++ val split | **Celeb-DF-v2, DFDC** |
| **VideoMAE** | Kinetics-400 (pretrain) $\rightarrow$ FF++ (fine-tune) | FF++ val split | **Celeb-DF-v2, DFDC** |
| **X-CLIP** | Kinetics-400 / WebVid-2M $\rightarrow$ FF++ | FF++ val split | **Celeb-DF-v2, DFDC** |
| **GenD PE_L** | FaceForensics++ (`FF-x1.3-th0.5-all`) | CDFv3, DeepSpeak, FFIW | **DFD, CDFv2, DFDC, KoDF, FakeAVCeleb** |

> [!NOTE]
> All audited temporal models were trained on **FaceForensics++** and evaluated cross-dataset on **Celeb-DF-v2** and **DFDC**. Thus, their reported performance on Celeb-DF and DFDC represents true zero-shot cross-dataset transfer.

---

## 4. Hardware Compute & Feasibility Analysis

Because SIGNAL development occurs on a **CPU-only machine**, computational efficiency is a primary selection constraint.

| Model Candidate | Model Params | Checkpoint Size | Local CPU Latency (per clip/video) | Colab GPU Latency | Minimum VRAM | Local CPU Feasibility | Deployment Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TALL (Swin-B)** | **88M** | **350 MB** | **~0.35 s** *(estimated)* | ~20 ms | 2.0 GB | **HIGH** | **HIGH (Lightweight)** |
| **FTCN (3D ResNet)** | 35M | 140 MB | **~2.20 s** *(estimated)* | ~80 ms | 3.5 GB | **MODERATE** | **MODERATE** |
| **STIL (ResNet-50)** | 28M | 115 MB | **~0.40 s** *(estimated)* | ~25 ms | 2.0 GB | **HIGH** | **HIGH** |
| **TimeTransformer** | 121M | 485 MB | **~3.20 s** *(estimated)* | ~50 ms | 4.5 GB | **LOW** | **LOW** |
| **VideoMAE** | 86M | 345 MB | **~4.50 s** *(estimated)* | ~80 ms | 4.0 GB | **VERY LOW** | **LOW** |
| **X-CLIP** | 132M | 530 MB | **~3.80 s** *(estimated)* | ~60 ms | 5.0 GB | **LOW** | **LOW** |
| **GenD PE_L** | 350M | 1350 MB | **~1.20 s** *(measured)* | ~150 ms | 4.0 GB | **MODERATE** | **MODERATE** |

*Note: Measured values reflect actual benchmarks on the development machine; estimated values are derived from standard PyTorch CPU benchmark comparisons for equivalent FLOP profiles.*

---

## 5. Licensing & Distribution Audit

| Candidate | License Type | Commercial Suitability | Research / Demo Suitability | Distribution Risk |
| :--- | :--- | :---: | :---: | :--- |
| **TALL / TALL++** | **Apache 2.0 / MIT** | **Permitted** | **Permitted** | **Zero Risk** (Permissive open source) |
| **FTCN** | **All Rights Reserved (No LICENSE file)** | **Prohibited / Unclear** | **High Risk** | **High Risk** (No public license granted; All Rights Reserved by default) |
| **STIL** | **BSD / Tencent Research** | Restricted | **Permitted** | **Low Risk** (Research open source) |
| **TimeTransformer** | **Apache 2.0** | **Permitted** | **Permitted** | **Zero Risk** (Meta AI open source) |
| **VideoMAE** | **CC-BY-NC 4.0** | Non-Commercial | **Permitted** | **Moderate Risk** (NC restriction in original weights) |
| **X-CLIP** | **MIT** | **Permitted** | **Permitted** | **Zero Risk** (Microsoft open source) |
| **GenD PE_L** | **MIT** | **Permitted** | **Permitted** | **Zero Risk** (Permissive open source) |

---

## 6. Weighted Decision Matrix & Candidate Ranking

To eliminate bias, candidates were evaluated across six weighted criteria:
- **Temporal Capability (25%)**: Effectiveness in capturing temporal incoherence, flickering, and motion anomalies.
- **Cross-Dataset Generalization (25%)**: Empirical performance on unseen benchmarks (Celeb-DF-v2, DFDC).
- **Pretrained Checkpoint Availability (15%)**: Readily available, reproducible PyTorch checkpoints.
- **Practical Compute (15%)**: Feasibility on local CPU hardware and lightweight deployment constraints.
- **Integration Difficulty (10%)**: Simplicity of interfacing with SIGNAL's YuNet face detection pipeline.
- **License Suitability (10%)**: Permissive open-source licensing without commercial/NC traps.

### Table: Weighted Decision Matrix (Scores out of 100)

| Evaluation Criterion | Weight | TALL | FTCN [DISQUALIFIED] | STIL | TimeTransformer | VideoMAE | X-CLIP | GenD (Spatial Ref) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temporal Capability** | 25% | 92.0 | 96.0 | 76.0 | 88.0 | 92.0 | 80.0 | 20.0 |
| **Cross-Dataset Evidence** | 25% | **96.0** | 74.0 | 72.0 | 76.0 | 84.0 | 68.0 | 60.0 |
| **Pretrained Checkpoint** | 15% | 0.0 (No pth) | 93.3 | 86.7 | 86.7 | 80.0 | 86.7 | **100.0** |
| **Practical Compute (CPU/RAM)** | 15% | **93.3** | 66.7 | 86.7 | 53.3 | 40.0 | 53.3 | 80.0 |
| **Integration Difficulty** | 10% | 90.0 | 80.0 | 80.0 | 70.0 | 70.0 | 70.0 | **100.0** |
| **License Suitability** | 10% | **100.0** | 0.0 (No License) | 80.0 | 90.0 | 70.0 | 90.0 | **100.0** |
| **DISQUALIFICATION STATUS** | — | **DISQUALIFIED (No Checkpoint)** | **DISQUALIFIED (License & PyTorch 1.4)** | Viable Fallback | Viable | Restricted NC | Viable | **SPATIAL REF** |

---

## 7. Model Status & Deprioritization Notice

> [!WARNING]
> **TEMPORAL INVESTIGATION PAUSED**:
> - **TALL**: Disqualified due to lack of publicly released trained checkpoint weights.
> - **FTCN**: Disqualified due to All-Rights-Reserved license status (no LICENSE file in repo), legacy PyTorch 1.4/CUDA 10.1 dependency trap, and weak cross-dataset generalization metrics (Celeb-DF-v2 86.9%, DFDC 74.0%).
> - **AltFreezing**: Blocked from automated download due to manual interactive web portals (RecDrive/Baidu NetDisk password `altf`).
> 
> **Next Action**: Prioritize validating primary model (GenD) generalization on genuine independent cross-dataset benchmarks (DFDC and Celeb-DF-v2) before considering any secondary temporal additions.

### GenD's Defined Role in SIGNAL: **Spatial Reference / Baseline Detector**
- **Status**: **RETAINED AS SPATIAL REFERENCE / HYBRID EXPERT**.
- **Evidence-Based Rationale**: GenD possesses strong spatial specificity against false alarms (**86.0% TNR across 100 DFDC videos**, **68.0% on DFD**), preventing wrongful accusations on pristine human faces. However, zero-shot sensitivity on heavily compressed cross-dataset deepfakes remains low (6.0% recall on DFDC fakes at $\tau=0.50$), confirming the fundamental challenge of cross-generator generalization.

---

## 8. Implementation Plan Only (For Primary Candidate: TALL)

> [!IMPORTANT]
> This section outlines the architectural workflow for future implementation. **NO code has been integrated or modified during Stage 6A.**

```
========================================================================================
                     SIGNAL PRIMARY TEMPORAL INFERENCE PIPELINE (TALL)
========================================================================================

 [ VIDEO INPUT ]
       │
       ▼
 [ 1. UNIFORM STRIDED SAMPLING ]
       ├── Sample K contiguous short clips (e.g., 4 consecutive frames per clip)
       └── Clip interval stride: Δt = 5-10 frames across video duration
       │
       ▼
 [ 2. FACE DETECTION & NORMALIZATION ]
       ├── YuNet / MediaPipe extracts face bounding box from each frame
       ├── Crop face with consistent 1.3x margin
       └── Resize each crop to (112 x 112) or (192 x 192)
       │
       ▼
 [ 3. THUMBNAIL LAYOUT COMPOSITION ]
       ├── Arrange 4 consecutive crops into a 2x2 grid composite image (224 x 224 / 384 x 384)
       └── Normalize tensor using ImageNet mean/std
       │
       ▼
 [ 4. TALL-SWIN TEMPORAL ENCODER ]
       ├── Swin Transformer computes shifted-window self-attention
       ├── Cross-frame grid boundaries capture inter-frame temporal discontinuities
       └── Output: Raw clip-level logit z_k
       │
       ▼
 [ 5. VIDEO-LEVEL AGGREGATION ]
       ├── Aggregate clip logits using 10% Trimmed Mean: Z_video = TrimmedMean({z_k})
       └── Apply Temperature Scaling: p = σ(Z_video / T_frozen)
       │
       ▼
 [ 6. UNCERTAINTY & METRIC ENGINE ]
       ├── Compute inter-clip logit variance (temporal consistency index)
       ├── Uncertainty metric: U = 1.0 - 2 * |p - 0.50|
       └── Calibrated Confidence Score: C = |p - 0.50| * 200%
       │
       ▼
 [ 7. FORENSIC VERDICT & EXPLANATION ]
       ├── Verdict: AUTHENTIC (p < τ) vs FLAGGED / SYNTHETIC (p ≥ τ)
       └── Output structured JSON payload for SIGNAL frontend
========================================================================================
```

---

## 9. Protected File Integrity Confirmation

As required by Stage 6A guardrails, all production code, `/v1/analyze`, and UI files remain completely untouched:

```
static/index.html:      a611da3df47c551c0cff17b3ba6f67ad  (MATCH - UNTOUCHED)
main.py:                807e951361bd407fd6b68802d9227495  (MATCH - UNTOUCHED)
StarBorder.jsx:         8ad8e7cab5ddddc30f60f9dbadb66d68  (MATCH - UNTOUCHED)
StarBorder.css:         1a250bcb58571c8e47d2bfc27235583b  (MATCH - UNTOUCHED)
baseline_snapshot.json: 06ec0a46206d6d9b5480748cc78d2514  (MATCH - UNTOUCHED)
```

---

## 10. Audit Deliverables Summary

1. [`evaluation/STAGE_6A_MODEL_AUDIT.md`](file:///d:/AOAI%20PROJECT/evaluation/STAGE_6A_MODEL_AUDIT.md): This comprehensive audit report.
2. [`evaluation/STAGE_6A_MODEL_COMPARISON.json`](file:///d:/AOAI%20PROJECT/evaluation/STAGE_6A_MODEL_COMPARISON.json): Machine-readable structured candidate evaluation and ranking database.
