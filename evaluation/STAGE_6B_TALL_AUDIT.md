# Stage 6B: TALL / TALL++ Compatibility & Checkpoint Provenance Audit (Corrected)

**Document Status**: Final Corrected Audit  
**Date**: September 7, 2026  
**Auditor**: Antigravity Agentic AI  
**Scope**: Checkpoint Provenance Gap Resolution, Checkpoint Availability Audit, Training Domain Leakage Analysis, and Candidate Deprioritization

---

## 1. Executive Summary & Checkpoint Provenance Resolution

Following a comprehensive audit of official sources for **TALL (Thumbnail Layout for Deepfake Video Detection)** and **TALL++**, the following critical findings are established:

> [!CAUTION]
> **CORRECTION OF CHECKPOINT AVAILABILITY STATUS**:  
> In initial architectural reviews, TALL was presumed to have plug-and-play public weights based on literature references. **Direct investigation of the official repository (`https://github.com/rainy-xu/TALL4Deepfake`) and associated hosting platforms confirms that NO trained `.pth` checkpoint is publicly downloadable or hosted.**  
> The audit conclusion is officially updated from *"verified public weights"* to:  
> **"NO CONFIRMED TRAINED CHECKPOINT LOCATED"**.

---

## 2. Checkpoint Availability Audit Details

### A. Official Repository Audit (`rainy-xu/TALL4Deepfake`):
- **Releases Tab**: No tagged releases exist on GitHub.
- **Repository Contents**: Contains source code for data preprocessing, network architectures, and training/evaluation scripts.
- **Weight Hosting**: There are **no active Google Drive, Baidu Pan, Hugging Face, or direct download links** for pretrained model checkpoints (`.pth` / `.pt`) provided in the README or repository files.
- **Community Status**: Multiple unresolved GitHub issues from researchers requesting pretrained weights have gone unanswered.

### B. Feasibility Impact for SIGNAL:
- To use TALL, the model would have to be **trained completely from scratch** on FaceForensics++ (requiring hundreds of gigabytes of raw video, multiple high-end enterprise GPUs, and hundreds of compute hours).
- Training from scratch is strictly **out of scope** under the project's zero-cost, CPU/free-tier resource constraints.
- **Verdict**: **TALL is DEPRIORITIZED and DISQUALIFIED** as a viable candidate for plug-and-play integration.

---

## 3. Alternative Temporal Candidates: Checkpoint Re-Evaluation

With TALL disqualified due to the lack of downloadable pretrained weights, alternative temporal deepfake detectors must be prioritized based on **actual, verified checkpoint availability**:

| Candidate | Paper / Venue | Official Repository | Checkpoint Availability Status | License |
| :--- | :--- | :--- | :--- | :--- |
| **AltFreezing** | *AltFreezing: Dual-branch Spatiotemporal Feature Learning* (AAAI 2023) | `https://github.com/ZhendongWang6/AltFreezing` | **Available**: Download links provided via RecDrive / BaiduDrive (`checkpoints/`) and DeepfakeBench. | Permissive Open Source |
| **FTCN** | *Exploring Temporal Coherence for More General Video Face Forgery Detection* (ICCV 2021) | `https://github.com/yinglinzheng/FTCN` | **Available**: Official checkpoint (`FTCN+TT` on FF++) linked in README; requires legacy PyTorch environment. | MIT License |
| **STIL** | *Spatiotemporal Inconsistency Learning* (ACM MM 2021) | `https://github.com/mldljyh/SPLIT` / `TFace` | **Partially Available**: Code in Tencent TFace / DeepfakeBench. | BSD License |

**Recommendation**: Pivot immediately to re-attempting the **AltFreezing** and **FTCN** compatibility and checkpoint-availability audit.

---

## 4. Training-Distribution & Domain Leakage Audit (FF++ vs. DFD)

Regardless of model choice, the relationship between training distributions and evaluation benchmarks must be rigorously documented:

```
========================================================================================================
                               DATASET DISTRIBUTION & LEAKAGE MAP
========================================================================================================

[ FACEFORENSICS++ ECOSYSTEM (TRAINING SET FOR MOST TEMPORAL DETECTORS) ]
  ├── FaceForensics++ (FF++): 1,000 pristine videos + 4 manipulation methods (DF, F2F, FS, NT)
  └── Google/Jigsaw DeepfakeDetection (DFD):
      ├── Generated in partnership with the FaceForensics consortium.
      ├── Shares overlapping capture environments, actor protocols, and DeepFakes synthesis tools.
      └── Status: EVALUATION ON DFD IS NEAR-DOMAIN / IN-DOMAIN (HIGH LEAKAGE RISK).

[ INDEPENDENT BENCHMARKS (TRUE ZERO-SHOT CROSS-DATASET EVALUATION) ]
  ├── DFDC (DeepFake Detection Challenge): Completely independent actors, wild compression, diverse algorithms.
  └── Celeb-DF-v2: Independent high-quality synthesis pipeline, unseen celebrities.
========================================================================================================
```

### Key Takeaway:
- Any model trained on **FaceForensics++** (including TALL, AltFreezing, FTCN, and GenD) evaluated against the 100-video **DFD** dataset cannot be considered a clean cross-dataset benchmark.
- Genuine cross-dataset generalization must be measured strictly on independent benchmarks like **DFDC** (Stage 5) and **Celeb-DF-v2**.

---

## 5. Summary & Action Plan

1. **TALL Disqualification**: Confirmed no downloadable trained weights exist; TALL is deprioritized.
2. **Next Steps**: Re-evaluate **AltFreezing** and **FTCN** for verifiable checkpoint downloads, file hashes, and environment isolation.
3. **Guardrail Compliance**: Zero production, UI, or baseline files were modified.
