# STAGE 6H: Controlled Sampling Counterfactual Report

> [!IMPORTANT]
> **Scientific Research Protocol**: This study performs an isolated counterfactual experiment to determine whether the temporal sampling scheme (sequential early clips vs. distributed timeline clips) is the causal factor behind the LipForensics false-positive result on `videoplayback.mp4`. No production code, API routes, thresholds, or model weights have been altered.

## 1. Experimental Overview & Primary Case

- **Primary Test Video**: `videoplayback.mp4` (`C:\Users\hp\Downloads\videoplayback.mp4`)
- **Ground Truth**: **`REAL`** (User-Verified Real In-The-Wild Video)
- **Video Properties**: `640x352`, `5019` total frames, `30.0 FPS`, `167.3s` duration
- **Face & Mouth ROI Extraction**: `5019/5019` valid mouth crops (`100.0%` detection rate)
- **Pretrained Architecture**: Official LipForensics (CVPR 2021) 3D Conv + ResNet-18 + MS-TCN (`lipforensics_ff.pth`)
- **Decision Threshold**: `0.50` (Fake probability >= 0.50 -> FAKE)

---

## 2. Part A: Counterfactual Sampling Strategy Results

### Master Strategy Comparison Table

| Strategy | Clip 1 | Clip 2 | Clip 3 | Clip 4 | Mean Logit | Fake Probability | Verdict | Correct? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Strategy A (Current Production Sequential)** | 0.9226 (L: +2.48) | 0.4564 (L: -0.17) | 0.9966 (L: +5.69) | 1.0000 (L: +10.96) | `+4.7400` | **`0.9913`** | **`FAKE`** | NO (False Positive) |
| **Strategy B (Distributed 5%, 30%, 55%, 80%)** | 0.9995 (L: +7.53) | 1.0000 (L: +10.49) | 0.7365 (L: +1.03) | 0.9913 (L: +4.74) | `+5.9455` | **`0.9974`** | **`FAKE`** | NO (False Positive) |

---

### Detailed Per-Clip Windows & Logits

#### Strategy A: Current Production Sampling (Sequential 0..99 frames)
- **Clip 1**: Frames `0..24` (Timestamp: `0.00s`) — Valid Crops: `25`, Raw Logit: `+2.4776`, Sigmoid Fake Prob: `0.9226`
- **Clip 2**: Frames `25..49` (Timestamp: `0.83s`) — Valid Crops: `25`, Raw Logit: `-0.1747`, Sigmoid Fake Prob: `0.4564`
- **Clip 3**: Frames `50..74` (Timestamp: `1.67s`) — Valid Crops: `25`, Raw Logit: `+5.6947`, Sigmoid Fake Prob: `0.9966`
- **Clip 4**: Frames `75..99` (Timestamp: `2.50s`) — Valid Crops: `25`, Raw Logit: `+10.9624`, Sigmoid Fake Prob: `1.0000`

#### Strategy B: Distributed Sampling (5%, 30%, 55%, 80%)
- **Clip 1 (5% timeline)**: Center Crop `251`, Frames `239..263` (Timestamp: `7.97s`) — Valid Crops: `25`, Raw Logit: `+7.5275`, Sigmoid Fake Prob: `0.9995`
- **Clip 2 (30% timeline)**: Center Crop `1505`, Frames `1493..1517` (Timestamp: `49.77s`) — Valid Crops: `25`, Raw Logit: `+10.4901`, Sigmoid Fake Prob: `1.0000`
- **Clip 3 (55% timeline)**: Center Crop `2760`, Frames `2748..2772` (Timestamp: `91.60s`) — Valid Crops: `25`, Raw Logit: `+1.0278`, Sigmoid Fake Prob: `0.7365`
- **Clip 4 (80% timeline)**: Center Crop `4014`, Frames `4002..4026` (Timestamp: `133.40s`) — Valid Crops: `25`, Raw Logit: `+4.7367`, Sigmoid Fake Prob: `0.9913`

---

## 3. Explicit Counterfactual Comparison

1. **Absolute Score Difference**: `|0.9974 - 0.9913| = 0.0061` (0.61% score shift)
2. **Verdict Change**: **NO** (`FAKE -> FAKE`)
3. **Outlier & Variance Analysis**:
   - In Strategy A (sequential early clips), Clip 4 is an extreme high-logit outlier (+10.9624, prob 1.0000) while Clip 2 is negative (-0.1747, prob 0.4564), yielding standard deviation sigma = 4.150. In Strategy B (distributed clips across 5%, 30%, 55%, 80%), logits are: Clip 1 (5%): +7.5275 (p=0.9995), Clip 2 (30%): +10.4901 (p=1.0000), Clip 3 (55%): +1.0278 (p=0.7365), Clip 4 (80%): +4.7367 (p=0.9913), with sigma = 3.493.

---

## 4. Scientific Framework: Observations, Hypothesis, Evidence & Conclusions

### A. Observations
1. In Strategy A (first 100 frames / 3.33s), Clip 1 (L=+2.48, P=0.9226), Clip 3 (L=+5.69, P=0.9966), and Clip 4 (L=+10.96, P=1.0000) produce strong positive synthetic logits.
2. In Strategy B (distributed across 167.3s timeline), logits across 5%, 30%, 55%, and 80% are: Clip 1 (+7.5275), Clip 2 (+10.4901), Clip 3 (+1.0278), and Clip 4 (+4.7367).
3. Mean logit shifts from `+4.7400` (Strategy A) to `+5.9455` (Strategy B).

### B. Hypothesis
- **Hypothesis**: The high false-positive probability (P=0.9913) under current production sampling was caused by localized early temporal artifacts in frames 0..100 (speech onset / motion cadence) rather than intrinsic video-wide properties. Distributing sample windows evenly across the timeline tests whether temporal distribution mitigates early local outlier bias.

### C. Evidence
- The absolute score difference between the two sampling schemes is **0.0061**.
- Under Strategy A (sequential), LipForensics outputs **FAKE** (P=0.9913).
- Under Strategy B (distributed), LipForensics outputs **FAKE** (P=0.9974).
- The verdict transition is **FAKE -> FAKE**.

### D. Scientific Conclusions
1. **Causal Attribution**: Temporal sampling alone does not eliminate the false-positive classification on this specific video. The model's 3D CNN + MS-TCN mouth dynamics remain sensitive to compression blockiness, non-frontal speaking angles, or acoustic-phoneme divergence present across multiple segments of the recording.
2. **Production Integrity Notice**: In accordance with experimental guardrails, no production code, thresholds, or deployment pipelines are modified based on this diagnostic inquiry.

---

## 5. Part B: Prepared Paired 10-Video Benchmark Specification

> [!NOTE]
> **Part B Status**: Code and dataset manifest have been prepared and validated in `evaluation/run_stage6h_part_b_benchmark.py`. In strict compliance with instructions, **Part B has NOT been automatically executed**.

### Prepared Dataset Manifest (10 Genuinely New Labeled Videos)

| ID | Filename | Ground Truth | Role |
| :--- | :--- | :---: | :--- |
| `new_real_01` | `01__outside_talking_pan_laughing.mp4` | **`REAL`** | New Unseen Authentic Test Video |
| `new_real_02` | `01__outside_talking_still_laughing.mp4` | **`REAL`** | New Unseen Authentic Test Video |
| `new_real_03` | `01__podium_speech_happy.mp4` | **`REAL`** | New Unseen Authentic Test Video |
| `new_real_04` | `01__secret_conversation.mp4` | **`REAL`** | New Unseen Authentic Test Video |
| `new_real_05` | `01__talking_against_wall.mp4` | **`REAL`** | New Unseen Authentic Test Video |
| `new_fake_01` | `01_11__meeting_serious__9OM3VE0Y.mp4` | **`FAKE`** | New Unseen Manipulated Test Video |
| `new_fake_02` | `01_11__secret_conversation__4OJNJLOO.mp4` | **`FAKE`** | New Unseen Manipulated Test Video |
| `new_fake_03` | `01_11__talking_against_wall__9229VVZ3.mp4` | **`FAKE`** | New Unseen Manipulated Test Video |
| `new_fake_04` | `01_11__walking_outside_cafe_disgusted__FAFWDR4W.mp4` | **`FAKE`** | New Unseen Manipulated Test Video |
| `new_fake_05` | `01_12__outside_talking_pan_laughing__TNI7KUZ6.mp4` | **`FAKE`** | New Unseen Manipulated Test Video |

### Prepared Execution Pipeline
- **Script**: `evaluation/run_stage6h_part_b_benchmark.py`
- **Execution Method**: Fresh evaluation under BOTH Strategy A (Sequential) and Strategy B (Distributed) for each of the 10 videos (20 total forward passes).
- **Primary Case Exclusion**: `videoplayback.mp4` remains isolated as the primary single case and is not conflated with the 10-video benchmark.
