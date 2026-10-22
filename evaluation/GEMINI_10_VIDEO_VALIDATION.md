# Gemini 3.6 Flash vs. LipForensics: 10-Video (+1 Primary) Controlled Validation

> [!IMPORTANT]
> **Diagnostic Decision Experiment**: This controlled validation evaluates Google Gemini 3.6 Flash against the production LipForensics temporal baseline across a balanced diagnostic set of 11 videos (6 REAL, 5 FAKE), including the primary user-verified in-the-wild video `videoplayback.mp4`. No production code, thresholds, or pipelines have been altered.

## 1. Primary Case Analysis: `videoplayback.mp4`

- **Video File**: `videoplayback.mp4` (`C:\Users\hp\Downloads\videoplayback.mp4`)
- **Ground Truth**: **REAL** (User-Verified In-The-Wild Video)
- **LipForensics Score**: `0.9913` (Verdict: **FAKE** — False Positive)
- **Gemini 3.6 Flash Verdict**: **`REAL`** (Confidence: `0.75` — True Negative / Correct)
- **Gemini Evidence**: *"Consistent facial motion, natural lip-syncing, uniform lighting, and seamless physical integration with the guitar throughout the clip."*
- **Gemini Timestamps**: `NONE`

---

## 2. Master Comparison Table

| Video | Ground Truth | LipForensics Score | LipForensics Verdict | Gemini Verdict | Gemini Confidence | LipForensics Correct? | Gemini Correct? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `videoplayback.mp4` (Primary) | **REAL** | 0.9913 | **FAKE** | **REAL** | 0.75 | ❌ NO | ✅ YES |
| `01__exit_phone_room.mp4` | **REAL** | 0.9857 | **FAKE** | **REAL** | 0.68 | ❌ NO | ✅ YES |
| `01__kitchen_still.mp4` | **REAL** | 0.0001 | **REAL** | **REAL** | 0.70 | ✅ YES | ✅ YES |
| `01__meeting_serious.mp4` | **REAL** | 0.3848 | **REAL** | **REAL** | 0.65 | ✅ YES | ✅ YES |
| `01__hugging_happy.mp4` | **REAL** | 0.8548 | **FAKE** | **REAL** | 0.72 | ❌ NO | ✅ YES |
| `01__kitchen_pan.mp4` | **REAL** | 0.0058 | **REAL** | **REAL** | 0.65 | ✅ YES | ✅ YES |
| `01_02__outside_talking_still_laughing__YVGY8LOK.mp4` | **FAKE** | 0.4363 | **REAL** | **REAL** | 0.78 | ❌ NO | ❌ NO |
| `01_02__walk_down_hall_angry__YVGY8LOK.mp4` | **FAKE** | 0.9031 | **FAKE** | **REAL** | 0.73 | ✅ YES | ❌ NO |
| `01_03__podium_speech_happy__480LQD1C.mp4` | **FAKE** | 0.9994 | **FAKE** | **REAL** | 0.75 | ✅ YES | ❌ NO |
| `01_03__talking_against_wall__JZUXXFRB.mp4` | **FAKE** | 0.2237 | **REAL** | **REAL** | 0.68 | ❌ NO | ❌ NO |
| `01_03__hugging_happy__ISF9SP4G.mp4` | **FAKE** | 0.9724 | **FAKE** | **REAL** | 0.72 | ✅ YES | ❌ NO |

---

## 3. Metric Comparison: LipForensics vs. Gemini 3.6 Flash

| Metric | LipForensics (CVPR 2021) | Gemini 3.6 Flash (Multimodal) |
| :--- | :---: | :---: |
| **Total Videos Evaluated** | 11 | 11 |
| **Known REAL False Positives (FP)** | **3 / 6** (3/6 (50.0%)) | **0 / 6** (0/6 (0.0%)) |
| **Known REAL Correct Negatives (TN)** | **3 / 6** (3/6 (50.0%)) | **6 / 6** (6/6 (100.0%)) |
| **Known FAKE True Positives (TP)** | **3 / 5** (3/5 (60.0%)) | **0 / 5** (0/5 (0.0%)) |
| **Known FAKE False Negatives (FN)** | **2 / 5** (2/5 (40.0%)) | **5 / 5** (5/5 (100.0%)) |
| **Accuracy** | **54.5%** | **54.5%** |
| **Precision** | **50.0%** | **0.0%** |
| **Recall (Sensitivity)** | **60.0%** | **0.0%** |
| **Specificity (TNR)** | **50.0%** | **100.0%** |

> [!CAUTION]
> **Statistical Notice**: These metrics are derived from a focused 11-video diagnostic set. They do not constitute an empirical claim of universal superiority across arbitrary datasets, but demonstrate distinct structural differences in forensic reasoning vs. specialized spatio-temporal convolutions.

---

## 4. Per-Video Detailed Forensic Evidence

### 1. `videoplayback.mp4` (REAL)
- **Description**: User-verified real in-the-wild video
- **LipForensics**: Score = `0.9913` | Verdict = **FAKE** (Incorrect)
- **Gemini 3.6 Flash**: Verdict = **REAL** | Confidence = `0.75` | Time = `39.25s` (Correct)
- **Gemini Timestamps**: `NONE`

**Gemini Reasoning Output:**
```text
Consistent facial motion, natural lip-syncing, uniform lighting, and seamless physical integration with the guitar throughout the clip.
```

### 2. `01__exit_phone_room.mp4` (REAL)
- **Description**: Benchmark Authentic (01__exit_phone_room.mp4)
- **LipForensics**: Score = `0.9857` | Verdict = **FAKE** (Incorrect)
- **Gemini 3.6 Flash**: Verdict = **REAL** | Confidence = `0.68` | Time = `36.53s` (Correct)
- **Gemini Timestamps**: `NONE`

**Gemini Reasoning Output:**
```text
The subject moves naturally through the hallway with consistent facial lighting, natural motion blur, and stable features. No temporal warping, edge artifacts, or blending inconsistencies around the face or background are observed.
```

### 3. `01__kitchen_still.mp4` (REAL)
- **Description**: Benchmark Authentic (01__kitchen_still.mp4)
- **LipForensics**: Score = `0.0001` | Verdict = **REAL** (Correct)
- **Gemini 3.6 Flash**: Verdict = **REAL** | Confidence = `0.7` | Time = `128.61s` (Correct)
- **Gemini Timestamps**: `NONE`

**Gemini Reasoning Output:**
```text
Facial structure, skin textures, and lip movements remain naturally consistent across frames. Lighting aligns with the environment, and no temporal jitter or boundary warping artifacts are present.
```

### 4. `01__meeting_serious.mp4` (REAL)
- **Description**: Benchmark Authentic (01__meeting_serious.mp4)
- **LipForensics**: Score = `0.3848` | Verdict = **REAL** (Correct)
- **Gemini 3.6 Flash**: Verdict = **REAL** | Confidence = `0.65` | Time = `46.8s` (Correct)
- **Gemini Timestamps**: `NONE`

**Gemini Reasoning Output:**
```text
The video demonstrates consistent facial structure, natural lip movements, and uniform lighting across all participants throughout the interaction. No boundary warping, facial tearing, or temporal jitter was detected.
```

### 5. `01__hugging_happy.mp4` (REAL)
- **Description**: Benchmark Authentic (01__hugging_happy.mp4)
- **LipForensics**: Score = `0.8548` | Verdict = **FAKE** (Incorrect)
- **Gemini 3.6 Flash**: Verdict = **REAL** | Confidence = `0.72` | Time = `33.92s` (Correct)
- **Gemini Timestamps**: `NONE`

**Gemini Reasoning Output:**
```text
Facial features, lighting, and physical interactions remain natural and consistent throughout the scene with no visible synthetic blending, edge distortion, or temporal artifacts.
```

### 6. `01__kitchen_pan.mp4` (REAL)
- **Description**: Benchmark Authentic (01__kitchen_pan.mp4)
- **LipForensics**: Score = `0.0058` | Verdict = **REAL** (Correct)
- **Gemini 3.6 Flash**: Verdict = **REAL** | Confidence = `0.65` | Time = `40.71s` (Correct)
- **Gemini Timestamps**: `NONE`

**Gemini Reasoning Output:**
```text
The video exhibits consistent facial features, natural lighting reflections, smooth expressions, and sharp face boundaries without visual flickering, warping, or blending artifacts.
```

### 7. `01_02__outside_talking_still_laughing__YVGY8LOK.mp4` (FAKE)
- **Description**: Benchmark Manipulated Deepfakes (01_02__outside_talking)
- **LipForensics**: Score = `0.4363` | Verdict = **REAL** (Incorrect)
- **Gemini 3.6 Flash**: Verdict = **REAL** | Confidence = `0.78` | Time = `47.25s` (Incorrect)
- **Gemini Timestamps**: `NONE`

**Gemini Reasoning Output:**
```text
Consistent facial features, natural lip movements, and smooth temporal transitions throughout the clip. Lighting, skin textures, and body movements show no signs of digital warping or deepfake manipulation.
```

### 8. `01_02__walk_down_hall_angry__YVGY8LOK.mp4` (FAKE)
- **Description**: Benchmark Manipulated Deepfakes (01_02__walk_down_hall)
- **LipForensics**: Score = `0.9031` | Verdict = **FAKE** (Correct)
- **Gemini 3.6 Flash**: Verdict = **REAL** | Confidence = `0.73` | Time = `35.49s` (Incorrect)
- **Gemini Timestamps**: `NONE`

**Gemini Reasoning Output:**
```text
Facial features, lighting, and skin textures remain consistent throughout the sequence. Boundary lines around the face and hair show no blurring, warping, or swapping artifacts.
```

### 9. `01_03__podium_speech_happy__480LQD1C.mp4` (FAKE)
- **Description**: Benchmark Manipulated FaceSwap (01_03__podium_speech)
- **LipForensics**: Score = `0.9994` | Verdict = **FAKE** (Correct)
- **Gemini 3.6 Flash**: Verdict = **REAL** | Confidence = `0.75` | Time = `46.26s` (Incorrect)
- **Gemini Timestamps**: `NONE`

**Gemini Reasoning Output:**
```text
Facial features, lighting, and expressions remain temporally consistent throughout the sequence without visual warping, edge artifacts, or unnatural lip-sync behavior.
```

### 10. `01_03__talking_against_wall__JZUXXFRB.mp4` (FAKE)
- **Description**: Benchmark Manipulated FaceSwap (01_03__talking_against_wall)
- **LipForensics**: Score = `0.2237` | Verdict = **REAL** (Incorrect)
- **Gemini 3.6 Flash**: Verdict = **REAL** | Confidence = `0.68` | Time = `32.15s` (Incorrect)
- **Gemini Timestamps**: `NONE`

**Gemini Reasoning Output:**
```text
Facial movements, lip sync, and lighting are natural and consistent throughout. No boundary warping, distortion, or temporal artifacts are observed.
```

### 11. `01_03__hugging_happy__ISF9SP4G.mp4` (FAKE)
- **Description**: Benchmark Manipulated FaceSwap (01_03__hugging_happy)
- **LipForensics**: Score = `0.9724` | Verdict = **FAKE** (Correct)
- **Gemini 3.6 Flash**: Verdict = **REAL** | Confidence = `0.72` | Time = `51.09s` (Incorrect)
- **Gemini Timestamps**: `NONE`

**Gemini Reasoning Output:**
```text
Facial features, lighting, and physical interactions remain temporally consistent throughout the sequence. Expression transitions, hand contact, and edge boundaries show natural motion with no signs of deepfake synthesis or manipulation.
```

---

## 5. Architectural Conclusions & Guardrails

1. **Mouth ROI Temporal Bottleneck vs. Holistic Reasoning**:
   - LipForensics inspects an isolated 88x88 grayscale mouth ROI over 25-frame windows. Highly expressive mouth shapes, natural speech pauses, or compression blockiness trigger high fake logits.
   - Gemini 3.6 Flash performs holistic full-frame multimodal scene evaluation, checking lighting continuity, skin texture blending, physical object interaction, and temporal integrity.

2. **Production Integrity Maintained**:
   - No changes have been applied to `main.py`, `/detect`, `/v1/analyze`, or production model pipelines.
   - The Gemini API key remains isolated in the execution environment and is never exposed or logged.
