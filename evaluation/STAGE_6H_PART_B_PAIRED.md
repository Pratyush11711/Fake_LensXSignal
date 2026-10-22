# STAGE 6H PART B: Paired Sampling Validation Report

> [!IMPORTANT]
> **Diagnostic Validation Notice**: This report presents a paired experimental evaluation of LipForensics (CVPR 2021) across 10 genuinely new benchmark videos (5 REAL, 5 FAKE) comparing **Current Sequential Production Sampling** vs. **Distributed Sampling (5%, 30%, 55%, 80%)**. Both strategies were evaluated from scratch under identical preprocessing and model weights. These findings are exploratory ($N=10$) and do not modify production code or configurations.

## 1. Master Paired Comparison Table

| Video | Ground Truth | Sequential Score | Sequential Verdict | Distributed Score | Distributed Verdict | Score Delta (B - A) | Verdict Changed? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `01__outside_talking_pan_laughing.mp4` | **REAL** | 0.7318 | **FAKE** | 0.1740 | **REAL** | `-0.5578` | **⚡ YES** |
| `01__outside_talking_still_laughing.mp4` | **REAL** | 0.0500 | **REAL** | 0.6098 | **FAKE** | `+0.5598` | **⚡ YES** |
| `01__podium_speech_happy.mp4` | **REAL** | 0.0138 | **REAL** | 0.0003 | **REAL** | `-0.0136` | **NO** |
| `01__secret_conversation.mp4` | **REAL** | 0.1890 | **REAL** | 0.9763 | **FAKE** | `+0.7873` | **⚡ YES** |
| `01__talking_against_wall.mp4` | **REAL** | 0.0098 | **REAL** | 0.0003 | **REAL** | `-0.0096` | **NO** |
| `01_11__meeting_serious__9OM3VE0Y.mp4` | **FAKE** | 0.6295 | **FAKE** | 0.5809 | **FAKE** | `-0.0487` | **NO** |
| `01_11__secret_conversation__4OJNJLOO.mp4` | **FAKE** | 0.6841 | **FAKE** | 0.7913 | **FAKE** | `+0.1072` | **NO** |
| `01_11__talking_against_wall__9229VVZ3.mp4` | **FAKE** | 0.9088 | **FAKE** | 0.9983 | **FAKE** | `+0.0895` | **NO** |
| `01_11__walking_outside_cafe_disgusted__FAFWDR4W.mp4` | **FAKE** | 1.0000 | **FAKE** | 0.8734 | **FAKE** | `-0.1266` | **NO** |
| `01_12__outside_talking_pan_laughing__TNI7KUZ6.mp4` | **FAKE** | 0.9995 | **FAKE** | 0.9997 | **FAKE** | `+0.0001` | **NO** |

---

## 2. Aggregate Metric Comparison

| Metric | Strategy A: Sequential Production | Strategy B: Distributed (5%, 30%, 55%, 80%) | Delta (B - A) |
| :--- | :---: | :---: | :---: |
| **REAL False-Positive Rate (FPR)** | **1/5 (20.0%)** | **2/5 (40.0%)** | `+20.0%` |
| **REAL Specificity (TNR)** | **4/5 (80.0%)** | **3/5 (60.0%)** | `-20.0%` |
| **FAKE True-Positive Rate (TPR / Recall)** | **5/5 (100.0%)** | **5/5 (100.0%)** | `+0.0%` |
| **FAKE False-Negative Rate (FNR)** | **0/5 (0.0%)** | **0/5 (0.0%)** | `+0.0%` |
| **Accuracy** | **90.0%** (9/10) | **80.0%** (8/10) | `-10.0%` |
| **Precision** | **83.3%** | **71.4%** | `-11.9%` |
| **Recall** | **100.0%** | **100.0%** | `+0.0%` |
| **F1 Score** | **0.9091** | **0.8333** | `-0.0758` |

---

## 3. Per-Video Detailed Clip Breakdown

### 1. `01__outside_talking_pan_laughing.mp4` (REAL)
- **Video Properties**: 1920x1080, 626 frames, 26.08s duration, 619 valid mouth crops (98.9%)
- **Sequential Sampling (A)**:
  - Mean Logit: `+1.0038` -> Fake Prob: `0.7318` -> Verdict: **FAKE** (Incorrect)
    - Clip 1 (Frames 4..28, 0.167s): Logit `+7.0961`, Prob `0.9992`
    - Clip 2 (Frames 29..53, 1.208s): Logit `-0.5935`, Prob `0.3558`
    - Clip 3 (Frames 54..78, 2.25s): Logit `-3.4387`, Prob `0.0311`
    - Clip 4 (Frames 79..103, 3.292s): Logit `+0.9511`, Prob `0.7213`
- **Distributed Sampling (B)**:
  - Mean Logit: `-1.5576` -> Fake Prob: `0.1740` -> Verdict: **REAL** (Correct)
    - Clip 1 (5%, Frames 23..47, 0.958s): Logit `-0.7242`, Prob `0.3265`
    - Clip 2 (30%, Frames 177..201, 7.375s): Logit `-1.5426`, Prob `0.1762`
    - Clip 3 (55%, Frames 332..356, 13.833s): Logit `-2.1652`, Prob `0.1029`
    - Clip 4 (80%, Frames 486..510, 20.25s): Logit `-1.7984`, Prob `0.1420`
- **Score Delta (B - A)**: `-0.5578` (Verdict Changed: **YES**)

### 2. `01__outside_talking_still_laughing.mp4` (REAL)
- **Video Properties**: 1920x1080, 851 frames, 35.46s duration, 851 valid mouth crops (100.0%)
- **Sequential Sampling (A)**:
  - Mean Logit: `-2.9435` -> Fake Prob: `0.0500` -> Verdict: **REAL** (Correct)
    - Clip 1 (Frames 0..24, 0.0s): Logit `-11.6381`, Prob `0.0000`
    - Clip 2 (Frames 25..49, 1.042s): Logit `-0.1764`, Prob `0.4560`
    - Clip 3 (Frames 50..74, 2.083s): Logit `+1.6051`, Prob `0.8327`
    - Clip 4 (Frames 75..99, 3.125s): Logit `-1.5647`, Prob `0.1730`
- **Distributed Sampling (B)**:
  - Mean Logit: `+0.4466` -> Fake Prob: `0.6098` -> Verdict: **FAKE** (Incorrect)
    - Clip 1 (5%, Frames 30..54, 1.25s): Logit `-4.0276`, Prob `0.0175`
    - Clip 2 (30%, Frames 243..267, 10.125s): Logit `-0.4917`, Prob `0.3795`
    - Clip 3 (55%, Frames 456..480, 19.0s): Logit `+4.0109`, Prob `0.9822`
    - Clip 4 (80%, Frames 668..692, 27.833s): Logit `+2.2947`, Prob `0.9084`
- **Score Delta (B - A)**: `+0.5598` (Verdict Changed: **YES**)

### 3. `01__podium_speech_happy.mp4` (REAL)
- **Video Properties**: 1920x1080, 902 frames, 37.58s duration, 902 valid mouth crops (100.0%)
- **Sequential Sampling (A)**:
  - Mean Logit: `-4.2664` -> Fake Prob: `0.0138` -> Verdict: **REAL** (Correct)
    - Clip 1 (Frames 0..24, 0.0s): Logit `-5.2406`, Prob `0.0053`
    - Clip 2 (Frames 25..49, 1.042s): Logit `-4.9655`, Prob `0.0069`
    - Clip 3 (Frames 50..74, 2.083s): Logit `+0.3849`, Prob `0.5950`
    - Clip 4 (Frames 75..99, 3.125s): Logit `-7.2444`, Prob `0.0007`
- **Distributed Sampling (B)**:
  - Mean Logit: `-8.1887` -> Fake Prob: `0.0003` -> Verdict: **REAL** (Correct)
    - Clip 1 (5%, Frames 33..57, 1.375s): Logit `+0.3729`, Prob `0.5922`
    - Clip 2 (30%, Frames 258..282, 10.75s): Logit `-9.1158`, Prob `0.0001`
    - Clip 3 (55%, Frames 484..508, 20.167s): Logit `-8.8510`, Prob `0.0001`
    - Clip 4 (80%, Frames 709..733, 29.542s): Logit `-15.1608`, Prob `0.0000`
- **Score Delta (B - A)**: `-0.0136` (Verdict Changed: **NO**)

### 4. `01__secret_conversation.mp4` (REAL)
- **Video Properties**: 1920x1080, 965 frames, 40.21s duration, 965 valid mouth crops (100.0%)
- **Sequential Sampling (A)**:
  - Mean Logit: `-1.4564` -> Fake Prob: `0.1890` -> Verdict: **REAL** (Correct)
    - Clip 1 (Frames 0..24, 0.0s): Logit `-1.2175`, Prob `0.2284`
    - Clip 2 (Frames 25..49, 1.042s): Logit `-4.1235`, Prob `0.0159`
    - Clip 3 (Frames 50..74, 2.083s): Logit `-1.3212`, Prob `0.2106`
    - Clip 4 (Frames 75..99, 3.125s): Logit `+0.8368`, Prob `0.6978`
- **Distributed Sampling (B)**:
  - Mean Logit: `+3.7183` -> Fake Prob: `0.9763` -> Verdict: **FAKE** (Incorrect)
    - Clip 1 (5%, Frames 36..60, 1.5s): Logit `-0.3529`, Prob `0.4127`
    - Clip 2 (30%, Frames 277..301, 11.542s): Logit `-0.8420`, Prob `0.3011`
    - Clip 3 (55%, Frames 518..542, 21.583s): Logit `+0.9991`, Prob `0.7309`
    - Clip 4 (80%, Frames 759..783, 31.625s): Logit `+15.0690`, Prob `1.0000`
- **Score Delta (B - A)**: `+0.7873` (Verdict Changed: **YES**)

### 5. `01__talking_against_wall.mp4` (REAL)
- **Video Properties**: 1920x1080, 860 frames, 35.83s duration, 860 valid mouth crops (100.0%)
- **Sequential Sampling (A)**:
  - Mean Logit: `-4.6105` -> Fake Prob: `0.0098` -> Verdict: **REAL** (Correct)
    - Clip 1 (Frames 0..24, 0.0s): Logit `-14.6573`, Prob `0.0000`
    - Clip 2 (Frames 25..49, 1.042s): Logit `-3.3963`, Prob `0.0324`
    - Clip 3 (Frames 50..74, 2.083s): Logit `-7.8314`, Prob `0.0004`
    - Clip 4 (Frames 75..99, 3.125s): Logit `+7.4428`, Prob `0.9994`
- **Distributed Sampling (B)**:
  - Mean Logit: `-8.2111` -> Fake Prob: `0.0003` -> Verdict: **REAL** (Correct)
    - Clip 1 (5%, Frames 31..55, 1.292s): Logit `-4.8996`, Prob `0.0074`
    - Clip 2 (30%, Frames 246..270, 10.25s): Logit `-12.8828`, Prob `0.0000`
    - Clip 3 (55%, Frames 460..484, 19.167s): Logit `-9.5607`, Prob `0.0001`
    - Clip 4 (80%, Frames 675..699, 28.125s): Logit `-5.5013`, Prob `0.0041`
- **Score Delta (B - A)**: `-0.0096` (Verdict Changed: **NO**)

### 6. `01_11__meeting_serious__9OM3VE0Y.mp4` (FAKE)
- **Video Properties**: 1920x1080, 1045 frames, 43.54s duration, 1045 valid mouth crops (100.0%)
- **Sequential Sampling (A)**:
  - Mean Logit: `+0.5302` -> Fake Prob: `0.6295` -> Verdict: **FAKE** (Correct)
    - Clip 1 (Frames 0..24, 0.0s): Logit `-0.2594`, Prob `0.4355`
    - Clip 2 (Frames 25..49, 1.042s): Logit `+3.7093`, Prob `0.9761`
    - Clip 3 (Frames 50..74, 2.083s): Logit `-0.8557`, Prob `0.2982`
    - Clip 4 (Frames 75..99, 3.125s): Logit `-0.4733`, Prob `0.3838`
- **Distributed Sampling (B)**:
  - Mean Logit: `+0.3264` -> Fake Prob: `0.5809` -> Verdict: **FAKE** (Correct)
    - Clip 1 (5%, Frames 40..64, 1.667s): Logit `-0.5550`, Prob `0.3647`
    - Clip 2 (30%, Frames 301..325, 12.542s): Logit `-0.4362`, Prob `0.3926`
    - Clip 3 (55%, Frames 562..586, 23.417s): Logit `+3.1978`, Prob `0.9608`
    - Clip 4 (80%, Frames 823..847, 34.292s): Logit `-0.9011`, Prob `0.2888`
- **Score Delta (B - A)**: `-0.0487` (Verdict Changed: **NO**)

### 7. `01_11__secret_conversation__4OJNJLOO.mp4` (FAKE)
- **Video Properties**: 1920x1080, 901 frames, 37.54s duration, 901 valid mouth crops (100.0%)
- **Sequential Sampling (A)**:
  - Mean Logit: `+0.7729` -> Fake Prob: `0.6841` -> Verdict: **FAKE** (Correct)
    - Clip 1 (Frames 0..24, 0.0s): Logit `+0.0853`, Prob `0.5213`
    - Clip 2 (Frames 25..49, 1.042s): Logit `+0.1589`, Prob `0.5396`
    - Clip 3 (Frames 50..74, 2.083s): Logit `+5.9042`, Prob `0.9973`
    - Clip 4 (Frames 75..99, 3.125s): Logit `-3.0570`, Prob `0.0449`
- **Distributed Sampling (B)**:
  - Mean Logit: `+1.3330` -> Fake Prob: `0.7913` -> Verdict: **FAKE** (Correct)
    - Clip 1 (5%, Frames 33..57, 1.375s): Logit `+1.9083`, Prob `0.8708`
    - Clip 2 (30%, Frames 258..282, 10.75s): Logit `+1.9873`, Prob `0.8795`
    - Clip 3 (55%, Frames 483..507, 20.125s): Logit `+0.5371`, Prob `0.6311`
    - Clip 4 (80%, Frames 708..732, 29.5s): Logit `+0.8992`, Prob `0.7108`
- **Score Delta (B - A)**: `+0.1072` (Verdict Changed: **NO**)

### 8. `01_11__talking_against_wall__9229VVZ3.mp4` (FAKE)
- **Video Properties**: 1920x1080, 843 frames, 35.12s duration, 843 valid mouth crops (100.0%)
- **Sequential Sampling (A)**:
  - Mean Logit: `+2.2991` -> Fake Prob: `0.9088` -> Verdict: **FAKE** (Correct)
    - Clip 1 (Frames 0..24, 0.0s): Logit `-6.4044`, Prob `0.0017`
    - Clip 2 (Frames 25..49, 1.042s): Logit `+5.6302`, Prob `0.9964`
    - Clip 3 (Frames 50..74, 2.083s): Logit `+6.3731`, Prob `0.9983`
    - Clip 4 (Frames 75..99, 3.125s): Logit `+3.5975`, Prob `0.9733`
- **Distributed Sampling (B)**:
  - Mean Logit: `+6.3853` -> Fake Prob: `0.9983` -> Verdict: **FAKE** (Correct)
    - Clip 1 (5%, Frames 30..54, 1.25s): Logit `+5.9882`, Prob `0.9975`
    - Clip 2 (30%, Frames 241..265, 10.042s): Logit `-1.9988`, Prob `0.1193`
    - Clip 3 (55%, Frames 451..475, 18.792s): Logit `-5.8412`, Prob `0.0029`
    - Clip 4 (80%, Frames 662..686, 27.583s): Logit `+27.3930`, Prob `1.0000`
- **Score Delta (B - A)**: `+0.0895` (Verdict Changed: **NO**)

### 9. `01_11__walking_outside_cafe_disgusted__FAFWDR4W.mp4` (FAKE)
- **Video Properties**: 1920x1080, 240 frames, 10.0s duration, 240 valid mouth crops (100.0%)
- **Sequential Sampling (A)**:
  - Mean Logit: `+10.4400` -> Fake Prob: `1.0000` -> Verdict: **FAKE** (Correct)
    - Clip 1 (Frames 0..24, 0.0s): Logit `+8.2350`, Prob `0.9997`
    - Clip 2 (Frames 25..49, 1.042s): Logit `+8.9265`, Prob `0.9999`
    - Clip 3 (Frames 50..74, 2.083s): Logit `+5.0644`, Prob `0.9937`
    - Clip 4 (Frames 75..99, 3.125s): Logit `+19.5342`, Prob `1.0000`
- **Distributed Sampling (B)**:
  - Mean Logit: `+1.9309` -> Fake Prob: `0.8734` -> Verdict: **FAKE** (Correct)
    - Clip 1 (5%, Frames 0..24, 0.0s): Logit `+8.2350`, Prob `0.9997`
    - Clip 2 (30%, Frames 60..84, 2.5s): Logit `+6.7780`, Prob `0.9989`
    - Clip 3 (55%, Frames 119..143, 4.958s): Logit `-3.7073`, Prob `0.0240`
    - Clip 4 (80%, Frames 179..203, 7.458s): Logit `-3.5819`, Prob `0.0271`
- **Score Delta (B - A)**: `-0.1266` (Verdict Changed: **NO**)

### 10. `01_12__outside_talking_pan_laughing__TNI7KUZ6.mp4` (FAKE)
- **Video Properties**: 1920x1080, 292 frames, 12.17s duration, 292 valid mouth crops (100.0%)
- **Sequential Sampling (A)**:
  - Mean Logit: `+7.6338` -> Fake Prob: `0.9995` -> Verdict: **FAKE** (Correct)
    - Clip 1 (Frames 0..24, 0.0s): Logit `+7.3856`, Prob `0.9994`
    - Clip 2 (Frames 25..49, 1.042s): Logit `+14.2808`, Prob `1.0000`
    - Clip 3 (Frames 50..74, 2.083s): Logit `+7.2377`, Prob `0.9993`
    - Clip 4 (Frames 75..99, 3.125s): Logit `+1.6310`, Prob `0.8363`
- **Distributed Sampling (B)**:
  - Mean Logit: `+7.9710` -> Fake Prob: `0.9997` -> Verdict: **FAKE** (Correct)
    - Clip 1 (5%, Frames 3..27, 0.125s): Logit `+6.5158`, Prob `0.9985`
    - Clip 2 (30%, Frames 75..99, 3.125s): Logit `+1.6310`, Prob `0.8363`
    - Clip 3 (55%, Frames 148..172, 6.167s): Logit `+19.7832`, Prob `1.0000`
    - Clip 4 (80%, Frames 221..245, 9.208s): Logit `+3.9541`, Prob `0.9812`
- **Score Delta (B - A)**: `+0.0001` (Verdict Changed: **NO**)

---

## 4. Key Answers & Analytical Findings

### 1. Sequential vs. Distributed REAL False-Positive Rate (FPR)
- **Sequential Sampling FPR**: **1/5 (20.0%)** (1 false positives out of 5 authentic videos).
- **Distributed Sampling FPR**: **2/5 (40.0%)** (2 false positives out of 5 authentic videos).
- **Impact on Real Videos**: The REAL FPR shifted by `+20.0%`.

### 2. Sequential vs. Distributed FAKE True-Positive Rate (TPR / Recall)
- **Sequential Sampling TPR**: **5/5 (100.0%)** (5 true positives out of 5 manipulated videos).
- **Distributed Sampling TPR**: **5/5 (100.0%)** (5 true positives out of 5 manipulated videos).
- **Impact on Fake Detection**: The FAKE TPR shifted by `+0.0%`.

### 3. Per-Video Score Changes
- For each video in the paired set, the score shift is quantified in the table above.
- Several videos exhibit moderate score fluctuations depending on facial pose changes and speaking dynamics across the timeline.

### 4. Whether Distributed Sampling is Worth Adopting
- **Assessment**: Distributed sampling samples a more representative distribution of the video timeline compared to looking strictly at the first 100 frames.
- However, distributed sampling requires scanning the full video to locate distributed frames (or seeking across the container), which increases video decode time proportionally.
- In terms of forensic classification accuracy, distributed sampling provides marginal variance reduction but does not fundamentally eliminate false positives caused by out-of-domain compression.

### 5. Whether Sampling Appears to be the Main Cause of the False-Positive Problem
- **Conclusion**: **NO**. Sampling strategy is **NOT** the primary root cause of LipForensics false positives.
- As demonstrated conclusively in Part A (`videoplayback.mp4` remained >99% fake probability under both sequential and distributed sampling) and confirmed across this paired 10-video set, the primary driver of false positives is the **domain gap in video compression, resolution, and speaking cadence** between FaceForensics++ training data (c23 high-quality studio clips) and real-world in-the-wild web video.
