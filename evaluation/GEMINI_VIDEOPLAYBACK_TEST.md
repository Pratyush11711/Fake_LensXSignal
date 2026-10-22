# Gemini Single-Video Forensic Evaluation: `videoplayback.mp4`

## 1. Overview & Test Metadata
- **Video File**: `videoplayback.mp4`
- **Full Path**: `C:\Users\hp\Downloads\videoplayback.mp4`
- **Ground Truth**: **REAL** (User-Verified)
- **Video Duration**: `167.30s` (5019 frames @ 30.0 FPS, 640x352)
- **File Size**: `7.65 MB`
- **Gemini Model**: `gemini-3.6-flash`
- **HTTP Status**: `200`
- **Total API Time**: `39.25s` (upload, ingestion, generateContent, cleanup)
- **File Deletion Status**: `HTTP 200 (Cleaned up from Gemini Files API)`

---

## 2. Gemini Forensic Analysis Results

| Metric | Result |
| :--- | :--- |
| **Gemini Verdict** | **`REAL`** |
| **Gemini Confidence** | **`0.75`** |
| **Ground Truth Agreement** | **AGREES (Correct - REAL)** |
| **Timestamps Cited** | `NONE` |

### Evidence Provided by Gemini
```text
Consistent facial motion, natural lip-syncing, uniform lighting, and seamless physical integration with the guitar throughout the clip.
```

### Full Raw Model Response
```text
VERDICT: REAL
CONFIDENCE: 0.75
EVIDENCE: Consistent facial motion, natural lip-syncing, uniform lighting, and seamless physical integration with the guitar throughout the clip.
TIMESTAMPS: NONE
```

---

## 3. Comparison with LipForensics Baseline

| Detector | Verdict | Confidence / Score | Runtime | Result vs Ground Truth |
| :--- | :---: | :---: | :---: | :---: |
| **LipForensics (CVPR 2021)** | `FAKE` | `0.9913` (Synthetic Prob) | ~47.8s | **False Positive** (tripped by mouth compression/motion) |
| **Gemini 3.6 Flash** | **`REAL`** | **`0.75`** (Confidence) | `39.25s` | **True Negative (Correct)** |

> [!NOTE]
> LipForensics evaluates mouth ROI temporal dynamics only, producing a false positive on compression and speech cadence artifacts. Gemini evaluates full-frame holistic consistency across facial structure, lighting, and timeline continuity, accurately verifying the authentic status of `videoplayback.mp4`.
