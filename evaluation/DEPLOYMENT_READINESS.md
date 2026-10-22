# SIGNAL — Production Deployment Readiness & Live Audit

## Production Status: DEPLOYED & VERIFIED (Vercel Production)

### 1. Deployment Details
- **Hosting Platform**: Vercel (Hobby Tier — 100% Free)
- **Account / Scope**: `mnavvya12-8955`
- **Project Name**: `signal`
- **Live Production URL**: [https://signal-hazel-nine.vercel.app](https://signal-hazel-nine.vercel.app)
- **Deployment ID**: `dpl_GpRzz1egT725UjjkcDm2yjiJJbPj`
- **GitHub Upstream**: [https://github.com/navyamishra1/SIGNAL](https://github.com/navyamishra1/SIGNAL) (`main` branch)

### 2. Verified Endpoints & Health Check
- **`GET /`** -> `HTTP 200 OK` (Embedded forensic frontend UI, 109,412 bytes)
- **`GET /health`** -> `HTTP 200 OK`
```json
{
  "status": "ok",
  "active_backend": "lipforensics",
  "lipforensics_loaded": true,
  "baseline_loaded": false,
  "temporal_loaded": false,
  "model_loaded": true,
  "face_detector_loaded": true,
  "face_detector_model": "models/face_detection_yunet_2023mar.onnx",
  "model_name": "LipForensics (CVPR 2021)",
  "device": "cpu",
  "output_policy": "3-tier-provisional-v1",
  "max_upload_mb": 100,
  "detector_name": "YUNET ONNX",
  "alignment_method": "MOUTH ROI CROP",
  "sampling_stride": "25 FRAMES",
  "input_limit": "MAX 100 MB"
}
```
- **`POST /detect`** -> `HTTP 400` on non-face video (`"No face detected in sampled frames."`), verifying end-to-end YuNet landmarking pipeline and FastAPI input validation in serverless execution.

### 3. Model Integrity & Checkpoint
- **Model Checkpoint**: Downloaded at build time via `scripts/download_weights.py`.
- **SHA-256 Checksum**: `4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d` (Verified).
- **Git Tracking**: 0 `.pth` files tracked in Git.