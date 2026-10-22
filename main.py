"""
FakeLens & SIGNAL - Unified Multimodal Detection Platform (FastAPI Entrypoint)
Provides high-performance asynchronous API endpoints for:
- Video Deepfake Detection (/detect, /v1/analyze)
- NLP Text Misinformation Classification (/predict, /api/predict)
- System Health & Model Telemetry (/health, /api/status)
"""

import asyncio
import logging
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Setup Paths & Environment
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = Path(os.getenv("STATIC_DIR", str(BASE_DIR / "static")))
TEMPLATES_DIR = BASE_DIR / "templates"

# Logging Setup
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fakelens-signal")


def _get_int_env(var_name: str, default: int, min_val: int, max_val: int) -> int:
    raw_val = os.getenv(var_name)
    if raw_val is None:
        return default
    try:
        val = int(raw_val)
        return val if min_val <= val <= max_val else default
    except ValueError:
        return default


_lipforensics_default_path = BASE_DIR / "models" / "weights" / "lipforensics_ff.pth"
_default_backend = "lipforensics" if _lipforensics_default_path.exists() else "baseline"
MODEL_BACKEND = os.getenv("MODEL_BACKEND", _default_backend).lower()
MODEL_NAME = os.getenv("MODEL_NAME", "dima806/deepfake_vs_real_image_detection")
FACE_DETECTOR_MODEL = Path(os.getenv("FACE_DETECTOR_MODEL", str(BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx")))
LIPFORENSICS_CHECKPOINT = os.getenv("LIPFORENSICS_CHECKPOINT", str(BASE_DIR / "models" / "weights" / "lipforensics_ff.pth"))
TEMPORAL_MODEL_WEIGHTS = os.getenv("TEMPORAL_MODEL_WEIGHTS", str(BASE_DIR / "models" / "best_temporal_model.pth"))
FRAMES_TO_CHECK = _get_int_env("FRAMES_TO_CHECK", default=16, min_val=1, max_val=60)
MAX_UPLOAD_MB = _get_int_env("MAX_UPLOAD_MB", default=100, min_val=1, max_val=500)
MAX_VIDEO_SECONDS = _get_int_env("MAX_VIDEO_SECONDS", default=300, min_val=1, max_val=3600)
INFERENCE_CONCURRENCY = _get_int_env("INFERENCE_CONCURRENCY", default=2, min_val=1, max_val=100)
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

# Upload directory
import tempfile
_is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
_default_upload_dir = Path(tempfile.gettempdir()) / "Uploaded_Files" if _is_serverless else BASE_DIR / "Uploaded_Files"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(_default_upload_dir)))

try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for item in UPLOAD_DIR.iterdir():
        if item.is_file():
            try:
                item.unlink()
            except Exception:
                pass
except Exception as e:
    logger.warning("Upload dir startup notice: %s", e)

inference_semaphore = asyncio.Semaphore(INFERENCE_CONCURRENCY)
app = FastAPI(title="FakeLens & SIGNAL - Multimodal Detection Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- INITIALIZE ENGINES ---
lipforensics_engine = None
baseline_engine = None
temporal_engine = None
face_detector = None
face_mesh_detector = None

try:
    if MODEL_BACKEND == "lipforensics" and Path(LIPFORENSICS_CHECKPOINT).exists():
        from src.lipforensics_video_inference import LipForensicsVideoInference
        lipforensics_engine = LipForensicsVideoInference(
            checkpoint_path=LIPFORENSICS_CHECKPOINT,
            yunet_model_path=str(FACE_DETECTOR_MODEL),
            device=device,
            frames_per_clip=25,
            max_clips_per_video=4,
        )
    elif MODEL_BACKEND == "temporal_xception" and Path(TEMPORAL_MODEL_WEIGHTS).exists():
        from src.pipeline.inference_engine import TemporalInferenceEngine
        temporal_engine = TemporalInferenceEngine(
            weights_path=TEMPORAL_MODEL_WEIGHTS,
            face_detector_path=str(FACE_DETECTOR_MODEL),
            sequence_length=FRAMES_TO_CHECK,
            device=device,
        )
    else:
        from src.baseline.baseline_vit import BaselineViTDetector
        baseline_engine = BaselineViTDetector(
            model_name=MODEL_NAME,
            face_detector_model=str(FACE_DETECTOR_MODEL),
            device=device,
        )
except Exception as exc:
    logger.warning("FastAPI Deepfake engine deferred initialization: %s", exc)

if lipforensics_engine:
    face_detector = lipforensics_engine.yunet_detector
elif baseline_engine:
    face_detector = baseline_engine.face_detector
    face_mesh_detector = baseline_engine.face_mesh
elif temporal_engine:
    face_detector = temporal_engine.detector.detector
    face_mesh_detector = temporal_engine.aligner.detector

# --- NLP Assets ---
nlp_model = None
nlp_vectorizer = None


def get_nlp_assets():
    global nlp_model, nlp_vectorizer
    if nlp_model is None or nlp_vectorizer is None:
        import joblib
        m_path = BASE_DIR / "model" / "logistic_regression.pkl"
        v_path = BASE_DIR / "model" / "tfidf_vectorizer.pkl"
        if m_path.exists() and v_path.exists():
            nlp_model = joblib.load(str(m_path))
            nlp_vectorizer = joblib.load(str(v_path))
    return nlp_model, nlp_vectorizer


# --- PROVISIONAL 3-TIER FORENSIC ASSESSMENT POLICY ---
def get_forensic_assessment_tier(score: float) -> Tuple[str, str]:
    if score < 0.40:
        return (
            "LIKELY AUTHENTIC",
            "Low forensic anomaly signal detected. Spatio-temporal consistency metrics align with natural recording parameters.",
        )
    elif score < 0.90:
        return (
            "INCONCLUSIVE — NEEDS REVIEW",
            "Ambiguous forensic signal. Independent review is recommended.",
        )
    else:
        return (
            "SIGNS OF POSSIBLE MANIPULATION",
            "High forensic anomaly signal detected. This does not by itself establish manipulation. Compression, motion, lighting, and camera conditions can elevate the score. Independent verification is recommended.",
        )


# --- FRONTEND ---
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        content = index_path.read_text(encoding="utf-8")
        # Replace Flask jinja asset helpers if raw HTML returned
        content = content.replace("{{ url_for('static', filename='style.css') }}", "/static/style.css")
        content = content.replace("{{ url_for('static', filename='script.js') }}", "/static/script.js")
        return HTMLResponse(content=content)
    return HTMLResponse("<h2>FakeLens & SIGNAL Frontend not found</h2>", status_code=404)


# --- HEALTH CHECK ---
@app.get("/health")
async def health_check():
    detector_display = "YUNET ONNX" if "yunet" in str(FACE_DETECTOR_MODEL).lower() else Path(FACE_DETECTOR_MODEL).stem.upper()
    alignment_display = "MOUTH ROI CROP" if MODEL_BACKEND == "lipforensics" else "5-PT AFFINE"
    stride_display = f"{25 if MODEL_BACKEND == 'lipforensics' else FRAMES_TO_CHECK} FRAMES"
    input_limit_display = f"MAX {MAX_UPLOAD_MB} MB"
    is_deepfake_loaded = (lipforensics_engine is not None) or (baseline_engine is not None) or (temporal_engine is not None)

    return {
        "status": "ok",
        "active_backend": MODEL_BACKEND,
        "lipforensics_loaded": lipforensics_engine is not None,
        "baseline_loaded": baseline_engine is not None,
        "temporal_loaded": temporal_engine is not None,
        "model_loaded": is_deepfake_loaded or FACE_DETECTOR_MODEL.exists(),
        "face_detector_loaded": FACE_DETECTOR_MODEL.exists(),
        "face_detector_model": str(FACE_DETECTOR_MODEL),
        "model_name": "LipForensics (CVPR 2021)" if MODEL_BACKEND == "lipforensics" else MODEL_NAME,
        "device": str(device),
        "output_policy": "3-tier-provisional-v1",
        "max_upload_mb": MAX_UPLOAD_MB,
        "detector_name": detector_display,
        "alignment_method": alignment_display,
        "sampling_stride": stride_display,
        "input_limit": input_limit_display,
    }


# --- DEEPFAKE VIDEO DETECTION ---
@app.post("/detect")
@app.post("/v1/analyze")
@app.post("/api/deepfake/detect")
async def detect_fake_video(video: UploadFile = File(...)):
    original_name = video.filename or ""
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension or 'unknown'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    safe_name = f"{uuid.uuid4().hex}{extension}"
    file_path = UPLOAD_DIR / safe_name
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    bytes_written = 0

    try:
        with open(file_path, "wb") as buffer:
            while chunk := await video.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_MB}MB upload limit.",
                    )
                buffer.write(chunk)

        async with inference_semaphore:
            if MODEL_BACKEND == "lipforensics" and lipforensics_engine is not None:
                res = await run_in_threadpool(lipforensics_engine.analyze_video, str(file_path))
                status = res.get("status")
                if status == "ERROR":
                    err = res.get("error", "")
                    if any(w in err.lower() for w in ["not found", "failed to open", "unreadable"]):
                        raise HTTPException(status_code=400, detail="Could not open video file.")
                    raise HTTPException(status_code=500, detail="Video analysis failed.")
                if status == "NO_FACE_DETECTED":
                    raise HTTPException(status_code=400, detail="No face detected in sampled frames.")
                if status == "INSUFFICIENT_MOUTH_EVIDENCE":
                    raise HTTPException(status_code=400, detail="Insufficient mouth evidence detected across consecutive frames.")

                fake_score = float(res.get("fake_score", 0.0))
                result = "FAKE" if fake_score >= 0.50 else "REAL"
                confidence = max(fake_score, 1.0 - fake_score)
                clip_scores = res.get("clip_scores", [])
                tier, message = get_forensic_assessment_tier(fake_score)

                response_payload = {
                    "status": "ANALYSIS_COMPLETE",
                    "model_backend": "lipforensics",
                    "result": result,
                    "detection_score": round(fake_score, 4),
                    "fake_probability": round(fake_score, 4),
                    "forensic_score": round(fake_score, 4),
                    "assessment_tier": tier,
                    "assessment_message": message,
                    "confidence": f"{confidence:.2%}",
                    "decision_threshold": 0.50,
                    "provisional_thresholds": {
                        "likely_authentic_max": 0.40,
                        "inconclusive_max": 0.90,
                        "possible_manipulation_min": 0.90,
                    },
                    "frames_analyzed": res.get("valid_mouth_crops", len(clip_scores) * 25),
                    "frame_scores": clip_scores if clip_scores else [round(fake_score, 4)],
                    "face_coverage": res.get("face_coverage", 1.0),
                    "execution_time": res.get("inference_time_sec", 0.0),
                }
                if res.get("face_scan") is not None:
                    response_payload["face_scan"] = res["face_scan"]
                return response_payload

            elif MODEL_BACKEND == "temporal_xception" and temporal_engine is not None:
                result_payload = await run_in_threadpool(
                    temporal_engine.process_video,
                    str(file_path),
                    frames_to_sample=FRAMES_TO_CHECK,
                    max_duration_sec=MAX_VIDEO_SECONDS,
                )
                if result_payload.get("status") == "ERROR":
                    err = result_payload.get("error")
                    if err == "VIDEO_OPEN_FAILED":
                        raise HTTPException(status_code=400, detail="Could not open video file.")
                    if err == "VIDEO_TOO_LONG":
                        raise HTTPException(status_code=400, detail=f"Video exceeds {MAX_VIDEO_SECONDS}s limit.")
                    raise HTTPException(status_code=500, detail="Video analysis failed.")
                return result_payload

            elif baseline_engine is not None:
                result, confidence, frame_scores, face_scan = await run_in_threadpool(
                    baseline_engine.predict_video,
                    str(file_path),
                    frames_to_check=FRAMES_TO_CHECK,
                    max_duration_sec=MAX_VIDEO_SECONDS,
                )
                if result == "VIDEO_OPEN_FAILED":
                    raise HTTPException(status_code=400, detail="Could not open video file.")
                if result == "VIDEO_TOO_LONG":
                    raise HTTPException(status_code=400, detail=f"Video exceeds {MAX_VIDEO_SECONDS}s limit.")
                if result == "NO_FACE_DETECTED" or result is None:
                    raise HTTPException(status_code=400, detail="No face detected in sampled frames.")
                if result == "CUDA_OOM":
                    raise HTTPException(status_code=503, detail="GPU temporarily out of memory.")
                if result == "INFERENCE_FAILED":
                    raise HTTPException(status_code=500, detail="Inference failed.")

                avg_fake = float(np.mean(frame_scores)) if frame_scores else 0.0
                tier, message = get_forensic_assessment_tier(avg_fake)
                response_payload = {
                    "status": "ANALYSIS_COMPLETE",
                    "model_backend": "baseline_vit",
                    "result": result,
                    "detection_score": round(avg_fake, 4),
                    "fake_probability": round(avg_fake, 4),
                    "forensic_score": round(avg_fake, 4),
                    "assessment_tier": tier,
                    "assessment_message": message,
                    "confidence": f"{confidence:.2%}" if isinstance(confidence, (int, float)) else str(confidence),
                    "decision_threshold": 0.50,
                    "provisional_thresholds": {
                        "likely_authentic_max": 0.40,
                        "inconclusive_max": 0.90,
                        "possible_manipulation_min": 0.90,
                    },
                    "frames_analyzed": len(frame_scores),
                    "frame_scores": [round(s, 4) for s in frame_scores],
                }
                if face_scan is not None:
                    response_payload["face_scan"] = face_scan
                return response_payload
            else:
                raise HTTPException(status_code=503, detail="Deepfake detector model is currently initializing.")

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled error: %s\n%s", original_name, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass


# --- NLP TEXT CLASSIFICATION ---
@app.post("/predict")
@app.post("/api/predict")
async def predict_text(request: Request):
    from model.preprocess import clean_text, extract_contributing_features
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    title = str(body.get("title", "")).strip()
    content = str(body.get("text", "") or body.get("content", "") or body.get("article", "")).strip()
    raw_text = f"{title} {content}".strip() if title and content else (content or title)

    if not raw_text or len(raw_text) < 10:
        raise HTTPException(status_code=400, detail="Please provide at least 10 characters of news text.")

    # 1. Primary Engine: RoBERTa Deep Contextual Transformer
    try:
        from src.roberta_detector import RobertaFakeNewsDetector
        if not hasattr(app.state, "roberta_detector") or app.state.roberta_detector is None:
            app.state.roberta_detector = RobertaFakeNewsDetector()
        if app.state.roberta_detector.is_ready:
            rob_res = await run_in_threadpool(app.state.roberta_detector.predict, raw_text)
            model_assets = get_nlp_assets()
            features = extract_contributing_features(raw_text, model_assets[1], model_assets[0], top_k=8) if model_assets[0] and model_assets[1] else []
            return {
                "status": "success",
                "prediction": rob_res["prediction"],
                "confidence": rob_res["confidence"],
                "fake_probability": rob_res["fake_probability"],
                "real_probability": rob_res["real_probability"],
                "model": "RoBERTa (hamzab/roberta-fake-news-classification)",
                "architecture": "Deep Bidirectional Transformer (RoBERTa-base)",
                "features": features,
            }
    except Exception as exc:
        logger.warning(f"RoBERTa inference fallback: {exc}")

    # 2. Secondary Engine: Calibrated Lexical Model Fallback
    model, vec = get_nlp_assets()
    if model is None or vec is None:
        raise HTTPException(status_code=503, detail="NLP model not trained or available.")

    cleaned = clean_text(raw_text, strip_source=False)
    X_vec = vec.transform([cleaned])
    features = extract_contributing_features(raw_text, vec, model, top_k=8)

    coefs = model.coef_[0]
    lexical_logit = float(X_vec.dot(coefs)[0])
    p_fake = float(1.0 / (1.0 + np.exp(-lexical_logit)))
    p_real = float(1.0 - p_fake)

    verdict = "FAKE" if lexical_logit > 0.0 else "REAL"
    confidence = round(float(p_fake if verdict == "FAKE" else p_real) * 100, 2)

    return {
        "status": "success",
        "prediction": verdict,
        "confidence": confidence,
        "fake_probability": round(p_fake, 4),
        "real_probability": round(p_real, 4),
        "lexical_score": round(lexical_logit, 4),
        "model": "TF-IDF + Logistic Regression (Calibrated Fallback)",
        "features": features,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
