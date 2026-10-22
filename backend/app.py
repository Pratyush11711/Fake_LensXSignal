"""
FakeLens & SIGNAL - Unified Multimodal Detection Platform Backend
Serves:
1. NLP Fake News Classification (TF-IDF + Logistic Regression, explainability, leakage audit)
2. SIGNAL Deepfake Video Forensics (YuNet face tracking, Vision Transformer / LipForensics / Temporal inference)
"""

import os
import sys
import json
import time
import uuid
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.preprocess import clean_text, extract_contributing_features

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
def _get_int_env(var_name: str, default: int, min_val: int, max_val: int) -> int:
    raw_val = os.getenv(var_name)
    if raw_val is None:
        return default
    try:
        val = int(raw_val)
        return val if min_val <= val <= max_val else default
    except ValueError:
        return default


def _get_float_env(var_name: str, default: float, min_val: float, max_val: float) -> float:
    raw_val = os.getenv(var_name)
    if raw_val is None:
        return default
    try:
        val = float(raw_val)
        return val if min_val <= val <= max_val else default
    except ValueError:
        return default


# Deepfake Detector Config
_lipforensics_default_path = PROJECT_ROOT / "models" / "weights" / "lipforensics_ff.pth"
_default_backend = "lipforensics" if _lipforensics_default_path.exists() else "baseline"
MODEL_BACKEND = os.getenv("MODEL_BACKEND", _default_backend).lower()
MODEL_NAME = os.getenv("MODEL_NAME", "dima806/deepfake_vs_real_image_detection")
FACE_DETECTOR_MODEL = Path(os.getenv("FACE_DETECTOR_MODEL", str(PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx")))
LIPFORENSICS_CHECKPOINT = os.getenv("LIPFORENSICS_CHECKPOINT", str(PROJECT_ROOT / "models" / "weights" / "lipforensics_ff.pth"))
TEMPORAL_MODEL_WEIGHTS = os.getenv("TEMPORAL_MODEL_WEIGHTS", str(PROJECT_ROOT / "models" / "best_temporal_model.pth"))
FRAMES_TO_CHECK = _get_int_env("FRAMES_TO_CHECK", default=16, min_val=1, max_val=60)
MAX_UPLOAD_MB = _get_int_env("MAX_UPLOAD_MB", default=100, min_val=1, max_val=500)
MAX_VIDEO_SECONDS = _get_int_env("MAX_VIDEO_SECONDS", default=300, min_val=1, max_val=3600)
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# Temporary Upload Directory
_is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
_default_upload_dir = Path(tempfile.gettempdir()) / "Uploaded_Files" if _is_serverless else PROJECT_ROOT / "Uploaded_Files"
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
    print(f"[WARNING] Upload dir init notice: {e}")

# Initialize Flask App
app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static")
)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# -----------------------------------------------------------------------------
# 1. NLP Model State & Assets (RoBERTa Transformer + Calibrated Baseline)
# -----------------------------------------------------------------------------
roberta_detector = None
roberta_loaded = False
clf_model = None
tfidf_vectorizer = None


def load_roberta_detector():
    """Initializes deep RoBERTa transformer for sequence classification."""
    global roberta_detector, roberta_loaded
    if roberta_loaded and roberta_detector is not None:
        return True
    try:
        from src.roberta_detector import RobertaFakeNewsDetector
        roberta_detector = RobertaFakeNewsDetector()
        roberta_loaded = roberta_detector.is_ready
        return roberta_loaded
    except Exception as e:
        print(f"[WARNING] RoBERTa detector load deferred: {e}")
        roberta_loaded = False
        return False


def load_nlp_model_assets():
    """Loads trained NLP model and TF-IDF vectorizer if available on disk."""
    global clf_model, tfidf_vectorizer
    model_path = PROJECT_ROOT / "model" / "logistic_regression.pkl"
    vec_path = PROJECT_ROOT / "model" / "tfidf_vectorizer.pkl"

    if model_path.exists() and vec_path.exists():
        try:
            clf_model = joblib.load(str(model_path))
            tfidf_vectorizer = joblib.load(str(vec_path))
            print("[INFO] NLP Model and TF-IDF vectorizer successfully loaded.")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load NLP model assets: {e}")
            clf_model = None
            tfidf_vectorizer = None
            return False
    else:
        print("[WARNING] NLP Model files not found. Running in setup mode.")
        return False


load_nlp_model_assets()
load_roberta_detector()

# -----------------------------------------------------------------------------
# 2. SIGNAL Deepfake Detection Engine State
# -----------------------------------------------------------------------------
lipforensics_engine = None
baseline_engine = None
temporal_engine = None
deepfake_engine_loaded = False
deepfake_engine_error = None
device_name = "cpu"


def init_deepfake_engine():
    """Initializes computer vision and deepfake detection engines."""
    global lipforensics_engine, baseline_engine, temporal_engine, deepfake_engine_loaded, deepfake_engine_error, device_name
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = str(device)
        print(f"[INFO] Initializing SIGNAL deepfake detection engine (Backend: {MODEL_BACKEND}, Device: {device})...")

        if MODEL_BACKEND == "lipforensics" and Path(LIPFORENSICS_CHECKPOINT).exists():
            from src.lipforensics_video_inference import LipForensicsVideoInference
            lipforensics_engine = LipForensicsVideoInference(
                checkpoint_path=LIPFORENSICS_CHECKPOINT,
                yunet_model_path=str(FACE_DETECTOR_MODEL),
                device=device,
                frames_per_clip=25,
                max_clips_per_video=4,
            )
            deepfake_engine_loaded = True
            print("[INFO] LipForensics engine successfully loaded.")
        elif MODEL_BACKEND == "temporal_xception" and Path(TEMPORAL_MODEL_WEIGHTS).exists():
            from src.pipeline.inference_engine import TemporalInferenceEngine
            temporal_engine = TemporalInferenceEngine(
                weights_path=TEMPORAL_MODEL_WEIGHTS,
                face_detector_path=str(FACE_DETECTOR_MODEL),
                sequence_length=FRAMES_TO_CHECK,
                device=device,
            )
            deepfake_engine_loaded = True
            print("[INFO] Temporal Xception engine successfully loaded.")
        else:
            # Baseline ViT Architecture
            from src.baseline.baseline_vit import BaselineViTDetector
            baseline_engine = BaselineViTDetector(
                model_name=MODEL_NAME,
                face_detector_model=str(FACE_DETECTOR_MODEL),
                device=device,
            )
            deepfake_engine_loaded = True
            print("[INFO] Baseline ViT engine successfully loaded.")
        return True
    except Exception as exc:
        deepfake_engine_error = str(exc)
        print(f"[WARNING] Deepfake engine initialization deferred: {exc}")
        return False


# Attempt deepfake engine load only if explicitly enabled or on demand
if os.getenv("EAGER_LOAD_DEEPFAKE", "0") == "1":
    init_deepfake_engine()


def get_forensic_assessment_tier(score: float) -> Tuple[str, str]:
    """
    Provisional 3-tier forensic assessment policy bands:
    - score < 0.40          -> LIKELY AUTHENTIC
    - 0.40 <= score < 0.90  -> INCONCLUSIVE — NEEDS REVIEW
    - score >= 0.90         -> SIGNS OF POSSIBLE MANIPULATION
    """
    if score < 0.40:
        return (
            "LIKELY AUTHENTIC",
            "Low forensic anomaly signal detected. Spatio-temporal consistency metrics align with natural recording parameters.",
        )
    elif score < 0.90:
        return (
            "INCONCLUSIVE — NEEDS REVIEW",
            "Ambiguous forensic signal. Independent verification is recommended.",
        )
    else:
        return (
            "SIGNS OF POSSIBLE MANIPULATION",
            "High forensic anomaly signal detected. This does not by itself establish manipulation. Compression, motion, lighting, and camera conditions can elevate the score. Independent verification is recommended.",
        )


@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# -----------------------------------------------------------------------------
# Web & Frontend Routes
# -----------------------------------------------------------------------------
@app.route('/')
def home():
    """Renders the unified multimodal detection interface."""
    return render_template('index.html')


# -----------------------------------------------------------------------------
# NLP Misinformation Detector Endpoints
# -----------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    """Classifies news article or headline text as FAKE or REAL."""
    global clf_model, tfidf_vectorizer

    if request.is_json:
        data = request.get_json(silent=True) or {}
        title = str(data.get('title', '')).strip()
        content = str(
            data.get('text', '') or 
            data.get('content', '') or 
            data.get('article', '') or 
            data.get('body', '')
        ).strip()
        raw_text = f"{title} {content}".strip() if title and content else (content or title)
    else:
        title = str(request.form.get('title', '')).strip()
        content = str(
            request.form.get('text', '') or 
            request.form.get('content', '') or 
            request.form.get('article', '') or 
            request.form.get('body', '')
        ).strip()
        raw_text = f"{title} {content}".strip() if title and content else (content or title)

    if not raw_text:
        return jsonify({
            "status": "error",
            "error": "Please provide a news headline, article body, or both to analyze."
        }), 400

    if len(raw_text) < 10:
        return jsonify({
            "status": "error",
            "error": "Input text is too short. Please provide at least 10 characters or a complete headline."
        }), 400

    # 1. Primary Engine: RoBERTa Deep Contextual Transformer
    global roberta_detector, roberta_loaded
    if not roberta_loaded:
        load_roberta_detector()

    if roberta_loaded and roberta_detector is not None:
        try:
            rob_res = roberta_detector.predict(raw_text)
            features = extract_contributing_features(raw_text, tfidf_vectorizer, clf_model, top_k=8) if tfidf_vectorizer and clf_model else []
            return jsonify({
                "status": "success",
                "prediction": rob_res["prediction"],
                "confidence": rob_res["confidence"],
                "fake_probability": rob_res["fake_probability"],
                "real_probability": rob_res["real_probability"],
                "model": "RoBERTa (hamzab/roberta-fake-news-classification)",
                "architecture": "Deep Bidirectional Transformer (RoBERTa-base)",
                "cleaned_text": raw_text[:300] + ("..." if len(raw_text) > 300 else ""),
                "features": features
            })
        except Exception as exc:
            print(f"[WARNING] RoBERTa inference fallback: {exc}")

    # 2. Secondary Engine: Calibrated Lexical Model (Domain-Robust Fallback)
    if clf_model is None or tfidf_vectorizer is None:
        if not load_nlp_model_assets():
            return jsonify({
                "status": "error",
                "error": "NLP model assets not available."
            }), 503

    cleaned = clean_text(raw_text, strip_source=False)
    if not cleaned or len(cleaned.split()) < 2:
        return jsonify({
            "status": "error",
            "error": "Input contains insufficient vocabulary after text normalization."
        }), 400

    try:
        X_vec = tfidf_vectorizer.transform([cleaned])
        features = extract_contributing_features(raw_text, tfidf_vectorizer, clf_model, top_k=8)

        coefs = clf_model.coef_[0]
        lexical_logit = float(X_vec.dot(coefs)[0])

        p_fake = float(1.0 / (1.0 + np.exp(-lexical_logit)))
        p_real = float(1.0 - p_fake)

        verdict = "FAKE" if lexical_logit > 0.0 else "REAL"
        confidence = round(float(p_fake if verdict == "FAKE" else p_real) * 100, 2)

        return jsonify({
            "status": "success",
            "prediction": verdict,
            "confidence": confidence,
            "fake_probability": round(p_fake, 4),
            "real_probability": round(p_real, 4),
            "lexical_score": round(lexical_logit, 4),
            "model": "TF-IDF + Logistic Regression (Calibrated Fallback)",
            "cleaned_text": cleaned[:300] + ("..." if len(cleaned) > 300 else ""),
            "features": features
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": f"An error occurred during prediction: {str(e)}"
        }), 500


@app.route('/api/status', methods=['GET'])
def api_status():
    """Returns status of both NLP and Deepfake Video detection systems."""
    global clf_model, tfidf_vectorizer, deepfake_engine_loaded
    fake_csv = (PROJECT_ROOT / 'dataset' / 'Fake.csv').exists()
    true_csv = (PROJECT_ROOT / 'dataset' / 'True.csv').exists()
    metrics_file = (PROJECT_ROOT / 'model' / 'metrics.json').exists()
    eda_file = (PROJECT_ROOT / 'eda' / 'eda_stats.json').exists()

    nlp_ready = (clf_model is not None and tfidf_vectorizer is not None) or roberta_loaded
    if not nlp_ready:
        nlp_ready = load_nlp_model_assets() or load_roberta_detector()

    return jsonify({
        "status": "success",
        "model_loaded": nlp_ready,
        "model_ready": nlp_ready,
        "nlp_loaded": nlp_ready,
        "roberta_loaded": roberta_loaded,
        "nlp_engine": "RoBERTa Transformer" if roberta_loaded else "TF-IDF + Logistic Regression",
        "deepfake_loaded": deepfake_engine_loaded or (baseline_engine is not None or lipforensics_engine is not None),
        "active_backend": MODEL_BACKEND,
        "device": device_name,
        "dataset_files_present": fake_csv and true_csv,
        "fake_csv_present": fake_csv,
        "true_csv_present": true_csv,
        "metrics_available": metrics_file,
        "eda_available": eda_file
    })


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    path = PROJECT_ROOT / 'model' / 'metrics.json'
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify({"status": "success", "data": data})
    return jsonify({"status": "unavailable", "message": "Metrics not generated yet."}), 404


@app.route('/api/stats', methods=['GET'])
def api_stats():
    model_stats_path = PROJECT_ROOT / 'model' / 'dataset_stats.json'
    eda_stats_path = PROJECT_ROOT / 'eda' / 'eda_stats.json'
    data = {}
    if model_stats_path.exists():
        with open(model_stats_path, 'r', encoding='utf-8') as f:
            data['dataset_stats'] = json.load(f)
    if eda_stats_path.exists():
        with open(eda_stats_path, 'r', encoding='utf-8') as f:
            data['eda_stats'] = json.load(f)
    if data:
        return jsonify({"status": "success", "data": data})
    return jsonify({"status": "unavailable", "message": "Dataset stats not generated yet."}), 404


@app.route('/api/leakage-comparison', methods=['GET'])
def api_leakage():
    path = PROJECT_ROOT / 'model' / 'leakage_comparison.json'
    strict_path = PROJECT_ROOT / 'model' / 'strict_dedup_metrics.json'
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if strict_path.exists():
            with open(strict_path, 'r', encoding='utf-8') as sf:
                strict_data = json.load(sf)
                data['strict_deduplication_model'] = {
                    "dataset_version": "Strict Dedup (0 Title/Prefix Contamination)",
                    "accuracy": strict_data["metrics"]["accuracy"],
                    "precision": strict_data["metrics"]["precision"],
                    "recall": strict_data["metrics"]["recall"],
                    "f1_score": strict_data["metrics"]["f1_score"],
                    "confusion_matrix": strict_data["metrics"]["confusion_matrix"]
                }
        return jsonify({"status": "success", "data": data})
    return jsonify({"status": "unavailable", "message": "Leakage study not executed yet."}), 404


@app.route('/api/error-analysis', methods=['GET'])
def api_error_analysis():
    err_path = PROJECT_ROOT / 'model' / 'error_analysis.json'
    feat_path = PROJECT_ROOT / 'model' / 'top_features.json'
    res = {}
    if err_path.exists():
        with open(err_path, 'r', encoding='utf-8') as f:
            res['errors'] = json.load(f)
    if feat_path.exists():
        with open(feat_path, 'r', encoding='utf-8') as f:
            res['top_features'] = json.load(f)
    if res:
        return jsonify({"status": "success", "data": res})
    return jsonify({"status": "unavailable", "message": "Error analysis not generated yet."}), 404


# -----------------------------------------------------------------------------
# SIGNAL Deepfake Video Detector Endpoints
# -----------------------------------------------------------------------------
@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
@app.route('/api/deepfake/health', methods=['GET'])
def health_check():
    """SIGNAL Health Check endpoint returning live detector configuration."""
    detector_display = "YUNET ONNX" if "yunet" in str(FACE_DETECTOR_MODEL).lower() else FACE_DETECTOR_MODEL.stem.upper()
    alignment_display = "MOUTH ROI CROP" if MODEL_BACKEND == "lipforensics" else "5-PT AFFINE"
    stride_display = f"{25 if MODEL_BACKEND == 'lipforensics' else FRAMES_TO_CHECK} FRAMES"
    input_limit_display = f"MAX {MAX_UPLOAD_MB} MB"

    is_deepfake_loaded = (baseline_engine is not None) or (lipforensics_engine is not None) or (temporal_engine is not None)
    is_face_detector_loaded = FACE_DETECTOR_MODEL.exists()

    return jsonify({
        "status": "ok",
        "active_backend": MODEL_BACKEND,
        "lipforensics_loaded": lipforensics_engine is not None,
        "baseline_loaded": baseline_engine is not None,
        "temporal_loaded": temporal_engine is not None,
        "model_loaded": is_deepfake_loaded or is_face_detector_loaded,
        "face_detector_loaded": is_face_detector_loaded,
        "face_detector_model": str(FACE_DETECTOR_MODEL),
        "model_name": "LipForensics (CVPR 2021)" if MODEL_BACKEND == "lipforensics" else MODEL_NAME,
        "device": device_name,
        "output_policy": "3-tier-provisional-v1",
        "max_upload_mb": MAX_UPLOAD_MB,
        "detector_name": detector_display,
        "alignment_method": alignment_display,
        "sampling_stride": stride_display,
        "input_limit": input_limit_display,
        "nlp_loaded": (clf_model is not None and tfidf_vectorizer is not None)
    })


@app.route('/detect', methods=['POST'])
@app.route('/v1/analyze', methods=['POST'])
@app.route('/api/deepfake/detect', methods=['POST'])
def detect_fake_video():
    """
    Analyzes uploaded video evidence frame-by-frame for facial manipulation.
    Expects multipart form-data with 'video' or 'file' field.
    """
    global baseline_engine, lipforensics_engine, temporal_engine

    video_file = request.files.get("video") or request.files.get("file")
    if not video_file or not video_file.filename:
        return jsonify({
            "detail": "No video file provided. Please upload a valid video file in the 'video' parameter.",
            "status": "error"
        }), 422

    original_name = video_file.filename
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({
            "detail": f"Unsupported file type '{extension or 'unknown'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            "status": "error"
        }), 400

    safe_name = f"{uuid.uuid4().hex}{extension}"
    file_path = UPLOAD_DIR / safe_name
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    bytes_written = 0

    try:
        # Stream save and enforce MAX_UPLOAD_MB strictly
        with open(file_path, "wb") as buffer:
            while True:
                chunk = video_file.stream.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    return jsonify({
                        "detail": f"File exceeds the {MAX_UPLOAD_MB}MB upload limit.",
                        "status": "error"
                    }), 413
                buffer.write(chunk)

        # Lazy initialize if needed
        if not ((baseline_engine is not None) or (lipforensics_engine is not None) or (temporal_engine is not None)):
            init_deepfake_engine()

        # Execute selected deepfake engine
        if MODEL_BACKEND == "lipforensics" and lipforensics_engine is not None:
            res = lipforensics_engine.analyze_video(str(file_path))
            status = res.get("status")
            if status == "ERROR":
                err = res.get("error", "")
                if any(w in err.lower() for w in ["not found", "failed to open", "unreadable"]):
                    return jsonify({"detail": "Could not open video file.", "status": "error"}), 400
                return jsonify({"detail": "Video analysis failed.", "status": "error"}), 500
            if status == "NO_FACE_DETECTED":
                return jsonify({"detail": "No face detected in sampled frames.", "status": "error"}), 400
            if status == "INSUFFICIENT_MOUTH_EVIDENCE":
                return jsonify({"detail": "Insufficient mouth evidence detected across consecutive frames.", "status": "error"}), 400

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
            return jsonify(response_payload)

        elif MODEL_BACKEND == "temporal_xception" and temporal_engine is not None:
            res_payload = temporal_engine.process_video(
                str(file_path),
                frames_to_sample=FRAMES_TO_CHECK,
                max_duration_sec=MAX_VIDEO_SECONDS,
            )
            if res_payload.get("status") == "ERROR":
                err = res_payload.get("error")
                if err == "VIDEO_OPEN_FAILED":
                    return jsonify({"detail": "Could not open video file.", "status": "error"}), 400
                if err == "VIDEO_TOO_LONG":
                    return jsonify({"detail": f"Video exceeds {MAX_VIDEO_SECONDS}s limit.", "status": "error"}), 400
                return jsonify({"detail": "Video analysis failed.", "status": "error"}), 500
            return jsonify(res_payload)

        elif baseline_engine is not None:
            result, confidence, frame_scores, face_scan = baseline_engine.predict_video(
                str(file_path),
                frames_to_check=FRAMES_TO_CHECK,
                max_duration_sec=MAX_VIDEO_SECONDS,
            )
            if result == "VIDEO_OPEN_FAILED":
                return jsonify({"detail": "Could not open video file.", "status": "error"}), 400
            if result == "VIDEO_TOO_LONG":
                return jsonify({"detail": f"Video exceeds {MAX_VIDEO_SECONDS}s limit.", "status": "error"}), 400
            if result == "NO_FACE_DETECTED" or result is None:
                return jsonify({"detail": "No face detected in sampled frames.", "status": "error"}), 400
            if result == "CUDA_OOM":
                return jsonify({"detail": "GPU temporarily out of memory.", "status": "error"}), 503
            if result == "INFERENCE_FAILED":
                return jsonify({"detail": "Inference failed.", "status": "error"}), 500

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
                "frame_scores": [round(float(s), 4) for s in frame_scores],
            }
            if face_scan is not None:
                response_payload["face_scan"] = face_scan
            return jsonify(response_payload)

        else:
            # Fallback if model weights are not loaded yet in offline mode
            return jsonify({
                "detail": f"Deepfake detector engine is not loaded. {deepfake_engine_error or 'Please install PyTorch/transformers and download model assets.'}",
                "status": "error"
            }), 503

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"detail": f"Internal server error: {str(exc)}", "status": "error"}), 500
    finally:
        # Crucial: Always clean up temporary video files
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" FakeLens & SIGNAL Server running on http://0.0.0.0:{port}")
    print(f" Multimodal Misinformation & Deepfake Video Detection")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False)
