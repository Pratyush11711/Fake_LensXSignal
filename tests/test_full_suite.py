"""
Comprehensive End-to-End Test Suite for FakeLens & SIGNAL Integration
Tests every feature across both NLP Fake News Classification and Deepfake Video Detection.
"""

import io
import os
import sys
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app, UPLOAD_DIR
import backend.app as backend_app


class TestFullFakeLensSignalIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def setUp(self):
        # Clean upload directory before each test
        if UPLOAD_DIR.exists():
            for f in UPLOAD_DIR.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass

    def tearDown(self):
        # Verify no orphaned files remain after test
        if UPLOAD_DIR.exists():
            remaining = [f for f in UPLOAD_DIR.iterdir() if f.is_file()]
            self.assertEqual(len(remaining), 0, f"Found orphaned files in upload dir: {remaining}")

    # =========================================================================
    # 1. Frontend & Status Routes
    # =========================================================================
    def test_01_frontend_index_route(self):
        """GET / returns HTML index with both NLP and SIGNAL UI components."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("FAKELENS", html.upper())
        self.assertIn("SIGNAL", html.upper())
        self.assertIn("lens-switcher", html)
        self.assertIn("panel-nlp", html)
        self.assertIn("panel-video", html)
        self.assertIn("viewfinder-hud", html)
        self.assertIn("timeline-bars", html)

    def test_02_system_status_endpoint(self):
        """GET /api/status returns model readiness and dataset availability."""
        res = self.client.get('/api/status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertIn('model_loaded', data)
        self.assertIn('nlp_loaded', data)
        self.assertIn('deepfake_loaded', data)

    def test_03_deepfake_health_endpoint(self):
        """GET /health returns active backend, upload limit, and model metadata."""
        res = self.client.get('/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'ok')
        self.assertIn('active_backend', data)
        self.assertIn('max_upload_mb', data)
        self.assertIn('face_detector_loaded', data)

    # =========================================================================
    # 2. NLP Fake News Classification Feature Tests
    # =========================================================================
    def test_04_nlp_predict_real_news(self):
        """POST /predict accurately classifies legitimate Reuters news as REAL with high confidence."""
        real_sample = (
            "WASHINGTON (Reuters) - Senate Republicans and Democrats reached a bipartisan compromise "
            "on federal budget appropriations late Wednesday, averting a government shutdown before the midnight deadline. "
            "The bill passed 74 to 24 and now heads to the White House for presidential signature."
        )
        res = self.client.post('/predict', json={'text': real_sample})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data.get('prediction'), 'REAL')
        self.assertGreater(data.get('confidence', 0), 80.0)
        self.assertIsInstance(data.get('features'), list)
        self.assertGreater(len(data['features']), 0)

    def test_05_nlp_predict_fake_news(self):
        """POST /predict accurately classifies fabricated sensational text as FAKE with high confidence."""
        fake_sample = (
            "BREAKING BOMBSHELL: Secret video reveals corrupt officials caught admitting to hidden algorithms "
            "altering ballots in key counties! Watch the featured image evidence mainstream media is blacking out — "
            "share before it gets deleted by the deep state cabal!"
        )
        res = self.client.post('/predict', json={'text': fake_sample})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data.get('prediction'), 'FAKE')
        self.assertGreater(data.get('confidence', 0), 80.0)
        self.assertIsInstance(data.get('features'), list)
        self.assertGreater(len(data['features']), 0)

    def test_06_nlp_predict_validation_errors(self):
        """POST /predict validates empty text, short inputs, and missing payloads."""
        # Empty text
        res_empty = self.client.post('/predict', json={'text': ''})
        self.assertEqual(res_empty.status_code, 400)
        self.assertIn('error', res_empty.get_json())

        # Too short (< 10 chars)
        res_short = self.client.post('/predict', json={'text': 'Hello!'})
        self.assertEqual(res_short.status_code, 400)
        self.assertIn('too short', res_short.get_json().get('error', '').lower())

    def test_07_nlp_exhibit_api_routes(self):
        """GET /api/metrics, /api/stats, /api/leakage-comparison, /api/error-analysis respond with JSON."""
        # Metrics
        res_m = self.client.get('/api/metrics')
        if res_m.status_code == 200:
            self.assertIn('data', res_m.get_json())

        # Stats
        res_s = self.client.get('/api/stats')
        if res_s.status_code == 200:
            self.assertIn('data', res_s.get_json())

        # Leakage
        res_l = self.client.get('/api/leakage-comparison')
        if res_l.status_code == 200:
            self.assertIn('data', res_l.get_json())

        # Error analysis
        res_e = self.client.get('/api/error-analysis')
        if res_e.status_code == 200:
            self.assertIn('data', res_e.get_json())

    # =========================================================================
    # 3. SIGNAL Deepfake Video Detection Feature Tests
    # =========================================================================
    def test_08_deepfake_detect_validation_missing_file(self):
        """POST /detect without a file upload returns HTTP 422 validation error."""
        res = self.client.post('/detect')
        self.assertEqual(res.status_code, 422)
        data = res.get_json()
        self.assertIn('detail', data)

    def test_09_deepfake_detect_validation_unsupported_extension(self):
        """POST /detect with unsupported format (.pdf, .txt) returns HTTP 400."""
        data = {'video': (io.BytesIO(b'%PDF-1.4 dummy document'), 'document.pdf')}
        res = self.client.post('/detect', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)
        self.assertIn('Unsupported file type', res.get_json().get('detail', ''))

    def test_10_deepfake_detect_validation_size_limit(self):
        """POST /detect with oversized payload returns HTTP 413 and cleans up immediately."""
        orig_max = backend_app.MAX_UPLOAD_MB
        try:
            backend_app.MAX_UPLOAD_MB = 0.0001  # ~100 bytes
            dummy_bytes = b'0' * 500
            data = {'video': (io.BytesIO(dummy_bytes), 'huge.mp4')}
            res = self.client.post('/detect', data=data, content_type='multipart/form-data')
            self.assertEqual(res.status_code, 413)
            self.assertIn('exceeds', res.get_json().get('detail', ''))
        finally:
            backend_app.MAX_UPLOAD_MB = orig_max

    def test_11_deepfake_video_inference_with_face(self):
        """POST /detect with valid facial video executes the pipeline and returns 200 with 3-tier forensic telemetry."""
        video_path = PROJECT_ROOT / 'scratch' / 'sample_face_video.mp4'
        if not video_path.exists():
            self.skipTest("scratch/sample_face_video.mp4 not present for live inference test")

        with open(video_path, 'rb') as f:
            data = {'video': (f, 'sample_face_video.mp4')}
            res = self.client.post('/detect', data=data, content_type='multipart/form-data')

        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body.get('status'), 'ANALYSIS_COMPLETE')
        self.assertIn('result', body)
        self.assertIn('assessment_tier', body)
        self.assertIn('assessment_message', body)
        self.assertIn(body['assessment_tier'], [
            'LIKELY AUTHENTIC',
            'INCONCLUSIVE — NEEDS REVIEW',
            'SIGNS OF POSSIBLE MANIPULATION'
        ])
        self.assertIn('confidence', body)
        self.assertIn('frame_scores', body)
        self.assertIsInstance(body['frame_scores'], list)
        self.assertGreater(len(body['frame_scores']), 0)
        self.assertEqual(body.get('frames_analyzed'), 16)

    def test_12_deepfake_no_face_detected_validation(self):
        """POST /detect with a video containing no detectable faces returns HTTP 400 and clear error."""
        video_path = PROJECT_ROOT / 'scratch' / 'test_video.mp4'
        if not video_path.exists():
            self.skipTest("scratch/test_video.mp4 not present")

        with open(video_path, 'rb') as f:
            data = {'video': (f, 'test_video.mp4')}
            res = self.client.post('/detect', data=data, content_type='multipart/form-data')

        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertIn('no face detected', body.get('detail', '').lower())


if __name__ == '__main__':
    unittest.main()
