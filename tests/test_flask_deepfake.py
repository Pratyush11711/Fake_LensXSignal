"""
Unit tests for FakeLens & SIGNAL Flask Deepfake Detection API routes:
- GET /health
- POST /detect (validation, unsupported formats, upload limits, cleanup)
"""

import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app, UPLOAD_DIR
import backend.app as backend_app


class TestFlaskDeepfakeAPI(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def tearDown(self):
        # Ensure upload dir is cleaned up
        if UPLOAD_DIR.exists():
            for f in UPLOAD_DIR.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass

    def test_flask_health_endpoint(self):
        res = self.client.get('/health')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get('status'), 'ok')
        self.assertIn('active_backend', data)
        self.assertIn('max_upload_mb', data)
        self.assertIn('face_detector_loaded', data)
        self.assertIn('nlp_loaded', data)

    def test_flask_detect_missing_file(self):
        res = self.client.post('/detect')
        self.assertEqual(res.status_code, 422)
        data = res.get_json()
        self.assertIn('detail', data)

    def test_flask_detect_unsupported_file_type(self):
        data = {
            'video': (io.BytesIO(b'Fake plain text'), 'unsupported.txt')
        }
        res = self.client.post('/detect', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertIn('Unsupported file type', body.get('detail', ''))
        # Confirm cleanup
        files = list(UPLOAD_DIR.iterdir())
        self.assertEqual(len(files), 0)

    def test_flask_detect_size_exceeded(self):
        # Temporarily mock MAX_UPLOAD_MB to tiny threshold
        orig_limit = backend_app.MAX_UPLOAD_MB
        try:
            backend_app.MAX_UPLOAD_MB = 0.0001  # ~100 bytes
            dummy_data = b'0' * 500
            data = {
                'video': (io.BytesIO(dummy_data), 'large_video.mp4')
            }
            res = self.client.post('/detect', data=data, content_type='multipart/form-data')
            self.assertEqual(res.status_code, 413)
            body = res.get_json()
            self.assertIn('exceeds', body.get('detail', ''))
            # Confirm cleanup
            files = list(UPLOAD_DIR.iterdir())
            self.assertEqual(len(files), 0)
        finally:
            backend_app.MAX_UPLOAD_MB = orig_limit


if __name__ == '__main__':
    unittest.main()
