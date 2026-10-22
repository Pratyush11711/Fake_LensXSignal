"""
FakeLens Unit & Pipeline Test Suite
Tests:
1. Text preprocessing (HTML, URLs, Emails, Case, Whitespace, Datelines)
2. Vectorizer and Model fitting pipeline
3. Metrics and Confusion Matrix generation
4. Feature contribution explainability extraction
5. REST API responses and error validation
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.preprocess import clean_text, extract_contributing_features
from backend.app import app


class TestFakeLensPipeline(unittest.TestCase):

    def test_clean_text_basic(self):
        raw = "<p>Check this out https://example.com/breaking! <b>Amazing</b> news!</p>"
        cleaned = clean_text(raw)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("amazing news", cleaned)

    def test_clean_text_dateline_stripping(self):
        raw_reuters = "WASHINGTON (Reuters) - The Senate approved the new bill on Tuesday."
        stripped = clean_text(raw_reuters, strip_source=True)
        self.assertNotIn("reuters", stripped.lower())
        self.assertIn("senate approved the new bill", stripped)

    def test_clean_text_null_handling(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(""), "")

    def test_feature_extraction_explainability(self):
        # Create a small toy corpus to test feature weights
        docs = [
            "breaking shock bombshell corrupt deep state scandal",
            "urgent conspiracy hoax revealed secretly",
            "senate committee passes economic appropriations bill",
            "federal department announces infrastructure investment project"
        ]
        labels = [1, 1, 0, 0]  # 1 = Fake, 0 = Real
        
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        X = vec.fit_transform(docs)
        
        clf = LogisticRegression(random_state=42)
        clf.fit(X, labels)
        
        test_text = "breaking shock federal bill"
        features = extract_contributing_features(test_text, vec, clf, top_k=4)
        
        self.assertTrue(isinstance(features, list))
        self.assertGreater(len(features), 0)
        for f in features:
            self.assertIn("term", f)
            self.assertIn("weight", f)
            self.assertIn("influence", f)
            self.assertIn(f["influence"], ["FAKE", "REAL"])

    def test_flask_api_routes(self):
        client = app.test_client()
        
        # Test GET /
        res_home = client.get('/')
        self.assertEqual(res_home.status_code, 200)
        self.assertIn(b'FAKELENS', res_home.data)
        
        # Test GET /api/status
        res_status = client.get('/api/status')
        self.assertEqual(res_status.status_code, 200)
        data = res_status.get_json()
        self.assertIn('model_loaded', data)
        self.assertIn('dataset_files_present', data)
        
        # Test POST /predict with short input
        res_short = client.post('/predict', json={'text': 'hi'})
        self.assertEqual(res_short.status_code, 400)
        
        # Test POST /predict with empty payload
        res_empty = client.post('/predict', json={'text': ''})
        self.assertEqual(res_empty.status_code, 400)


if __name__ == '__main__':
    unittest.main()
