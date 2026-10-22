import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root is available on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Mock transformers model and processor before importing main so no downloads/GPU are needed
mock_config = MagicMock()
mock_config.id2label = {0: "real", 1: "fake"}

mock_model = MagicMock()
mock_model.config = mock_config
mock_model.eval.return_value = None
mock_model.to.return_value = mock_model

mock_processor = MagicMock()

patcher_proc = patch("transformers.AutoImageProcessor.from_pretrained", return_value=mock_processor)
patcher_model = patch("transformers.AutoModelForImageClassification.from_pretrained", return_value=mock_model)

patcher_proc.start()
patcher_model.start()

import main
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def clean_upload_dir(tmp_path, monkeypatch):
    """Isolate and ensure cleanup of the upload directory for every test."""
    temp_dir = tmp_path / "test_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "UPLOAD_DIR", temp_dir)
    yield temp_dir
    # Teardown: ensure no temporary files remain
    if temp_dir.exists():
        for f in temp_dir.iterdir():
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass


@pytest.fixture
def client():
    """FastAPI TestClient fixture with mocked dependencies."""
    return TestClient(main.app)
