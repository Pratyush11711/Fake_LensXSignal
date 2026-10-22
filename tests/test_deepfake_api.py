import io
import main


def test_health_endpoint(client):
    """GET /health returns HTTP 200 and JSON with status: ok and live pipeline configuration."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["face_detector_loaded"] is True
    assert "face_detector_model" in data
    assert "model_name" in data
    assert "device" in data
    assert "detector_name" in data
    assert "alignment_method" in data
    assert "sampling_stride" in data
    assert "input_limit" in data
    assert "max_upload_mb" in data


def test_detect_no_file(client):
    """POST /detect with no uploaded file returns HTTP 422 (validation error)."""
    response = client.post("/detect")
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_detect_unsupported_file_type(client, clean_upload_dir):
    """POST /detect with a .txt file returns HTTP 400 and unsupported file type message."""
    files = {"video": ("sample.txt", io.BytesIO(b"Hello world text file"), "text/plain")}
    response = client.post("/detect", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file type '.txt'" in data["detail"]
    # Verify no temporary files remain in UPLOAD_DIR
    remaining_files = list(clean_upload_dir.iterdir())
    assert len(remaining_files) == 0, f"Expected clean directory, found: {remaining_files}"


def test_detect_file_size_exceeded(client, clean_upload_dir, monkeypatch):
    """POST /detect with a file exceeding MAX_UPLOAD_MB returns HTTP 413."""
    # Set MAX_UPLOAD_MB to a tiny limit (~0.0001 MB = 104 bytes) without creating a huge file
    tiny_max_mb = 100 / (1024 * 1024)  # 100 bytes limit
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", tiny_max_mb)

    # 500 bytes of dummy video data (exceeds 100 bytes limit)
    oversized_data = b"0" * 500
    files = {"video": ("oversized.mp4", io.BytesIO(oversized_data), "video/mp4")}

    response = client.post("/detect", files=files)
    assert response.status_code == 413
    data = response.json()
    assert "File exceeds the" in data["detail"]
    # Verify no temporary files remain in UPLOAD_DIR
    remaining_files = list(clean_upload_dir.iterdir())
    assert len(remaining_files) == 0, f"Expected clean directory, found: {remaining_files}"


def test_cleanup_after_rejected_uploads(client, clean_upload_dir):
    """Verify that after multiple rejected uploads, no orphaned files remain."""
    # 1. Non-video file rejection
    client.post("/detect", files={"video": ("document.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")})
    assert len(list(clean_upload_dir.iterdir())) == 0

    # 2. Corrupt video data causing open failure
    client.post("/detect", files={"video": ("corrupt.mp4", io.BytesIO(b"not a valid video stream"), "video/mp4")})
    assert len(list(clean_upload_dir.iterdir())) == 0
