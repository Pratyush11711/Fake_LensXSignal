"""
scripts/download_weights.py

Ensures all required model weights and checkpoints are present locally before
server startup. Executed during the Render build step.
"""

import hashlib
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

LIPFORENSICS_CHECKPOINT_HASH = "4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d"
LIPFORENSICS_GDRIVE_ID = "1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7"


def verify_sha256(file_path: Path, expected_hash: str) -> bool:
    """Verifies that the file exists and matches the expected SHA-256 hash."""
    if not file_path.exists():
        return False
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest().lower() == expected_hash.lower()


def compute_sha256(file_path: Path) -> str:
    """Computes the SHA-256 hash of a file."""
    if not file_path.exists():
        return ""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def download_lipforensics_checkpoint(target_path: Path) -> None:
    """
    Downloads the verified official LipForensics checkpoint (CVPR 2021)
    and validates its SHA-256 checksum against the official release.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if verify_sha256(target_path, LIPFORENSICS_CHECKPOINT_HASH):
        print(f"[OK] LipForensics checkpoint verified at {target_path} (SHA-256: {LIPFORENSICS_CHECKPOINT_HASH})")
        return

    print(f"Downloading official LipForensics weights to {target_path}...")
    cookie_file = target_path.parent / "gdrive_cookie.txt"
    temp_html = target_path.parent / "gdrive_temp.html"

    download_success = False

    # Attempt 1: curl with Google Drive confirmation token handling
    try:
        cmd1 = [
            "curl", "-c", str(cookie_file), "-s", "-L",
            f"https://drive.google.com/uc?export=download&id={LIPFORENSICS_GDRIVE_ID}",
            "-o", str(temp_html),
        ]
        res1 = subprocess.run(cmd1, capture_output=True)
        if res1.returncode == 0 and temp_html.exists():
            html_text = temp_html.read_text(encoding="utf-8", errors="ignore")
            form_match = re.search(r'<form[^>]*action=[\'"]([^\'"]+)[\'"]', html_text)
            if form_match:
                action = form_match.group(1)
                inputs = re.findall(r'<input[^>]*name=[\'"]([^\'"]+)[\'"][^>]*value=[\'"]([^\'"]*)[\'"]', html_text)
                params = "&".join([f"{k}={v}" for k, v in inputs])
                download_url = f"{action}?{params}" if params else action
                subprocess.run(["curl", "-b", str(cookie_file), "-L", download_url, "-o", str(target_path)], check=True)
                download_success = True
            else:
                match = re.search(r"confirm=([0-9A-Za-z_-]+)", html_text)
                code = match.group(1) if match else "t"
                download_url = f"https://drive.google.com/uc?export=download&confirm={code}&id={LIPFORENSICS_GDRIVE_ID}"
                subprocess.run(["curl", "-b", str(cookie_file), "-L", download_url, "-o", str(target_path)], check=True)
                download_success = True
    except Exception as e:
        print(f"Curl download failed ({e}), trying direct urllib...")

    # Attempt 2: Direct urllib fallback if needed
    if not download_success or not target_path.exists() or target_path.stat().st_size < 10_000_000:
        try:
            direct_url = f"https://drive.google.com/uc?export=download&confirm=t&id={LIPFORENSICS_GDRIVE_ID}"
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-Agent", "Mozilla/5.0")]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(direct_url, str(target_path))
        except Exception as exc:
            print(f"urllib fallback failed: {exc}")

    # Cleanup temporary session files
    if cookie_file.exists():
        try:
            cookie_file.unlink()
        except Exception:
            pass
    if temp_html.exists():
        try:
            temp_html.unlink()
        except Exception:
            pass

    # Strict Checksum Validation
    if not verify_sha256(target_path, LIPFORENSICS_CHECKPOINT_HASH):
        actual_hash = compute_sha256(target_path)
        actual_size = target_path.stat().st_size if target_path.exists() else 0
        error_msg = (
            f"FATAL: Checkpoint verification failed for {target_path}!\n"
            f"Expected SHA-256: {LIPFORENSICS_CHECKPOINT_HASH}\n"
            f"Actual SHA-256:   {actual_hash}\n"
            f"File Size:        {actual_size} bytes"
        )
        print(error_msg, file=sys.stderr)
        sys.exit(1)

    print(f"[OK] Successfully verified LipForensics weights ({target_path.stat().st_size / (1024*1024):.1f} MB, SHA-256 verified).")


def main() -> None:
    weights_path = Path("models/weights/lipforensics_ff.pth")
    download_lipforensics_checkpoint(weights_path)


if __name__ == "__main__":
    main()
