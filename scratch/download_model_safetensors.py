import urllib.request
import os
import sys
import hashlib
import time

url = "https://huggingface.co/Vansh180/VideoMae-ffc23-deepfake-detector/resolve/main/model.safetensors"
dest = r"C:\Users\hp\.gemini\antigravity-ide\brain\8eb498d9-82de-432c-bd00-797161a92588\scratch\videomae_ffc23.safetensors"

print(f"Downloading {url} -> {dest}", flush=True)
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        total_size = int(resp.headers.get('Content-Length', 0))
        print(f"Total size: {total_size} bytes ({total_size / (1024*1024):.2f} MB)", flush=True)
        downloaded = 0
        chunk_size = 2 * 1024 * 1024 # 2MB
        start_t = time.time()
        with open(dest, 'wb') as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                pct = (downloaded / total_size * 100) if total_size else 0
                elapsed = time.time() - start_t
                speed = (downloaded / (1024*1024)) / elapsed if elapsed > 0 else 0
                print(f"Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", flush=True)

    print("Download finished!", flush=True)
    # Compute SHA-256
    print("Computing SHA-256...", flush=True)
    sha256 = hashlib.sha256()
    with open(dest, 'rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            sha256.update(chunk)
    print(f"SHA-256: {sha256.hexdigest()}", flush=True)
    print(f"Final file size: {os.path.getsize(dest)} bytes", flush=True)
except Exception as e:
    print(f"Error downloading: {e}", flush=True)
