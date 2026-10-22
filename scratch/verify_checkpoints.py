import urllib.request
import urllib.parse
import http.cookiejar
import json
import hashlib
import os
import torch
import sys

def p(msg):
    print(msg, flush=True)

# 1. Download LipForensics weights from Google Drive
def download_gdrive(file_id, dest_path):
    p(f"Attempting GDrive download for ID: {file_id} -> {dest_path}")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with opener.open(req) as resp:
        content = resp.read()
        
    # Check if we got the confirm token
    if b"confirm=" in content or b"Google Drive - Virus scan warning" in content or b"download_warning" in content:
        p("Received confirmation page, extracting token...")
        text = content.decode('utf-8', errors='ignore')
        import re
        confirm_match = re.search(r'confirm=([0-9A-Za-z_-]+)', text)
        if confirm_match:
            confirm_token = confirm_match.group(1)
            confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
            req2 = urllib.request.Request(confirm_url, headers={'User-Agent': 'Mozilla/5.0'})
            with opener.open(req2) as resp2:
                content = resp2.read()
    
    if len(content) < 10000 and (b"<html" in content or b"<!DOCTYPE" in content):
        p(f"Failed to download binary; received HTML of length {len(content)}")
        return False
    
    with open(dest_path, "wb") as f:
        f.write(content)
    p(f"Successfully downloaded {len(content)} bytes to {dest_path}")
    return True

# 2. Download Hugging Face model file
def download_hf_file(repo_id, filename, dest_path):
    p(f"Attempting HF download: {repo_id}/{filename} -> {dest_path}")
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            content = resp.read()
        with open(dest_path, "wb") as f:
            f.write(content)
        p(f"Successfully downloaded {len(content)} bytes to {dest_path}")
        return True
    except Exception as e:
        p(f"HF download error: {e}")
        return False

scratch_dir = r"C:\Users\hp\.gemini\antigravity-ide\brain\8eb498d9-82de-432c-bd00-797161a92588\scratch"
os.makedirs(scratch_dir, exist_ok=True)

# Test LipForensics
lip_dest = os.path.join(scratch_dir, "lipforensics_ff.pth")
if not os.path.exists(lip_dest):
    success_lip = download_gdrive("1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7", lip_dest)
else:
    p(f"LipForensics file already exists at {lip_dest}")
    success_lip = True

if os.path.exists(lip_dest) and os.path.getsize(lip_dest) > 10000:
    sha = hashlib.sha256(open(lip_dest, "rb").read()).hexdigest()
    size = os.path.getsize(lip_dest)
    p(f"\n[LIPFORENSICS CHECKPOINT VERIFIED]")
    p(f"  Path: {lip_dest}")
    p(f"  Size: {size} bytes ({size / (1024*1024):.2f} MB)")
    p(f"  SHA-256: {sha}")
    try:
        sd = torch.load(lip_dest, map_location="cpu")
        if isinstance(sd, dict):
            keys = list(sd.keys())
            p(f"  State dict keys count: {len(keys)}")
            p(f"  Sample keys: {keys[:5]}")
            if 'model_state_dict' in sd:
                inner_keys = list(sd['model_state_dict'].keys())
                p(f"  Inner model_state_dict keys count: {len(inner_keys)}")
                p(f"  Sample inner keys: {inner_keys[:5]}")
    except Exception as e:
        p(f"  PyTorch load error: {e}")

# Test VideoMAE deepfake detector (Vansh180/VideoMae-ffc23-deepfake-detector)
vmae_dest = os.path.join(scratch_dir, "videomae_ffc23_model.safetensors")
if not os.path.exists(vmae_dest):
    success_vmae = download_hf_file("Vansh180/VideoMae-ffc23-deepfake-detector", "model.safetensors", vmae_dest)
else:
    p(f"VideoMAE file already exists at {vmae_dest}")

if os.path.exists(vmae_dest) and os.path.getsize(vmae_dest) > 10000:
    sha = hashlib.sha256(open(vmae_dest, "rb").read()).hexdigest()
    size = os.path.getsize(vmae_dest)
    p(f"\n[VIDEOMAE FFC23 CHECKPOINT VERIFIED]")
    p(f"  Path: {vmae_dest}")
    p(f"  Size: {size} bytes ({size / (1024*1024):.2f} MB)")
    p(f"  SHA-256: {sha}")
    try:
        from safetensors.torch import load_file
        sd = load_file(vmae_dest)
        keys = list(sd.keys())
        p(f"  Safetensors keys count: {len(keys)}")
        p(f"  Sample keys: {keys[:5]}")
        p(f"  Classifier head shape: {sd.get('classifier.weight', getattr(sd.get('head.weight'), 'shape', 'N/A'))}")
    except Exception as e:
        p(f"  Safetensors load error: {e}")
