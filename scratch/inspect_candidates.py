import urllib.request
import urllib.parse
import json
import os
import sys

def p(msg):
    print(msg, flush=True)

def fetch(url, timeout=5):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"ERROR: {e}"

p("=== 1. LIPFORENSICS LICENSE & DETAILS ===")
lip_lic = fetch('https://raw.githubusercontent.com/ahaliassos/LipForensics/master/LICENSE')
p(f"LipForensics LICENSE: {lip_lic[:300]}")

p("\n=== 2. ALTFREEZING REPO LICENSE ===")
alt_lic = fetch('https://raw.githubusercontent.com/ZhendongWang6/AltFreezing/main/LICENSE')
p(f"AltFreezing LICENSE: {alt_lic[:300]}")

p("\n=== 3. STIL / SPLIT LICENSE ===")
stil_lic = fetch('https://raw.githubusercontent.com/mldljyh/SPLIT/main/LICENSE')
p(f"STIL/SPLIT LICENSE: {stil_lic[:300]}")

p("\n=== 4. TEST DIRECT DOWNLOAD FOR LIPFORENSICS PTH ===")
# Test gdown/direct drive download URL for 1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7
gdrive_id = "1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7"
gdrive_url = f"https://drive.google.com/uc?id={gdrive_id}&export=download"
try:
    req = urllib.request.Request(gdrive_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        headers = dict(resp.headers)
        p(f"GDrive response content-type: {headers.get('Content-Type')}, content-length: {headers.get('Content-Length')}")
except Exception as e:
    p(f"GDrive direct test error: {e}")

p("\n=== 5. CHECK HUGGING FACE VIDEO DETECTORS ===")
hf_models = [
    "Vansh180/VideoMae-ffc23-deepfake-detector",
    "shylhy/videomae-large-finetuned-deepfake-subset",
    "ash12321/deepfake-detector-timesformer",
    "facebook/timesformer-base-finetuned-k400",
    "MCG-NJU/videomae-base"
]
for m in hf_models:
    res = fetch(f"https://huggingface.co/api/models/{m}")
    try:
        data = json.loads(res)
        files = [s['rfilename'] for s in data.get('siblings', [])]
        p(f"HF Model: {m} | Files: {files} | Downloads: {data.get('downloads')}")
    except Exception as e:
        p(f"HF Model: {m} | Error: {res[:100]}")
