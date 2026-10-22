import urllib.request
import urllib.parse
import json
import hashlib
import os
import torch

def fetch_text(url, timeout=10):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"ERROR: {e}"

def check_hf_model(model_id):
    url = f"https://huggingface.co/api/models/{model_id}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        files = [s['rfilename'] for s in data.get('siblings', [])]
        card = data.get('cardData', {})
        return {
            'found': True,
            'pipeline_tag': data.get('pipeline_tag'),
            'downloads': data.get('downloads'),
            'likes': data.get('likes'),
            'files': files,
            'card': card
        }
    except Exception as e:
        return {'found': False, 'error': str(e)}

print("==================================================")
print("1. ALTFREEZING DEEP AUDIT")
print("==================================================")
# Check AltFreezing repo links
alt_readme = fetch_text('https://raw.githubusercontent.com/ZhendongWang6/AltFreezing/main/README.md')
print("AltFreezing README excerpt:")
for l in alt_readme.splitlines()[:30]:
    if 'drive' in l.lower() or 'recdrive' in l.lower() or 'baidu' in l.lower() or 'checkpoint' in l.lower() or 'weight' in l.lower() or 'password' in l.lower():
        print("  ", l)

# Test if RecDrive is directly downloadable via curl/python
recdrive_url = "https://rec.ustc.edu.cn/share/e87360b0-7b2e-11ef-aeef-a9fd0832d537"
rec_resp = fetch_text(recdrive_url)
print(f"RecDrive response (length {len(rec_resp)}): {rec_resp[:200]}...")

print("\n==================================================")
print("2. STIL DEEP AUDIT")
print("==================================================")
# Check STIL / SPLIT
stil_repo = fetch_text('https://raw.githubusercontent.com/mldljyh/SPLIT/main/README.md')
print("STIL/SPLIT README download info:")
for l in stil_repo.splitlines():
    if any(k in l.lower() for k in ['checkpoint', 'weight', 'pth', 'download', 'drive']):
        print("  ", l)

print("\n==================================================")
print("3. LIPFORENSICS AUDIT")
print("==================================================")
lip_readme = fetch_text('https://raw.githubusercontent.com/ahaliassos/LipForensics/master/README.md')
print("LipForensics README download info:")
for l in lip_readme.splitlines():
    if any(k in l.lower() for k in ['checkpoint', 'weight', 'pth', 'download', 'gdrive', 'google drive', 'dropbox', 'box', 'license']):
        print("  ", l)

print("\n==================================================")
print("4. HUGGING FACE VIDEO DEEPFAKE DETECTORS AUDIT")
print("==================================================")
candidates_hf = [
    "Vansh180/VideoMae-ffc23-deepfake-detector",
    "shylhy/videomae-large-finetuned-deepfake-subset",
    "ash12321/deepfake-detector-timesformer",
    "mhamza-007/timesformer-deepfake-detection",
    "facebook/timesformer-base-finetuned-k400",
    "MCG-NJU/videomae-base"
]
for c in candidates_hf:
    info = check_hf_model(c)
    print(f"\nModel: {c}")
    print(f"  Found: {info.get('found')}")
    if info.get('found'):
        print(f"  Files: {info.get('files')}")
        print(f"  Pipeline: {info.get('pipeline_tag')}")
        print(f"  Downloads: {info.get('downloads')}")
