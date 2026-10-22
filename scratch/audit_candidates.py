import urllib.request
import urllib.error
import json
import re

def fetch(url, headers=None, timeout=5):
    h = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"ERROR: {e}"

print("=== 1. ALTFREEZING REPOSITORY AUDIT ===")
alt_readme = fetch('https://raw.githubusercontent.com/ZhendongWang6/AltFreezing/main/README.md')
print("AltFreezing README lines:")
for line in alt_readme.splitlines():
    if any(w in line.lower() for w in ['http', 'download', 'weight', 'checkpoint', 'drive', 'baidu', 'pan', 'model', 'trained', 'eval', 'test']):
        print("  ", line)

print("\n=== 2. STIL REPOSITORY AUDIT ===")
stil_readme = fetch('https://raw.githubusercontent.com/mldljyh/SPLIT/main/README.md')
print("STIL/SPLIT README lines:")
for line in stil_readme.splitlines():
    if any(w in line.lower() for w in ['http', 'download', 'weight', 'checkpoint', 'drive', 'baidu', 'pan', 'model', 'trained', 'eval', 'test']):
        print("  ", line)

print("\n=== 3. DEEPFAKEBENCH PRETRAINED WEIGHTS AUDIT ===")
dfb_weights = fetch('https://raw.githubusercontent.com/SCLBD/DeepfakeBench/main/training/weights/README.md')
print("DeepfakeBench weights README:", dfb_weights[:400])

dfb_readme = fetch('https://raw.githubusercontent.com/SCLBD/DeepfakeBench/main/README.md')
print("DeepfakeBench main README download mentions:")
for line in dfb_readme.splitlines():
    if any(w in line.lower() for w in ['weights', 'pretrained', 'google drive', 'hugging face', 'download', 'checkpoint', 'altfreezing', 'stil', 'timesformer', 'videomae']):
        print("  ", line[:120])
