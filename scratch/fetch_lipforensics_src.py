import urllib.request
import json
import os

def fetch_file(repo, path, branch='master'):
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"ERROR: {e}"

# Get repo tree from GitHub API if possible or fetch standard files
files_to_check = [
    "models/spatiotemporal_net.py",
    "models/resnet.py",
    "models/tcn.py",
    "evaluate.py",
    "preprocessing/crop_mouth_from_video.py",
    "preprocessing/utils.py",
    "preprocessing/extract_landmarks.py",
    "data/transforms.py",
    "data/dataset.py",
    "options.py"
]

out_dir = r"C:\Users\hp\.gemini\antigravity-ide\brain\8eb498d9-82de-432c-bd00-797161a92588\scratch\lipforensics_src"
os.makedirs(out_dir, exist_ok=True)

print("=== FETCHING LIPFORENSICS OFFICIAL SOURCE FILES ===")
for f in files_to_check:
    content = fetch_file("ahaliassos/LipForensics", f)
    if not content.startswith("ERROR"):
        local_path = os.path.join(out_dir, os.path.basename(f))
        with open(local_path, "w", encoding="utf-8") as out:
            out.write(content)
        print(f"[FETCHED] {f} -> {len(content)} chars")
    else:
        print(f"[NOT FOUND] {f}: {content}")
