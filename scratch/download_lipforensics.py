import subprocess
import os
import re
import hashlib
import torch

dest = r"C:\Users\hp\.gemini\antigravity-ide\brain\8eb498d9-82de-432c-bd00-797161a92588\scratch\lipforensics_ff.pth"
cookie_file = r"C:\Users\hp\.gemini\antigravity-ide\brain\8eb498d9-82de-432c-bd00-797161a92588\scratch\cookie.txt"
file_id = "1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7"

cmd1 = ['curl', '-c', cookie_file, '-s', '-L', f'https://drive.google.com/uc?export=download&id={file_id}', '-o', f'{dest}.html']
subprocess.run(cmd1)

with open(f'{dest}.html', 'r', encoding='utf-8', errors='ignore') as f:
    html = f.read()

print(f'Initial HTML length: {len(html)}')
# Look for form or download link
form_match = re.search(r'<form[^>]*action=[\'"]([^\'"]+)[\'"]', html)
if form_match:
    action = form_match.group(1)
    print(f'Form action: {action}')
    inputs = re.findall(r'<input[^>]*name=[\'"]([^\'"]+)[\'"][^>]*value=[\'"]([^\'"]*)[\'"]', html)
    print(f'Form inputs: {inputs}')
    params = "&".join([f"{k}={v}" for k, v in inputs])
    download_url = f"{action}?{params}" if params else action
    print(f'Downloading from: {download_url[:120]}...')
    cmd2 = ['curl', '-b', cookie_file, '-L', download_url, '-o', dest]
    subprocess.run(cmd2)
else:
    # Look for confirm code directly
    match = re.search(r'confirm=([0-9A-Za-z_-]+)', html)
    if match:
        code = match.group(1)
        print(f'Confirm code: {code}')
        cmd2 = ['curl', '-b', cookie_file, '-L', f'https://drive.google.com/uc?export=download&confirm={code}&id={file_id}', '-o', dest]
        subprocess.run(cmd2)

if os.path.exists(dest):
    size = os.path.getsize(dest)
    print(f'Downloaded size: {size} bytes ({size / (1024*1024):.2f} MB)')
    if size > 100000:
        sha256 = hashlib.sha256(open(dest, 'rb').read()).hexdigest()
        print(f'SHA-256: {sha256}')
        try:
            sd = torch.load(dest, map_location='cpu')
            print(f'Loaded checkpoint successfully! Type: {type(sd)}')
            if isinstance(sd, dict):
                print(f'Keys count: {len(sd.keys())}')
                print(f'Sample keys: {list(sd.keys())[:5]}')
        except Exception as e:
            print(f'PyTorch error: {e}')
