import urllib.request
import json
import uuid
import time

target_file = r"C:\Users\hp\Downloads\videoplayback.mp4"
boundary = "----WebKitFormBoundary" + uuid.uuid4().hex

with open(target_file, "rb") as f:
    file_bytes = f.read()

header = f'--{boundary}\r\nContent-Disposition: form-data; name="video"; filename="videoplayback.mp4"\r\nContent-Type: video/mp4\r\n\r\n'.encode('utf-8')
footer = f'\r\n--{boundary}--\r\n'.encode('utf-8')
body = header + file_bytes + footer

req = urllib.request.Request(
    "http://127.0.0.1:8000/detect",
    data=body,
    headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "User-Agent": "SIGNAL-Diagnostic-Client/1.0"
    },
    method="POST"
)

print(f"Sending {len(file_bytes)} bytes to http://127.0.0.1:8000/detect...")
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        elapsed = time.time() - t0
        print(f"HTTP Status: {resp.status} (elapsed: {elapsed:.2f}s)")
        resp_body = resp.read().decode("utf-8")
        parsed = json.loads(resp_body)
        print("Response JSON:")
        print(json.dumps(parsed, indent=2))
except Exception as e:
    elapsed = time.time() - t0
    print(f"POST /detect failed after {elapsed:.2f}s with error: {e}")
