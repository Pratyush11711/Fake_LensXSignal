import urllib.request
import json
import uuid

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

try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        print("HTTP Status:", resp.status)
        resp_body = resp.read().decode("utf-8")
        print("Response JSON:")
        print(json.dumps(json.loads(resp_body), indent=2))
except Exception as e:
    print("POST /detect failed:", e)
