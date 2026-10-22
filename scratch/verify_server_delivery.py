import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/") as res:
    html = res.read().decode("utf-8")
    print("HTTP Status:", res.status)
    print("HTML Length:", len(html))
    print("Contains <style> and </style>:", ("<style>" in html) and ("</style>" in html))
    print("Contains hero section:", 'class="hero"' in html)
    print("Contains evidence panel:", 'id="tiltPanel"' in html)
    print("Contains workflow section:", 'id="workflow"' in html)
    print("Contains methodology section:", 'id="methodology"' in html)
    print("Contains architecture section:", 'id="architecture"' in html)
    print("Contains archive section:", 'id="archive"' in html)
    print("Contains report modal:", 'id="report-modal"' in html)
    print("Contains download pdf button:", 'id="modal-btn-download-pdf"' in html)
