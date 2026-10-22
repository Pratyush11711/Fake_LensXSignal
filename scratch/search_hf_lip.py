import urllib.request
import json

def search(q):
    url = f'https://huggingface.co/api/models?search={q}&limit=10'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    print(f'HF search "{q}": {[m["id"] for m in data]}')

search('lipforensics')
search('haliassos')
search('lipreading')
