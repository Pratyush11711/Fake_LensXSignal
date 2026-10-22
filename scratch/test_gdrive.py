import urllib.request
import urllib.parse
import re
import http.cookiejar
import os

file_id = '1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7'
url = f'https://drive.google.com/uc?export=download&id={file_id}'
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
with opener.open(req) as resp:
    html = resp.read().decode('utf-8', errors='ignore')

print('HTML length:', len(html))
# Find form action
form_match = re.search(r'<form[^>]*action=([^ >]+)', html)
if form_match:
    action = form_match.group(1).strip('"\'')
    print('Action URL:', action)
    inputs = dict(re.findall(r'<input[^>]*name=([^ >]+)[^>]*value=([^ >]+)', html))
    print('Inputs:', inputs)
    clean_inputs = {}
    for k, v in inputs.items():
        clean_inputs[k.strip('"\'')] = v.strip('"\'')
    
    download_url = action + '?' + urllib.parse.urlencode(clean_inputs)
    print('Final download URL:', download_url[:150])
    
    req2 = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
    dest = r'C:\Users\hp\.gemini\antigravity-ide\brain\8eb498d9-82de-432c-bd00-797161a92588\scratch\lipforensics_ff.pth'
    with opener.open(req2) as resp2:
        print('Resp headers:', dict(resp2.headers))
        content = resp2.read()
        print('Downloaded bytes:', len(content))
        if len(content) > 100000:
            with open(dest, 'wb') as f:
                f.write(content)
            print('Successfully saved LipForensics checkpoint!')
