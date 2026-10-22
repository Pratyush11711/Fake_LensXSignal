import urllib.request
import concurrent.futures
import os
import time
import hashlib

def get_file_size(url):
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return int(resp.headers.get('Content-Length', 0))

def download_chunk(url, start, end, chunk_idx, part_file):
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0',
        'Range': f'bytes={start}-{end}'
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with open(part_file, 'wb') as f:
        f.write(data)
    return chunk_idx, len(data)

def fast_download(url, dest_path, num_workers=8):
    total_size = get_file_size(url)
    print(f"Total size: {total_size} bytes ({total_size / (1024*1024):.2f} MB)")
    if total_size == 0:
        return False
        
    chunk_size = total_size // num_workers
    futures = []
    part_files = []
    t0 = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            start = i * chunk_size
            end = total_size - 1 if i == num_workers - 1 else (i + 1) * chunk_size - 1
            part_file = f"{dest_path}.part{i}"
            part_files.append(part_file)
            futures.append(executor.submit(download_chunk, url, start, end, i, part_file))
            
        completed = 0
        for f in concurrent.futures.as_completed(futures):
            idx, b_len = f.result()
            completed += b_len
            print(f"Part {idx} done ({b_len / (1024*1024):.1f} MB) - Total: {completed / (1024*1024):.1f} MB ({completed / total_size * 100:.1f}%) in {time.time() - t0:.1f}s")
            
    # Combine parts
    print("Combining parts...")
    with open(dest_path, 'wb') as outfile:
        for p in part_files:
            with open(p, 'rb') as infile:
                outfile.write(infile.read())
            os.remove(p)
            
    elapsed = time.time() - t0
    print(f"Downloaded in {elapsed:.1f}s ({total_size / (1024*1024) / elapsed:.2f} MB/s)")
    sha256 = hashlib.sha256(open(dest_path, 'rb').read()).hexdigest()
    print(f"SHA-256: {sha256}")
    return True

if __name__ == '__main__':
    url = 'https://huggingface.co/Vansh180/VideoMae-ffc23-deepfake-detector/resolve/main/model.safetensors'
    dest = r'C:\Users\hp\.gemini\antigravity-ide\brain\8eb498d9-82de-432c-bd00-797161a92588\scratch\videomae_ffc23_fast.safetensors'
    fast_download(url, dest, num_workers=16)
