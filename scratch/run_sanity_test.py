import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath('.'))
from src.lipforensics_video_inference import LipForensicsVideoInference

# Load DFD manifest to pick 1 known real and 1 known fake video
with open('D:/benchmark_dataset/manifest.json') as f:
    data = json.load(f)

real_entry = [v for v in data['videos'] if v['ground_truth'] == 'REAL'][0]
fake_entry = [v for v in data['videos'] if v['ground_truth'] == 'FAKE'][0]

print("Selected REAL diagnostic video:", real_entry['local_path'])
print("Selected FAKE diagnostic video:", fake_entry['local_path'])

engine = LipForensicsVideoInference()

print("\n=== RUNNING REAL VIDEO SANITY TEST ===")
t0 = time.time()
real_res = engine.analyze_video(real_entry['local_path'])
print("REAL Video Result:")
print(json.dumps(real_res, indent=2))

print("\n=== RUNNING FAKE VIDEO SANITY TEST ===")
t0 = time.time()
fake_res = engine.analyze_video(fake_entry['local_path'])
print("FAKE Video Result:")
print(json.dumps(fake_res, indent=2))

# Save sanity results to scratch for audit report creation
sanity_output = {
    "real_video": {
        "metadata": real_entry,
        "inference_result": real_res
    },
    "fake_video": {
        "metadata": fake_entry,
        "inference_result": fake_res
    }
}

with open(r'C:\Users\hp\.gemini\antigravity-ide\brain\8eb498d9-82de-432c-bd00-797161a92588\scratch\lipforensics_sanity_results.json', 'w') as f:
    json.dump(sanity_output, f, indent=2)

print("\nSanity tests completed and saved!")
