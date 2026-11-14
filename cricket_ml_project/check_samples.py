import os
from app import analyze_video, SAMPLE_VIDEOS

for name, info in SAMPLE_VIDEOS.items():
    p = info['path']
    if os.path.exists(p):
        res = analyze_video(p, info['length'], info['line'])
        pred = res.get('shot_played')
        top3 = [t['name'] for t in res.get('top3', [])]
        print(f"Sample: {name}")
        print(f"  Path: {p}")
        print(f"  Model Prediction: {pred}")
        print(f"  Top 3: {top3}")
        print()
