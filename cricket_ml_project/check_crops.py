import cv2
import numpy as np
from pathlib import Path
from extract_data import _locate_striker_box, _trim_before_camera_pan

samples = [
    ("Cover Drive", "Cricket Shots/Cover Drive/Screen Recording 2026-10-14 003430.mp4"),
    ("Backfoot Punch", "Cricket Shots/Backfoot Punch/Screen Recording 2026-10-14 002446.mp4"),
    ("Front Defense", "Cricket Shots/Front-foot Defense/FD3.mp4"),
    ("Flick Shot", "Cricket Shots/Flick Shot/Screen Recording 2026-10-14 013130.mp4"),
    ("Pull Shot", "Cricket Shots/Pull Shot/Screen Recording 2026-10-13 233206.mp4"),
    ("Sweep", "Cricket Shots/Sweep/Screen Recording 2026-10-16 020723.mp4"),
]

for label, p in samples:
    cap = cv2.VideoCapture(p)
    frames = []
    for _ in range(60):
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    trimmed = _trim_before_camera_pan(frames)
    H, W = trimmed[0].shape[:2]
    x1, y1, x2, y2 = _locate_striker_box(trimmed)
    print(f"{label:16s} ({W}x{H}): Crop=({x1}, {y1}, {x2}, {y2}), Size=({x2-x1}x{y2-y1}), Center=({(x1+x2)/(2*W):.2f}, {(y1+y2)/(2*H):.2f})")
