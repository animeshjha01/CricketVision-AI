import cv2
import numpy as np
from extract_data import _locate_striker_box, _trim_before_camera_pan, extract_sequence

video_path = r'C:\Users\hp\AppData\Local\Temp\tmpeql2ps9t.mp4'

cap = cv2.VideoCapture(video_path)
frames = []
while cap.isOpened() and len(frames) < 120:
    ret, f = cap.read()
    if not ret: break
    frames.append(f)
cap.release()

print(f"Total frames read: {len(frames)}")
trimmed = _trim_before_camera_pan(frames)
print(f"Trimmed frames: {len(trimmed)}")

H, W = trimmed[0].shape[:2]
x1, y1, x2, y2 = _locate_striker_box(trimmed)
print(f"Frame dimensions: {W}x{H}")
print(f"Crop box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
print(f"Crop center relative: cx={(x1+x2)/(2*W):.3f}, cy={(y1+y2)/(2*H):.3f}")
print(f"Crop box size: width={x2-x1}, height={y2-y1}")

# Run extract_sequence with return_vis=True
pose_seq, video_emb, vis = extract_sequence(video_path, return_vis=True)
cv2.imwrite("debug_crop_vis.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print("Saved debug_crop_vis.jpg!")
