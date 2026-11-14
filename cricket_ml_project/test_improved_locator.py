import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(r'C:\Users\hp\AppData\Local\Temp\tmpeql2ps9t.mp4')
frames = []
for _ in range(60):
    ret, f = cap.read()
    if not ret: break
    frames.append(f)
cap.release()

H, W = frames[0].shape[:2]

# Test improved striker localization:
# 1. Probe frames at imgsz=1024, conf=0.15
probe_idxs = np.linspace(5, len(frames) - 5, min(8, len(frames))).astype(int)
probe_frames = [frames[i] for i in probe_idxs]
res_list = model(probe_frames, verbose=False, imgsz=1024, conf=0.15)

striker_candidates = []
for r in res_list:
    if r.boxes is None or len(r.boxes.xywhn) == 0:
        continue
    for b in r.boxes.xywhn.cpu().numpy():
        cx, cy, bw, bh = b
        # In broadcast view: striker is in the central corridor between y=0.28 and y=0.52
        if 0.40 <= cx <= 0.60 and 0.28 <= cy <= 0.52 and 0.10 <= bh <= 0.35:
            striker_candidates.append([cx, cy, bw, bh])

print(f"Found {len(striker_candidates)} striker candidate detections across probe frames!")
if striker_candidates:
    med = np.median(striker_candidates, axis=0)
    cx, cy, bw, bh = med
    print(f"Median striker position: cx={cx:.3f}, cy={cy:.3f}, bw={bw:.3f}, bh={bh:.3f}")
    hw, hh = max(bw * 1.5, 0.10), max(bh * 1.1, 0.18)
    x1, y1 = int(max(0, (cx - hw) * W)), int(max(0, (cy - hh) * H))
    x2, y2 = int(min(W, (cx + hw) * W)), int(min(H, (cy + hh) * H))
    print(f"Computed Crop Box: ({x1}, {y1}) to ({x2}, {y2}) -> size ({x2-x1}x{y2-y1})")
    crop = frames[20][y1:y2, x1:x2]
    cv2.imwrite("improved_striker_crop.jpg", crop)
    print("Saved improved_striker_crop.jpg!")
