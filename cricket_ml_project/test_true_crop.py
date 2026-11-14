import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')
cap = cv2.VideoCapture(r'C:\Users\hp\AppData\Local\Temp\tmpeql2ps9t.mp4')
frames = []
for _ in range(80):
    ret, f = cap.read()
    if not ret: break
    frames.append(f)
cap.release()

H, W = frames[0].shape[:2]

# Let's crop around Person 3 (cx=0.505, cy=0.393) with good padding
cx, cy, bw, bh = 0.505, 0.400, 0.08, 0.24
x1 = int(max(0, (cx - bw) * W))
x2 = int(min(W, (cx + bw) * W))
y1 = int(max(0, (cy - bh) * H))
y2 = int(min(H, (cy + bh) * H))

print(f"Striker True Crop: ({x1}, {y1}) to ({x2}, {y2}) -> size ({x2-x1}x{y2-y1})")

crops = [f[y1:y2, x1:x2] for f in frames[15:45]]
cv2.imwrite("true_striker_crop.jpg", crops[10])

# Run YOLO-pose on these true striker crops
yolo_res = model(crops, verbose=False, imgsz=384)
valid_kpts = 0
for r in yolo_res:
    if r.keypoints is not None and len(r.keypoints.xy) > 0:
        valid_kpts += 1

print(f"Out of 30 frames, YOLO detected pose in {valid_kpts} frames!")
