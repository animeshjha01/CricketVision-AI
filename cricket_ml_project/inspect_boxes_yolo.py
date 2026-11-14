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
results = model(frames[20:30], verbose=False, imgsz=640)

print(f"Video size: {W}x{H}")
for f_idx, r in enumerate(results):
    print(f"\n--- Frame {f_idx+20} ---")
    if r.boxes is not None:
        boxes_xywhn = r.boxes.xywhn.cpu().numpy()
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for i, (b, b_px, c) in enumerate(zip(boxes_xywhn, boxes_xyxy, confs)):
            cx, cy, bw, bh = b
            y1, y2 = cy - bh/2.0, cy + bh/2.0
            print(f"  Person {i}: conf={c:.2f}, cx={cx:.3f}, cy={cy:.3f}, bw={bw:.3f}, bh={bh:.3f}, y1={y1:.3f}, y2={y2:.3f} | Pixel: {b_px.astype(int)}")
