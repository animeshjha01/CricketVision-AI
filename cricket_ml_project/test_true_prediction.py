import cv2
import numpy as np
import joblib
import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18
from ultralytics import YOLO
from train_model import map_to_core_shot, StackedEnsemble, build_hybrid_features

video_path = r'C:\Users\hp\AppData\Local\Temp\tmpeql2ps9t.mp4'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = R3D_18_Weights.DEFAULT
r3d_backbone = r3d_18(weights=weights)
r3d_backbone.fc = nn.Identity()
r3d_backbone = r3d_backbone.to(DEVICE).eval()
MEAN_3D = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1, 1)
STD_3D = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1, 1)

pose_model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(video_path)
raw_frames = []
while cap.isOpened() and len(raw_frames) < 120:
    ret, f = cap.read()
    if not ret: break
    raw_frames.append(f)
cap.release()

H, W = raw_frames[0].shape[:2]

# True striker box: (556, 105) to (766, 421)
x1, y1, x2, y2 = 556, 105, 766, 421
all_crops = [f[y1:y2, x1:x2] for f in raw_frames]

# Compute optical flow
grays = [cv2.cvtColor(cv2.resize(c, (96, 96)), cv2.COLOR_BGR2GRAY) for c in all_crops]
energies, flow_series = [0.0], []
for i in range(1, len(grays)):
    flow = cv2.calcOpticalFlowFarneback(grays[i - 1], grays[i], None, 0.5, 2, 11, 2, 5, 1.1, 0)
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.hypot(fx, fy)
    energies.append(float(np.mean(mag[16:80, 16:80])))
    grid_feats = []
    for r_idx in range(3):
        for c_idx in range(3):
            sub_fx = fx[r_idx * 32:(r_idx + 1) * 32, c_idx * 32:(c_idx + 1) * 32]
            sub_fy = fy[r_idx * 32:(r_idx + 1) * 32, c_idx * 32:(c_idx + 1) * 32]
            grid_feats.extend([float(np.mean(np.maximum(0.0, -sub_fx))), float(np.mean(np.maximum(0.0, sub_fx))),
                               float(np.mean(np.maximum(0.0, -sub_fy))), float(np.mean(np.maximum(0.0, sub_fy)))])
    flow_series.append(grid_feats)
flow_series.insert(0, flow_series[0])
all_flows = np.array(flow_series, dtype=np.float32)

SEQ_LEN = 16
peak_idx = int(np.argmax(energies))
half_w = SEQ_LEN // 2
start_w = max(0, min(len(raw_frames) - SEQ_LEN, peak_idx - half_w + 1))
sel_indices = np.arange(start_w, start_w + SEQ_LEN)
seq_crops = [all_crops[i] for i in sel_indices]
seq_flows = all_flows[sel_indices]

# YOLO11-pose on seq_crops
crop_yolo = pose_model(seq_crops, verbose=False, imgsz=256)
raw_kpts = []
for r in crop_yolo:
    if r.keypoints is not None and len(r.keypoints.xyn) > 0:
        kp = r.keypoints.xyn.cpu().numpy()[0].copy()
        zero_mask = (kp[:, 0] == 0) & (kp[:, 1] == 0)
        if np.sum(~zero_mask) >= 5:
            kp[zero_mask] = np.mean(kp[~zero_mask], axis=0)
            mid_hip = 0.5 * (kp[11] + kp[12]) if np.all(kp[[11, 12]] > 0) else np.mean(kp, axis=0)
            kp = (kp - mid_hip) / float(np.std(kp) + 1e-5)
            raw_kpts.append(kp.flatten())
        else: raw_kpts.append(np.zeros(34, dtype=np.float32))
    else: raw_kpts.append(np.zeros(34, dtype=np.float32))

pose_seq = np.array(raw_kpts, dtype=np.float32)
# 3D CNN
crops_112 = [cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (112, 112)) for c in seq_crops]
rgb_np = np.stack(crops_112, axis=0).astype(np.float32) / 255.0
diff_np = np.zeros_like(rgb_np)
diff_np[1:] = np.clip(np.abs(rgb_np[1:] - rgb_np[:-1]) * 4.0, 0.0, 1.0)
diff_np[0] = diff_np[1]
diff_tensor = (torch.from_numpy(diff_np).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE) - MEAN_3D) / STD_3D
motion_emb_3d = r3d_backbone(diff_tensor).cpu().detach().numpy().flatten().astype(np.float32)

flow_summary = np.concatenate([
    np.mean(seq_flows, axis=0),
    np.max(seq_flows, axis=0),
    np.mean(seq_flows[max(0, half_w - 2):min(SEQ_LEN, half_w + 3)], axis=0),
    np.mean(seq_flows[half_w:], axis=0) - np.mean(seq_flows[:max(1, half_w)], axis=0),
], axis=-1).astype(np.float32)
combined_emb = np.concatenate([flow_summary, motion_emb_3d], axis=-1)

# Now test model prediction
import sys
sys.modules["__main__"].StackedEnsemble = StackedEnsemble
bundle = joblib.load("shot_model.pkl")
core_model = bundle["core_model"]
sub_model = bundle["sub_model"]
core_classes = list(bundle["core_classes"])
sub_classes = list(bundle["sub_classes"])

feat = build_hybrid_features(pose_seq[np.newaxis, ...], combined_emb[np.newaxis, ...])
core_probs = core_model.predict_proba(feat)[0]
sub_probs = sub_model.predict_proba(feat)[0]

c_idx = np.argmax(core_probs)
s_idx = np.argmax(sub_probs)
print(f"PREDICTED CORE: {core_classes[c_idx]} ({core_probs[c_idx]*100:.1f}%)")
print(f"PREDICTED SUB:  {sub_classes[s_idx]} ({sub_probs[s_idx]*100:.1f}%)")

# Top 3 sub shots
top3 = np.argsort(sub_probs)[::-1][:3]
for idx in top3:
    print(f"  Top: {sub_classes[idx]} ({sub_probs[idx]*100:.1f}%)")
