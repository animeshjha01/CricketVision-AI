"""
extract_data.py — turns a batting clip into (pose_seq, video_emb) features.

Pipeline (v2):
  1. Read the WHOLE clip (up to 20 s) at ~15 FPS, strip letterbox bars, mirror left-handers.
  2. striker_tracker.locate_striker(): track every person, pick the batter by behaviour,
     follow them through camera zooms, and find the shot moment (batter's peak wrist speed).
  3. Cut a 16-frame window around that moment (6 before, impact, 9 after) and, for every
     frame, a square crop that FOLLOWS the batter (sized from their body height).
  4. Pose (YOLO11-pose on the crop), camera-compensated optical flow, and a 3D-CNN motion
     embedding are computed on those crops. Output shapes are unchanged:
        pose_seq  : (16, 34)  crop-normalised keypoints, centred on the mid-hip
        video_emb : (656,)    144 optical-flow summary + 512 r3d_18 motion embedding
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18
from ultralytics import YOLO

from config import DATASET_DIR, DATA_NPZ_PATH, VALID_EXTS, clean_label
from striker_tracker import crop_box_for, locate_striker, read_frames, smooth_boxes

SEQ_LEN = 16
PRE_IMPACT = 6
CROP_PX = 224

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    YOLO_DEV = 0
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    YOLO_DEV = "mps"
else:
    DEVICE = torch.device("cpu")
    YOLO_DEV = "cpu"

_HERE = Path(__file__).resolve().parent
pose_model = YOLO(str(_HERE / "yolo11n-pose.pt"))          # used for tracking (stateful)
crop_pose_model = YOLO(str(_HERE / "yolo11n-pose.pt"))     # used for per-crop pose (stateless)

weights = R3D_18_Weights.DEFAULT
r3d_backbone = r3d_18(weights=weights)
r3d_backbone.fc = nn.Identity()
r3d_backbone = r3d_backbone.to(DEVICE).eval()

MEAN_3D = torch.tensor([0.43216, 0.394666, 0.37645], device=DEVICE).view(1, 3, 1, 1, 1)
STD_3D = torch.tensor([0.22803, 0.22145, 0.216989], device=DEVICE).view(1, 3, 1, 1, 1)

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]


def is_left_handed(handedness: str) -> bool:
    h = str(handedness).lower()
    return "left" in h or "lhb" in h


def compute_video_fingerprint(video_path: str) -> str:
    """Kept for backwards compatibility (no longer used to override predictions)."""
    try:
        p = Path(video_path)
        size = p.stat().st_size
        h = hashlib.md5()
        with open(p, "rb") as f:
            h.update(f.read(65536))
            if size > 65536:
                f.seek(max(0, size - 65536))
                h.update(f.read(65536))
        return f"{size}_{h.hexdigest()}"
    except Exception:
        return ""


def _compute_optical_flow_features(crops_bgr: list) -> tuple:
    """Camera-compensated Farneback flow on 96x96 crops -> (energies (T,), 3x3x4 grid flow (T, 36))."""
    grays = [cv2.cvtColor(cv2.resize(c, (96, 96)), cv2.COLOR_BGR2GRAY) for c in crops_bgr]
    flow_series = []
    energies = [0.0]

    border_mask = np.ones((96, 96), dtype=bool)
    border_mask[12:84, 12:84] = False

    for i in range(1, len(grays)):
        flow = cv2.calcOpticalFlowFarneback(grays[i - 1], grays[i], None, 0.5, 2, 11, 2, 5, 1.1, 0)
        bg_vx = float(np.median(flow[border_mask, 0]))
        bg_vy = float(np.median(flow[border_mask, 1]))
        fx = flow[..., 0] - bg_vx
        fy = flow[..., 1] - bg_vy

        mag = np.hypot(fx, fy)
        active = mag > 0.35
        fx = np.where(active, fx, 0.0)
        fy = np.where(active, fy, 0.0)
        energies.append(float(np.mean(mag[16:80, 16:80])))

        grid_feats = []
        for r_idx in range(3):
            for c_idx in range(3):
                sub_fx = fx[r_idx * 32:(r_idx + 1) * 32, c_idx * 32:(c_idx + 1) * 32]
                sub_fy = fy[r_idx * 32:(r_idx + 1) * 32, c_idx * 32:(c_idx + 1) * 32]
                grid_feats.extend([
                    float(np.mean(np.maximum(0.0, -sub_fx))),   # off side (for a RHB, behind-bowler view)
                    float(np.mean(np.maximum(0.0, sub_fx))),    # leg side
                    float(np.mean(np.maximum(0.0, -sub_fy))),   # up / loft
                    float(np.mean(np.maximum(0.0, sub_fy))),    # down / ground
                ])
        flow_series.append(grid_feats)

    if not flow_series:
        flow_series = [[0.0] * 36]
    flow_series.insert(0, flow_series[0])
    return np.array(energies, dtype=np.float32), np.array(flow_series, dtype=np.float32)


def _fallback_track(frames: list) -> dict:
    """No person tracked: use a centred crop and the frame-difference peak as the impact."""
    H, W = frames[0].shape[:2]
    small = [cv2.cvtColor(cv2.resize(f, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    motion = [0.0] + [float(np.mean(np.abs(small[i] - small[i - 1]))) for i in range(1, len(small))]
    box = np.array([0.30 * W, 0.15 * H, 0.70 * W, 0.85 * H], dtype=np.float32)
    return dict(boxes=np.tile(box, (len(frames), 1)), impact_t=int(np.argmax(motion)), fallback=True)


def _select_window(n_frames: int, impact_t: int, seg: np.ndarray = None) -> tuple:
    # Keep the window inside the impact's own scene segment so a broadcast cut
    # (replay, graphic, crowd reaction) right after the shot doesn't bleed into the window.
    lo, hi = 0, n_frames - 1
    if seg is not None and len(seg) == n_frames:
        in_seg = np.where(seg == seg[impact_t])[0]
        if len(in_seg):
            lo, hi = int(in_seg[0]), int(in_seg[-1])

    span = hi - lo + 1
    if span >= SEQ_LEN:
        start = int(np.clip(impact_t - PRE_IMPACT, lo, hi - SEQ_LEN + 1))
        idx = np.arange(start, start + SEQ_LEN)
    elif n_frames >= SEQ_LEN:
        # segment too short around impact: fall back to a window centred in the full clip,
        # still anchored as close to impact as the clip allows
        start = int(np.clip(impact_t - PRE_IMPACT, 0, n_frames - SEQ_LEN))
        idx = np.arange(start, start + SEQ_LEN)
    else:
        idx = np.linspace(0, n_frames - 1, SEQ_LEN).round().astype(int)
    impact_local = int(np.argmin(np.abs(idx - impact_t)))
    return idx, impact_local


def _crop_pose(crops: list) -> tuple:
    """Pose on each square crop; returns (T,17,2) normalised coords, (T,17) confidences."""
    res = crop_pose_model(crops, verbose=False, device=YOLO_DEV, imgsz=320, conf=0.10)
    T = len(crops)
    kp = np.zeros((T, 17, 2), dtype=np.float32)
    kc = np.zeros((T, 17), dtype=np.float32)
    prev = np.array([0.5, 0.5], dtype=np.float32)
    for t, r in enumerate(res):
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            continue
        b = r.boxes.xywhn.cpu().numpy()
        # the batter is the person nearest the crop centre / previous position, preferring larger boxes
        d = np.hypot(b[:, 0] - prev[0], b[:, 1] - prev[1]) - 0.3 * b[:, 3]
        i = int(np.argmin(d))
        kp[t] = r.keypoints.xyn.cpu().numpy()[i]
        kc[t] = (r.keypoints.conf.cpu().numpy()[i] if r.keypoints.conf is not None else 1.0)
        prev = b[i, :2]
    return kp, kc


def _clean_pose(kp: np.ndarray, kc: np.ndarray) -> np.ndarray:
    """Interpolate low-confidence joints through time, centre on mid-hip. Returns (T, 34)."""
    T = kp.shape[0]
    out = kp.copy()
    tt = np.arange(T)
    for j in range(17):
        ok = kc[:, j] > 0.30
        if ok.sum() >= 2:
            for d in range(2):
                out[:, j, d] = np.interp(tt, tt[ok], kp[ok, j, d])
        elif ok.sum() == 1:
            out[:, j] = kp[ok, j][0]
        else:
            out[:, j] = np.nan
    if np.all(np.isnan(out)):
        return np.zeros((T, 34), dtype=np.float32)
    # joints never seen: put them at the mean of the visible joints of that frame
    for t in range(T):
        miss = np.isnan(out[t, :, 0])
        if miss.any():
            out[t, miss] = np.nanmean(out[t], axis=0)
    mid_hip = 0.5 * (out[:, 11] + out[:, 12])
    out = out - mid_hip[:, None, :]
    return out.reshape(T, 34).astype(np.float32)


@torch.no_grad()
def extract_sequence(video_path: str, handedness: str = "Right-Handed (RHB)", return_vis: bool = False,
                     return_debug: bool = False):
    frames, _eff_fps = read_frames(video_path, mirror=is_left_handed(handedness))
    if len(frames) < 5:
        return (None, None, None) if return_vis else (None, None)

    H, W = frames[0].shape[:2]
    trk = locate_striker(pose_model, frames, YOLO_DEV)
    if trk is None or np.all(np.isnan(trk["boxes"][:, 0])):
        trk = _fallback_track(frames)

    boxes = smooth_boxes(trk["boxes"])
    sel, impact_local = _select_window(len(frames), int(trk["impact_t"]), trk.get("seg"))

    crop_boxes = [crop_box_for(boxes[i], W, H) for i in sel]
    crops = []
    for i, (x1, y1, x2, y2) in zip(sel, crop_boxes):
        c = frames[i][y1:y2, x1:x2]
        if c.size == 0:
            c = frames[i]
        crops.append(cv2.resize(c, (CROP_PX, CROP_PX)))

    kp, kc = _crop_pose(crops)
    pose_seq = _clean_pose(kp, kc)

    _energies, flows = _compute_optical_flow_features(crops)

    crops_112 = [cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (112, 112)) for c in crops]
    rgb_np = np.stack(crops_112, axis=0).astype(np.float32) / 255.0
    diff_np = np.zeros_like(rgb_np)
    diff_np[1:] = np.clip(np.abs(rgb_np[1:] - rgb_np[:-1]) * 4.0, 0.0, 1.0)
    diff_np[0] = diff_np[1]
    diff_tensor = (torch.from_numpy(diff_np).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE) - MEAN_3D) / STD_3D
    motion_emb_3d = r3d_backbone(diff_tensor).cpu().numpy().flatten().astype(np.float32)

    il = impact_local
    flow_summary = np.concatenate([
        np.mean(flows, axis=0),
        np.max(flows, axis=0),
        np.mean(flows[max(0, il - 2):min(SEQ_LEN, il + 3)], axis=0),
        np.mean(flows[il:], axis=0) - np.mean(flows[:max(1, il)], axis=0),
    ], axis=-1).astype(np.float32)
    combined_emb = np.concatenate([flow_summary, motion_emb_3d], axis=-1)

    if not (return_vis or return_debug):
        return pose_seq, combined_emb

    # Visualisation: impact frame with the batter crop and the skeleton drawn on it
    vis_f = frames[sel[il]].copy()
    x1, y1, x2, y2 = crop_boxes[il]
    cv2.rectangle(vis_f, (x1, y1), (x2, y2), (0, 255, 120), 3)
    cv2.putText(vis_f, "BATTER @ IMPACT" + (" (fallback)" if trk.get("fallback") else ""),
                (x1 + 4, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
    pts = {}
    for j in range(17):
        if kc[il, j] > 0.30:
            px = int(x1 + kp[il, j, 0] * (x2 - x1))
            py = int(y1 + kp[il, j, 1] * (y2 - y1))
            pts[j] = (px, py)
            cv2.circle(vis_f, (px, py), 4, (0, 215, 255), -1)
    for a, b in SKELETON_EDGES:
        if a in pts and b in pts:
            cv2.line(vis_f, pts[a], pts[b], (16, 185, 129), 2)
    vis_rgb = cv2.cvtColor(vis_f, cv2.COLOR_BGR2RGB)

    if return_debug:
        strip = np.hstack([cv2.resize(c, (112, 112)) for c in crops])
        return pose_seq, combined_emb, vis_rgb, strip
    return pose_seq, combined_emb, vis_rgb


def build_dataset(debug_dir: str = "debug_crops", augment_mirror: bool = False):
    """augment_mirror=True also extracts a horizontally-mirrored copy of every clip
    (forced via the opposite handedness flag), doubling the dataset with physically
    valid "played by the other hand" samples of the same shot label. Pose features are
    already hip-centred per-crop, so a mirrored copy is a free, label-preserving
    augmentation that also makes the model robust to a wrong handedness selection."""
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Folder '{DATASET_DIR}' not found.")

    print(f"Extracting batter-tracked Pose + Optical Flow + Motion-3D features on {DEVICE}...")
    X_pose, X_emb, y, classes, display_names, paths = [], [], [], [], {}, []
    golden_templates = {}
    dbg = Path(debug_dir)

    subfolders = sorted([f for f in DATASET_DIR.iterdir() if f.is_dir()])
    for label_idx, folder in enumerate(subfolders):
        norm_name = clean_label(folder.name)
        classes.append(norm_name)
        display_names[norm_name] = folder.name
        shot_poses = []
        (dbg / norm_name).mkdir(parents=True, exist_ok=True)

        video_files = sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS])
        print(f"\nProcessing '{folder.name}' ({len(video_files)} videos)...", flush=True)

        for vid_path in video_files:
            variants = [("Right-Handed (RHB)", "")]
            if augment_mirror:
                variants.append(("Left-Handed (LHB)", "_mirror"))
            for handedness, tag in variants:
                try:
                    out = extract_sequence(vid_path, handedness=handedness, return_debug=True)
                except Exception as ex:  # one broken file must not kill a 15-minute extraction
                    print(f"  [ERROR] {vid_path.name}{tag}: {ex}")
                    continue
                pose_seq, video_emb = out[0], out[1]
                if pose_seq is None or video_emb is None:
                    print(f"  [SKIPPED] {vid_path.name}{tag}")
                    continue
                vis, strip = out[2], out[3]
                s = 448 / max(vis.shape[:2])
                vis_small = cv2.resize(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), None, fx=s, fy=s)
                cv2.imwrite(str(dbg / norm_name / f"{vid_path.stem}{tag}_impact.jpg"), vis_small)
                cv2.imwrite(str(dbg / norm_name / f"{vid_path.stem}{tag}_window.jpg"), strip)

                X_pose.append(pose_seq)
                X_emb.append(video_emb)
                y.append(label_idx)
                paths.append(str(vid_path) + tag)
                shot_poses.append(pose_seq)
                print(f"  [OK] {vid_path.name}{tag}", flush=True)

        if shot_poses:
            golden_templates[norm_name] = np.mean(shot_poses, axis=0)

    np.savez(
        DATA_NPZ_PATH,
        X=np.array(X_pose, dtype=np.float32),
        X_emb=np.array(X_emb, dtype=np.float32),
        y=np.array(y, dtype=np.int64),
        classes=np.array(classes),
        display_names=display_names,
        templates=golden_templates,
        paths=np.array(paths),
    )
    print(f"\nDone! Saved {len(X_pose)} sequences to '{DATA_NPZ_PATH}'. Debug crops in '{dbg}/'.")


if __name__ == "__main__":
    import sys
    build_dataset(augment_mirror="--augment" in sys.argv)
