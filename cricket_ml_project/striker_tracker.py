"""
striker_tracker.py — finds and follows the striking batter through a whole clip.

The old approach picked ONE fixed box from the first few frames. Broadcast cameras
zoom in during the shot, so that box ended up on the wicketkeeper or empty grass.

This module instead:
  1. Reads the whole clip (up to MAX_SECONDS) at ~DET_FPS and strips black letterbox bars.
  2. Runs YOLO-pose with ByteTrack on every frame, so every person gets a track id.
  3. Scores each track by *behaviour*: the batter is near the middle of the frame,
     stays roughly in place (unlike the bowler / running non-striker), and has the
     fastest wrist movement relative to their own body (unlike umpire / keeper).
  4. Finds the shot moment as the batter's peak wrist speed, and returns a per-frame
     box that follows the batter (re-linking across zooms / lost detections).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

DET_FPS = 15.0
MAX_SECONDS = 20.0
MAX_SIDE = 960          # frames are downscaled so the long side is at most this
DET_IMGSZ = 960


@dataclass
class TrackObs:
    t: int
    box: np.ndarray        # x1, y1, x2, y2 (px)
    kpts: np.ndarray       # (17, 2) px
    kconf: np.ndarray      # (17,)


@dataclass
class Track:
    tid: int
    obs: list = field(default_factory=list)


def read_frames(video_path: str, mirror: bool = False):
    """Reads frames at ~DET_FPS, downscaled, letterbox-trimmed, optionally mirrored (left-handers)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    step = max(1, int(round(fps / DET_FPS)))
    max_idx = int(fps * MAX_SECONDS)
    frames, idx = [], 0
    while True:
        ok = cap.grab()
        if not ok or idx > max_idx:
            break
        if idx % step == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            h, w = f.shape[:2]
            s = MAX_SIDE / max(h, w)
            if s < 1.0:
                f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(f)
        idx += 1
    cap.release()
    if not frames:
        return [], fps / step

    # Trim black letterbox / pillarbox bars (screen recordings with the video in the middle)
    sample = np.stack(frames[:: max(1, len(frames) // 12)]).max(axis=0).max(axis=2)
    rows = np.where(sample.max(axis=1) > 25)[0]
    cols = np.where(sample.max(axis=0) > 25)[0]
    if len(rows) > 20 and len(cols) > 20:
        y1, y2, x1, x2 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        if (y2 - y1) * (x2 - x1) < 0.92 * sample.size:
            frames = [f[y1:y2, x1:x2] for f in frames]

    if mirror:
        frames = [cv2.flip(f, 1) for f in frames]
    return frames, fps / step


def scene_ids(frames: list) -> np.ndarray:
    """Assigns a shot-segment id to each frame; increments on hard cuts."""
    def sig(img):
        hsv = cv2.cvtColor(cv2.resize(img, (48, 48)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [12, 12], [0, 180, 0, 256]).flatten()
        return h / (np.linalg.norm(h) + 1e-6)
    sigs = [sig(f) for f in frames]
    ids = np.zeros(len(frames), dtype=int)
    for i in range(1, len(frames)):
        ids[i] = ids[i - 1] + (1 if float(np.dot(sigs[i - 1], sigs[i])) < 0.55 else 0)
    return ids


def run_tracking(pose_model, frames: list, device) -> dict:
    tracks = {}
    for t, f in enumerate(frames):
        r = pose_model.track(f, persist=(t > 0), verbose=False, device=device,
                             imgsz=DET_IMGSZ, conf=0.15, tracker="bytetrack.yaml")[0]
        if r.boxes is None or r.boxes.id is None or len(r.boxes) == 0:
            continue
        ids = r.boxes.id.cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.cpu().numpy()
        kp = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kp.shape[:2])
        for i, tid in enumerate(ids):
            tracks.setdefault(tid, Track(tid)).obs.append(TrackObs(t, boxes[i], kp[i], kc[i]))
    return tracks


def _wrist_rel_speed(tr: Track) -> np.ndarray:
    """Per-observation wrist speed relative to the hip, in body heights per frame (0 for first obs)."""
    sp = np.zeros(len(tr.obs))
    for j in range(1, len(tr.obs)):
        a, b = tr.obs[j - 1], tr.obs[j]
        dt = b.t - a.t
        if dt <= 0 or dt > 3:
            continue
        h = 0.5 * ((a.box[3] - a.box[1]) + (b.box[3] - b.box[1])) + 1e-6
        def rel(o):
            hip = o.kpts[[11, 12]].mean(axis=0)
            wr_ok = o.kconf[[9, 10]] > 0.3
            if not wr_ok.any():
                return None
            return o.kpts[[9, 10]][wr_ok].mean(axis=0) - hip
        ra, rb = rel(a), rel(b)
        if ra is None or rb is None:
            continue
        sp[j] = float(np.linalg.norm(rb - ra)) / h / dt
    return sp


def _scene_context(tracks: dict, tid: int, W: int, H: int) -> dict:
    """
    Per-track context cues computed against everyone else visible in the same frame.

    In the standard behind-the-bowler TV view the people in the central column, from far
    to near, are: keeper (+slips) -> batter -> bowler / umpire / non-striker. So:
      * above1: someone central is 'up the pitch' from me (true for batter AND bowler/umpire)
      * above2: two or more people are up the pitch from me (bowler / umpire / non-striker only)
      * keeper: nobody is above me but a much taller central person stands in front of me
      * edge:   my box is cut by the top/bottom of the frame (partial keeper, net bowler, overlays)
    """
    by_t = {}
    for otid, tr in tracks.items():
        if otid == tid or len(tr.obs) < 3:
            continue
        for o in tr.obs:
            by_t.setdefault(o.t, []).append(o.box)
    above1 = above2 = keeper = edge = 0
    own = tracks[tid].obs
    for o in own:
        x1, y1, x2, y2 = o.box
        h = y2 - y1
        cx = 0.5 * (x1 + x2)
        n_above, taller_in_front = 0, False
        for b in by_t.get(o.t, []):
            bh = b[3] - b[1]
            bcx = 0.5 * (b[0] + b[2])
            central = abs(bcx / W - 0.5) < 0.28
            if central and 0.25 < bh / (h + 1e-6) < 1.1 and b[3] < y2 - 0.25 * h:
                n_above += 1
            if central and bh > 1.25 * h and b[3] > y2 + 0.1 * h and abs(bcx - cx) < 2.0 * bh:
                taller_in_front = True
        above1 += n_above >= 1
        above2 += n_above >= 2
        keeper += (n_above == 0 and taller_in_front)
        edge += (y1 < 0.012 * H) or (y2 > 0.988 * H)
    n = max(1, len(own))
    return dict(above1=above1 / n, above2=above2 / n, keeper=keeper / n, edge=edge / n)


def score_tracks(tracks: dict, W: int, H: int, seg: np.ndarray = None) -> list:
    """Returns [(score, tid, info)] sorted best-first."""
    scored = []
    uniq = np.unique(seg) if seg is not None else None
    for tid, tr in tracks.items():
        if len(tr.obs) < 5:
            continue
        boxes = np.array([o.box for o in tr.obs])
        hgt = boxes[:, 3] - boxes[:, 1]
        med_h = float(np.median(hgt))
        if med_h < 0.06 * H:
            continue  # tiny distant fielders / crowd
        cx = 0.5 * (boxes[:, 0] + boxes[:, 2]) / W
        centrality = float(np.median(np.abs(cx - 0.5)))
        # horizontal travel of the box centre relative to body height (bowler / runners move a lot)
        centres = 0.5 * (boxes[:, :2] + boxes[:, 2:])
        travel = float(np.ptp(centres[:, 0]) / (med_h + 1e-6))
        sp = _wrist_rel_speed(tr)
        sp_s = np.convolve(sp, np.ones(3) / 3, mode="same")
        peak = float(np.percentile(sp_s, 95)) if len(sp_s) else 0.0
        mean_kconf = float(np.mean([o.kconf.mean() for o in tr.obs]))
        persistence = min(1.0, len(tr.obs) / 15.0)
        ctx = _scene_context(tracks, tid, W, H)
        seg_idx = int(np.searchsorted(uniq, seg[tr.obs[0].t])) if uniq is not None else 0

        score = (
            2.0 * min(peak, 0.22)                 # a real swing, but bowling-arm speed must not dominate
            - 1.6 * centrality
            - 0.40 * max(0.0, travel - 1.2)
            + 0.25 * persistence
            + 1.00 * mean_kconf                   # the batter is usually the clearest, best-lit skeleton
            + 0.15 * min(1.0, med_h / (0.45 * H))
            - 0.25 * ctx["above1"]
            - 0.70 * ctx["above2"]
            - 0.70 * ctx["keeper"]
            - 0.30 * ctx["edge"]
            - 0.06 * min(seg_idx, 4)
        )
        info = dict(peak=peak, centrality=centrality, travel=travel, up=ctx["above2"], kc=mean_kconf,
                    n=len(tr.obs), med_h=med_h, sp=sp_s, **ctx)
        scored.append((score, tid, info))
    scored.sort(key=lambda s: -s[0])
    return scored


def follow_batter(tracks: dict, best_tid: int, n_frames: int, seg: np.ndarray) -> np.ndarray:
    """
    Returns a (n_frames, 4) array of batter boxes; NaN where unknown.
    Starts from the best track and greedily re-links to other tracks when it is lost
    (same scene segment, nearest box relative to the last known body height).
    """
    out = np.full((n_frames, 4), np.nan, dtype=np.float32)
    by_t = {}
    for tid, tr in tracks.items():
        for o in tr.obs:
            by_t.setdefault(o.t, []).append((tid, o.box))
    for o in tracks[best_tid].obs:
        out[o.t] = o.box

    def relink(direction):
        rng = range(n_frames) if direction > 0 else range(n_frames - 1, -1, -1)
        last = None
        for t in rng:
            if not np.isnan(out[t, 0]):
                last = (t, out[t].copy())
                continue
            if last is None or seg[t] != seg[last[0]] or abs(t - last[0]) > 8:
                continue
            lb = last[1]
            lh = lb[3] - lb[1]
            lc = 0.5 * (lb[:2] + lb[2:])
            best, best_d = None, 0.6
            for _, b in by_t.get(t, []):
                c = 0.5 * (b[:2] + b[2:])
                h = b[3] - b[1]
                d = np.linalg.norm(c - lc) / (lh + 1e-6) + 0.5 * abs(np.log((h + 1e-6) / (lh + 1e-6)))
                if d < best_d:
                    best, best_d = b, d
            if best is not None:
                out[t] = best
                last = (t, best.copy())

    relink(+1)
    relink(-1)
    return out


def locate_striker(pose_model, frames: list, device):
    """Main entry: returns dict with per-frame batter boxes, impact frame index and debug info."""
    H, W = frames[0].shape[:2]
    seg = scene_ids(frames)
    tracks = run_tracking(pose_model, frames, device)
    scored = score_tracks(tracks, W, H, seg)
    if not scored:
        return None
    best_score, best_tid, info = scored[0]
    boxes = follow_batter(tracks, best_tid, len(frames), seg)

    # impact = peak (smoothed) relative wrist speed of the chosen track
    tr = tracks[best_tid]
    t_obs = np.array([o.t for o in tr.obs])
    peak_t = int(t_obs[int(np.argmax(info["sp"]))])
    return dict(boxes=boxes, impact_t=peak_t, tid=best_tid, seg=seg,
                score=best_score, info=info, ranking=scored[:4])


def crop_box_for(box: np.ndarray, W: int, H: int, pad: float = 0.35) -> tuple:
    """Square crop around a batter box, padded so the bat swing stays inside."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1
    side = max(h, w) * (1.0 + 2 * pad)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    side = max(side, 32)
    return (int(np.clip(cx - side / 2, 0, W - 1)), int(np.clip(cy - side / 2, 0, H - 1)),
            int(np.clip(cx + side / 2, 1, W)), int(np.clip(cy + side / 2, 1, H)))


def smooth_boxes(boxes: np.ndarray) -> np.ndarray:
    """Fill NaNs by interpolation/hold and smooth the box path to remove jitter."""
    b = boxes.copy()
    n = len(b)
    valid = ~np.isnan(b[:, 0])
    if not valid.any():
        return b
    idx = np.arange(n)
    for k in range(4):
        b[:, k] = np.interp(idx, idx[valid], b[valid, k])
    if n >= 3:
        k = np.array([0.25, 0.5, 0.25])
        for c in range(4):
            padded = np.pad(b[:, c], 1, mode="edge")
            b[:, c] = np.convolve(padded, k, mode="valid")
    return b
