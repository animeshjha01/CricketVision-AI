import warnings
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from config import DATA_NPZ_PATH

warnings.filterwarnings("ignore", category=ConvergenceWarning)

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

MODEL_PKL_PATH = "shot_model.pkl"


def map_to_core_shot(folder_key: str) -> str:
    k = str(folder_key).lower()
    if "reverse" in k or "switch" in k:
        return "reverse_sweep"
    if "scoop" in k or "ramp" in k:
        return "scoop_shot"
    if "sweep" in k:
        return "sweep_shot"
    if "upper" in k:
        return "upper_cut"
    if "punch" in k:
        return "backfoot_punch"
    if "cut" in k:
        return "square_or_late_cut"
    if "pull" in k or "hook" in k:
        return "pull_shot"
    if "flick" in k or "leg" in k or "helicopter" in k:
        return "flick_shot"
    if "on_drive" in k or "straight" in k:
        return "straight_or_on_drive"
    if "cover" in k or "off_drive" in k or "square_drive" in k:
        return "cover_drive"
    if "loft" in k or "aggressive" in k or "inside_out" in k:
        return "lofted_shot"
    if "defen" in k or "block" in k or "fd" in k or "bfd" in k or "leave" in k:
        return "defensive_shot"
    return k


ANGLE_TRIPLETS = [
    (5, 7, 9), (6, 8, 10), (11, 5, 7), (12, 6, 8),
    (5, 11, 13), (6, 12, 14), (11, 13, 15), (12, 14, 16)
]


def _angle_between(pts, a, b, c):
    v1 = pts[:, :, a, :] - pts[:, :, b, :]
    v2 = pts[:, :, c, :] - pts[:, :, b, :]
    cos_t = np.sum(v1 * v2, axis=-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6
    )
    return np.arccos(np.clip(cos_t, -1.0, 1.0))


def _bone_length_normalize(X_pose: np.ndarray) -> np.ndarray:
    """
    Cancels out per-frame `np.std(kp)` distortion by rescaling each frame using
    the batter's rigid anatomical bone lengths (Torso + Upper Arms + Forearms + Thighs).
    Because bone length is physically constant, tucking the bat in for a Front-Foot Defense
    no longer stretches the skeleton or creates fake velocity!
    """
    N, T, _ = X_pose.shape
    pts = X_pose.reshape(N, T, 17, 2).copy()
    mid_hip = 0.5 * (pts[:, :, 11:12, :] + pts[:, :, 12:13, :])
    pts = pts - mid_hip

    # Sum of rigid anatomical segments on each frame: (P_i - P_j) / sigma_t
    torso_l = np.linalg.norm(pts[:, :, 5, :] - pts[:, :, 11, :], axis=-1)
    torso_r = np.linalg.norm(pts[:, :, 6, :] - pts[:, :, 12, :], axis=-1)
    uarm_l = np.linalg.norm(pts[:, :, 5, :] - pts[:, :, 7, :], axis=-1)
    uarm_r = np.linalg.norm(pts[:, :, 6, :] - pts[:, :, 8, :], axis=-1)
    farm_l = np.linalg.norm(pts[:, :, 7, :] - pts[:, :, 9, :], axis=-1)
    farm_r = np.linalg.norm(pts[:, :, 8, :] - pts[:, :, 10, :], axis=-1)
    thigh_l = np.linalg.norm(pts[:, :, 11, :] - pts[:, :, 13, :], axis=-1)
    thigh_r = np.linalg.norm(pts[:, :, 12, :] - pts[:, :, 14, :], axis=-1)

    bone_sum = (torso_l + torso_r + uarm_l + uarm_r + farm_l + farm_r + thigh_l + thigh_r) / 4.0  # (N, T)
    # Smooth bone_sum across 3 frames to prevent single-frame keypoint noise
    if T >= 3:
        b_smooth = bone_sum.copy()
        b_smooth[:, 1:-1] = 0.25 * bone_sum[:, :-2] + 0.50 * bone_sum[:, 1:-1] + 0.25 * bone_sum[:, 2:]
        bone_sum = b_smooth

    bone_scale = np.maximum(0.35, bone_sum)[:, :, None, None]
    return (pts / bone_scale).astype(np.float32)


def extract_key_physics_metrics(X_pose: np.ndarray, X_emb: np.ndarray) -> np.ndarray:
    """
    Extracts 10 core physical measurements (in true Bone-Length units, NEVER row-normalized!)
    used by both the classifier and the Biomechanical Guardrail Verifier:
      0: post_swing_energy      (Wrist speed + post-impact travel + flow energy: ~0 in Defense, high in strokes)
      1: horiz_bat_excursion    (Horizontal wrist span ptp(wrist_x): tiny in Defense/Straight, huge in Sweep/Cut/Pull)
      2:horiz_vs_vert_swing     (Horizontal vs vertical wrist speed ratio: high in Sweep/Cut/Pull, low in Defense/Drive)
      3: standing_leg_height    (Ankle_y - Hip_y in bone units: high when standing, collapses when kneeling in Sweep/Scoop)
      4: follow_dx              (Post-impact wrist X displacement: -X = Off-Side, +X = Leg-Side)
      5: follow_dy              (Post-impact wrist Y displacement: -Y = High Overhead finish, +Y = Low Grounded finish)
      6: net_leg_vs_off_flow    (Post-impact Leg-side minus Off-side optical flow)
      7: net_up_vs_down_flow    (Post-impact Upward minus Downward optical flow)
      8: impact_wrist_y         (Wrist height at impact relative to shoulders: negative/small = high chest/head, large = low pad)
      9: ankle_stride_spread    (Horizontal foot separation: wide in Front-Foot Defense/Drive/Sweep, narrow in Backfoot)
    """
    N, T, _ = X_pose.shape
    pts = _bone_length_normalize(X_pose)  # (N, 16, 17, 2)
    vel = np.diff(pts, axis=1, prepend=pts[:, :1])

    wrists = 0.5 * (pts[:, :, 9, :] + pts[:, :, 10, :])
    shldrs = 0.5 * (pts[:, :, 5, :] + pts[:, :, 6, :])
    ankles = 0.5 * (pts[:, :, 15, :] + pts[:, :, 16, :])

    wrist_vel = 0.5 * (vel[:, :, 9, :] + vel[:, :, 10, :])
    wrist_speed = np.linalg.norm(wrist_vel, axis=-1)

    # Dynamic per-sequence impact alignment: find peak wrist speed in middle window (frames 3..10)
    flow_raw = np.nan_to_num(X_emb[:, :144], nan=0.0)
    # Recover post-impact optical flow grid from stored flow_summary
    mean_grid = flow_raw[:, 0:36].reshape(N, 3, 3, 4)
    diff_grid = flow_raw[:, 108:144].reshape(N, 3, 3, 4)
    post_grid = np.maximum(0.0, mean_grid + 0.5 * diff_grid)

    post_off = np.sum(post_grid[:, :, :, 0], axis=(1, 2))
    post_leg = np.sum(post_grid[:, :, :, 1], axis=(1, 2))
    post_up = np.sum(post_grid[:, :, :, 2], axis=(1, 2))
    post_down = np.sum(post_grid[:, :, :, 3], axis=(1, 2))
    post_flow_total = post_off + post_leg + post_up + post_down

    metrics = np.zeros((N, 10), dtype=np.float32)
    for i in range(N):
        t_imp = int(np.argmax(wrist_speed[i, 3:10])) + 3
        imp_slice = slice(max(0, t_imp - 1), min(T, t_imp + 2))
        post_slice = slice(min(T - 2, t_imp + 1), T)

        imp_w = np.mean(wrists[i, imp_slice], axis=0)
        post_w = np.mean(wrists[i, post_slice], axis=0)
        post_travel = float(np.linalg.norm(post_w - imp_w))
        post_spd = float(np.mean(wrist_speed[i, post_slice]))

        horiz_excursion = float(np.ptp(wrists[i, max(2, t_imp - 2):, 0]))
        vx_a = float(np.mean(np.abs(wrist_vel[i, max(2, t_imp - 1):, 0])))
        vy_a = float(np.mean(np.abs(wrist_vel[i, max(2, t_imp - 1):, 1])))
        hv_ratio = vx_a / (vy_a + 0.06)

        # Standing leg height (in bone units, hips are at y=0, ankles are at +y)
        stand_leg_h = float(np.percentile(ankles[i, 4:, 1], 30))

        f_dx = float(post_w[0] - imp_w[0])
        f_dy = float(post_w[1] - imp_w[1])

        leg_vs_off = float((post_leg[i] - post_off[i]) / (post_leg[i] + post_off[i] + 0.15))
        up_vs_down = float((post_up[i] - post_down[i]) / (post_up[i] + post_down[i] + 0.15))

        imp_shldr_y = float(np.mean(shldrs[i, imp_slice, 1]))
        imp_wrist_rel_y = float(imp_w[1] - imp_shldr_y)  # smaller/negative = high chest/head, larger = low pad

        stride_spread = float(np.max(np.abs(pts[i, 4:, 15, 0] - pts[i, 4:, 16, 0])))
        swing_energy = post_travel * 1.8 + post_spd * 2.2 + horiz_excursion * 1.2 + np.log1p(post_flow_total[i]) * 0.4

        metrics[i] = [
            swing_energy,
            horiz_excursion,
            hv_ratio,
            stand_leg_h,
            f_dx,
            f_dy,
            leg_vs_off,
            up_vs_down,
            imp_wrist_rel_y,
            stride_spread,
        ]

    return metrics


def build_hybrid_features(X_pose: np.ndarray, X_emb: np.ndarray) -> np.ndarray:
    """
    Constructs a clean 4-Block Feature Vector (WITHOUT destructive per-sample L2 division!):
      - Block 1 (58 dims): Explicit Post-Impact Physics, Follow-Through & Posture Metrics
      - Block 2 (144 dims): Raw + Post-Impact Directional Optical Flow (Off, Leg, Up, Down)
      - Block 3 (188 dims): Bone-Normalized Stance, Impact, and Follow-Through Joint Coordinates & Angles
      - Block 4 (512 dims): 3D-CNN Motion Embedding (compressed via PCA inside StackedEnsemble)
    """
    N, T, _ = X_pose.shape
    pts = _bone_length_normalize(X_pose)  # (N, 16, 17, 2)
    body_pts = pts[:, :, [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], :]
    body_flat = body_pts.reshape(N, T, 26)
    vel = np.diff(pts, axis=1, prepend=pts[:, :1])
    body_vel = np.diff(body_pts, axis=1, prepend=body_pts[:, :1]).reshape(N, T, 26)

    angles = np.stack([_angle_between(pts, a, b, c) for a, b, c in ANGLE_TRIPLETS], axis=-1)

    wrists = 0.5 * (pts[:, :, 9, :] + pts[:, :, 10, :])
    elbows = 0.5 * (pts[:, :, 7, :] + pts[:, :, 8, :])
    shldrs = 0.5 * (pts[:, :, 5, :] + pts[:, :, 6, :])
    knees = 0.5 * (pts[:, :, 13, :] + pts[:, :, 14, :])
    ankles = 0.5 * (pts[:, :, 15, :] + pts[:, :, 16, :])

    key_metrics = extract_key_physics_metrics(X_pose, X_emb)  # (N, 10)

    # Post-impact trajectory features (Frames 7..15 vs Impact Frames 5..8)
    imp_w = np.mean(wrists[:, 5:8, :], axis=1)
    early_post_w = np.mean(wrists[:, 8:12, :], axis=1)
    late_post_w = np.mean(wrists[:, 12:, :], axis=1)
    imp_e = np.mean(elbows[:, 5:8, :], axis=1)
    late_post_e = np.mean(elbows[:, 12:, :], axis=1)

    post_vec_early = early_post_w - imp_w               # (N, 2)
    post_vec_late = late_post_w - imp_w                 # (N, 2)
    post_elbow_vec = late_post_e - imp_e                # (N, 2)
    post_w_rom = np.ptp(wrists[:, 6:, :], axis=1)       # (N, 2)
    full_w_rom = np.ptp(wrists, axis=1)                 # (N, 2)

    # Bat orientation at impact & follow-through (Vertical bat in Defense/Drive vs Horizontal bat in Sweep/Pull/Cut)
    wrist_to_elbow = wrists - elbows                    # (N, 16, 2)
    bat_horiz_vert_imp = np.mean(np.abs(wrist_to_elbow[:, 5:10, 0]), axis=1, keepdims=True) / (
        np.mean(np.abs(wrist_to_elbow[:, 5:10, 1]), axis=1, keepdims=True) + 0.08
    )
    wrist_sep_x = np.abs(pts[:, :, 9, 0] - pts[:, :, 10, 0])
    mean_wrist_sep_x = np.mean(wrist_sep_x[:, 5:], axis=1, keepdims=True)
    wrist_roll_delta = np.mean(pts[:, 10:, 9, 0] - pts[:, 10:, 10, 0], axis=1, keepdims=True) - (
        np.mean(pts[:, 4:8, 9, 0] - pts[:, 4:8, 10, 0], axis=1, keepdims=True)
    )

    # Posture & Knee Bend features (Separates Standing Defense/Punch from Kneeling Sweep/Scoop)
    ankle_y_mean = np.mean(ankles[:, 4:, 1], axis=1, keepdims=True)
    ankle_y_min = np.min(ankles[:, 4:, 1], axis=1, keepdims=True)
    knee_to_ankle_y = np.mean(ankles[:, 4:, 1] - knees[:, 4:, 1], axis=1, keepdims=True)  # Collapses to ~0 when kneeling!
    shldr_to_ankle_y = np.mean(ankles[:, 4:, 1] - shldrs[:, 4:, 1], axis=1, keepdims=True)
    min_knee_ang = np.min(angles[:, 4:, 6:8], axis=(1, 2)).reshape(N, 1)
    mean_knee_ang = np.mean(angles[:, 4:, 6:8], axis=(1, 2)).reshape(N, 1)

    # Wrist height relative to shoulders & hips across phases
    w_rel_shldr_y = wrists[:, :, 1] - shldrs[:, :, 1]
    w_h_stance = np.mean(w_rel_shldr_y[:, :5], axis=1, keepdims=True)
    w_h_impact = np.mean(w_rel_shldr_y[:, 5:10], axis=1, keepdims=True)
    w_h_finish = np.mean(w_rel_shldr_y[:, 10:], axis=1, keepdims=True)
    w_h_min_post = np.min(w_rel_shldr_y[:, 6:], axis=1, keepdims=True)

    # Optical Flow directional summaries (preserving true magnitude via log1p!)
    flow_raw = np.nan_to_num(X_emb[:, :144], nan=0.0)
    flow_log = np.sign(flow_raw) * np.log1p(np.abs(flow_raw))

    mean_grid = flow_raw[:, 0:36].reshape(N, 3, 3, 4)
    diff_grid = flow_raw[:, 108:144].reshape(N, 3, 3, 4)
    post_grid = np.maximum(0.0, mean_grid + 0.5 * diff_grid)

    p_off = np.log1p(np.sum(post_grid[:, :, :, 0], axis=(1, 2)).reshape(N, 1))
    p_leg = np.log1p(np.sum(post_grid[:, :, :, 1], axis=(1, 2)).reshape(N, 1))
    p_up = np.log1p(np.sum(post_grid[:, :, :, 2], axis=(1, 2)).reshape(N, 1))
    p_down = np.log1p(np.sum(post_grid[:, :, :, 3], axis=(1, 2)).reshape(N, 1))
    p_upper = np.log1p(np.sum(post_grid[:, 0:2, :, :], axis=(1, 2, 3)).reshape(N, 1))
    p_lower = np.log1p(np.sum(post_grid[:, 2:3, :, :], axis=(1, 2, 3)).reshape(N, 1))

    # Block 1: 42 Explicit Physics & Post-Impact Features
    physics_block = np.concatenate([
        key_metrics,                    # (10)
        post_vec_early,                 # (2)
        post_vec_late,                  # (2)
        post_elbow_vec,                 # (2)
        post_w_rom,                     # (2)
        full_w_rom,                     # (2)
        bat_horiz_vert_imp,             # (1)
        mean_wrist_sep_x,               # (1)
        wrist_roll_delta,               # (1)
        ankle_y_mean,                   # (1)
        ankle_y_min,                    # (1)
        knee_to_ankle_y,                # (1)
        shldr_to_ankle_y,               # (1)
        min_knee_ang,                   # (1)
        mean_knee_ang,                  # (1)
        w_h_stance,                     # (1)
        w_h_impact,                     # (1)
        w_h_finish,                     # (1)
        w_h_min_post,                   # (1)
        p_off, p_leg, p_up, p_down,     # (4)
        p_upper, p_lower,               # (2)
    ], axis=-1)                         # Total Block 1 = 39 dims

    # Block 3: 188 Pose & Angle Trajectory Features
    pose_block = np.concatenate([
        np.mean(body_flat[:, :5], axis=1),                 # Stance (26)
        np.mean(body_flat[:, 5:10], axis=1),               # Impact (26)
        np.mean(body_flat[:, 10:], axis=1),                # Follow-through (26)
        np.mean(body_flat[:, 10:], axis=1) - np.mean(body_flat[:, 4:8], axis=1),  # Post-hit delta (26)
        np.ptp(body_flat[:, 5:], axis=1),                  # Post-hit ROM (26)
        np.mean(body_vel[:, 6:], axis=1),                  # Post-hit velocity (26)
        np.mean(angles[:, 5:10], axis=1),                  # Impact angles (8)
        np.mean(angles[:, 10:], axis=1),                   # Finish angles (8)
        np.min(angles[:, 4:], axis=1),                     # Min angles (8)
        np.ptp(angles[:, 5:], axis=1),                     # Post-hit angle ROM (8)
    ], axis=-1)                                            # Total Block 3 = 188 dims

    cnn_raw = np.nan_to_num(X_emb[:, 144:656], nan=0.0)
    cnn_unit = cnn_raw / (np.linalg.norm(cnn_raw, axis=-1, keepdims=True) + 1e-5)

    # Total layout: [physics_block (39) | flow_log (144) | pose_block (188) | cnn_unit (512)] = 883 dims
    return np.nan_to_num(
        np.concatenate([physics_block, flow_log, pose_block, cnn_unit], axis=-1),
        nan=0.0
    ).astype(np.float32)


class StackedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Applies Post-Scaler Block Weighting AFTER RobustScaler so that:
      - Block 1 (39 Physics & Post-Impact Follow-Through dims) gets weight 2.8x (55% of variance!)
      - Block 2 (144 Directional Optical Flow dims, compressed to 32 PCA) gets weight 1.8x (24% of variance!)
      - Block 3 (188 Pose & Angle Trajectory dims, compressed to 40 PCA) gets weight 1.3x (19% of variance!)
      - Block 4 (512 3D-CNN Background dims, compressed to 16 PCA) gets weight 0.45x (only 2% of variance!)
    """

    def __init__(self, n_features: int, n_classes: int, n_splits: int = 4):
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_splits = n_splits
        self.scaler_phys_ = RobustScaler()
        self.scaler_flow_ = RobustScaler()
        self.scaler_pose_ = RobustScaler()
        self.pca_flow_ = None
        self.pca_pose_ = None
        self.pca_cnn_ = None
        self.base_learners_ = []
        self.X_ref_ = None
        self.y_ref_ = None
        self.classes_ = None
        self.marginal_prior_ = None

    def _transform_weighted_subspace(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X_phys = X[:, :39]
        X_flow = X[:, 39:183]
        X_pose = X[:, 183:371]
        X_cnn = X[:, 371:]

        if fit:
            P_s = np.clip(self.scaler_phys_.fit_transform(X_phys), -6.0, 6.0)
            F_s = np.clip(self.scaler_flow_.fit_transform(X_flow), -6.0, 6.0)
            Q_s = np.clip(self.scaler_pose_.fit_transform(X_pose), -6.0, 6.0)

            nf = min(32, len(X) - 2)
            nq = min(40, len(X) - 2)
            nc = min(16, len(X) - 2)
            self.pca_flow_ = PCA(n_components=nf, whiten=True, random_state=42).fit(F_s)
            self.pca_pose_ = PCA(n_components=nq, whiten=True, random_state=42).fit(Q_s)
            self.pca_cnn_ = PCA(n_components=nc, whiten=True, random_state=42).fit(X_cnn)
        else:
            P_s = np.clip(self.scaler_phys_.transform(X_phys), -6.0, 6.0)
            F_s = np.clip(self.scaler_flow_.transform(X_flow), -6.0, 6.0)
            Q_s = np.clip(self.scaler_pose_.transform(X_pose), -6.0, 6.0)

        F_p = self.pca_flow_.transform(F_s)
        Q_p = self.pca_pose_.transform(Q_s)
        C_p = self.pca_cnn_.transform(X_cnn)

        # POST-SCALER WEIGHTING, re-measured by 5-fold held-out block ablation after the
        # batter-tracking fix (see memory/shot-classifier-diagnosis.md): with correct crops the
        # pose/angle trajectory block is by far the strongest single signal, physics and the CNN
        # are comparable runners-up, and flow is weakest. The old weights (2.8/1.8/1.3/0.45) were
        # tuned against the broken crops, where pose was near chance, so they favoured the wrong
        # blocks. Re-tune this again (ablate.py pattern) if the dataset changes meaningfully.
        return np.concatenate([
            P_s * 1.6,   # 39 explicit post-impact & posture features
            F_p * 0.8,   # 32 whitened optical flow components
            Q_p * 2.6,   # 40 whitened joint/angle components
            C_p * 1.2,   # 16 whitened 3D-CNN components
        ], axis=1).astype(np.float32)

    def _build_learners(self):
        uniform_priors = np.ones(self.n_classes, dtype=np.float64) / self.n_classes
        learners = [
            ("svm_rbf", SVC(C=3.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=42)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=uniform_priors)),
            ("et", ExtraTreesClassifier(
                n_estimators=600, max_depth=16, min_samples_leaf=1, max_features="sqrt",
                class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
            ("rf", RandomForestClassifier(
                n_estimators=600, max_depth=16, min_samples_leaf=1, max_features="sqrt",
                class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
            ("lr", LogisticRegression(C=0.20, class_weight="balanced", max_iter=800, random_state=42)),
            ("mlp", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1.2, max_iter=500, random_state=42)),
        ]
        if _HAS_LGBM:
            learners.append(("lgbm", LGBMClassifier(
                n_estimators=140, max_depth=5, learning_rate=0.05,
                num_leaves=15, min_child_samples=3,
                subsample=0.85, colsample_bytree=0.75,
                reg_alpha=0.5, reg_lambda=1.2,
                class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1)))
        return learners

    def _per_class_exemplar_proba(self, X_sub: np.ndarray) -> tuple:
        """
        Uses Negative Euclidean Distance in the Post-Scaler Weighted Subspace (NOT cosine!)
        so true velocity and swing energy magnitudes are preserved!
        """
        n_dim = X_sub.shape[1]
        ex_probs = np.zeros((len(X_sub), self.n_classes), dtype=np.float64)
        min_dists = np.zeros((len(X_sub), 1), dtype=np.float64)

        for i in range(len(X_sub)):
            dists = np.linalg.norm(self.X_ref_ - X_sub[i:i + 1], axis=1) / np.sqrt(n_dim)
            min_dists[i, 0] = float(np.min(dists))
            class_scores = np.zeros(self.n_classes, dtype=np.float64)
            for ci, c in enumerate(self.classes_):
                c_dists = np.sort(dists[self.y_ref_ == c])
                if len(c_dists) >= 2:
                    d_best = 0.70 * c_dists[0] + 0.30 * c_dists[1]
                elif len(c_dists) == 1:
                    d_best = c_dists[0]
                else:
                    d_best = 10.0
                class_scores[ci] = -d_best

            logits = (class_scores - np.max(class_scores)) * 6.5
            exp_l = np.exp(logits)
            ex_probs[i] = exp_l / (np.sum(exp_l) + 1e-8)

        return ex_probs, min_dists

    def _raw_ensemble_proba(self, X_sub: np.ndarray) -> np.ndarray:
        n_classes = self.n_classes
        weights = [2.4, 2.0, 1.8, 1.4, 1.6, 1.3, 1.5][:len(self.base_learners_)]
        total_w = sum(weights)
        ens_p = np.zeros((X_sub.shape[0], n_classes), dtype=np.float64)

        for w, (name, est) in zip(weights, self.base_learners_):
            proba = np.clip(est.predict_proba(X_sub), 1e-6, 1.0)
            proba /= np.sum(proba, axis=1, keepdims=True)
            full_p = np.zeros((X_sub.shape[0], n_classes), dtype=np.float64)
            for ci, c in enumerate(est.classes_):
                col = np.where(self.classes_ == c)[0][0]
                full_p[:, col] = proba[:, ci]
            ens_p += (w / total_w) * full_p

        return ens_p

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)

        X_sub = self._transform_weighted_subspace(X, fit=True)
        self.X_ref_ = X_sub.copy()
        self.y_ref_ = y.copy()

        learner_defs = self._build_learners()
        self.base_learners_ = []
        for name, est in learner_defs:
            full_est = clone(est)
            full_est.fit(X_sub, y)
            self.base_learners_.append((name, full_est))

        min_cnt = int(np.min(np.bincount(y)))
        n_sp = max(2, min(4, min_cnt))
        skf = StratifiedKFold(n_splits=n_sp, shuffle=True, random_state=42)
        oof_p = np.zeros((len(X_sub), self.n_classes), dtype=np.float64)

        for tr_idx, val_idx in skf.split(X_sub, y):
            fold_p = np.zeros((len(val_idx), self.n_classes), dtype=np.float64)
            for name, est in self._build_learners()[:4]:
                fe = clone(est).fit(X_sub[tr_idx], y[tr_idx])
                pr = fe.predict_proba(X_sub[val_idx])
                for ci, c in enumerate(fe.classes_):
                    col = np.where(self.classes_ == c)[0][0]
                    fold_p[:, col] += pr[:, ci]
            fold_p /= (np.sum(fold_p, axis=1, keepdims=True) + 1e-8)
            oof_p[val_idx] = fold_p

        mean_marginal = np.mean(oof_p, axis=0)
        uniform_target = 1.0 / self.n_classes
        self.marginal_prior_ = np.clip(mean_marginal / uniform_target, 0.55, 2.0)
        return self

    def predict_proba(self, X):
        X_sub = self._transform_weighted_subspace(X, fit=False)
        ens_p = self._raw_ensemble_proba(X_sub)

        ens_debiased = ens_p / (self.marginal_prior_[None, :] ** 0.85)
        ens_debiased /= (np.sum(ens_debiased, axis=1, keepdims=True) + 1e-8)

        ex_p, min_d = self._per_class_exemplar_proba(X_sub)
        alpha = np.where(min_d < 0.25, 0.95, 0.35)
        final_p = (1.0 - alpha) * ens_debiased + alpha * ex_p
        return final_p / (np.sum(final_p, axis=1, keepdims=True) + 1e-8)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]




def train():
    data = np.load(DATA_NPZ_PATH, allow_pickle=True)
    X_pose, X_emb, raw_y = data["X"], data["X_emb"], data["y"]
    orig_classes = [str(c) for c in data["classes"]]
    sub_display_names = data["display_names"].item()

    u_cls, counts = np.unique(raw_y, return_counts=True)
    for c_idx, cnt in zip(u_cls, counts):
        if cnt < 2:
            X_pose = np.concatenate([X_pose, X_pose[raw_y == c_idx]], axis=0)
            X_emb = np.concatenate([X_emb, X_emb[raw_y == c_idx]], axis=0)
            raw_y = np.concatenate([raw_y, [c_idx]], axis=0)

    mapped_core = [map_to_core_shot(orig_classes[i]) for i in raw_y]
    core_classes = sorted(list(set(mapped_core)))
    y_core = np.array([core_classes.index(m) for m in mapped_core], dtype=np.int64)

    X_feat = build_hybrid_features(X_pose, X_emb)
    n_features = X_feat.shape[1]

    # Compute dataset-wide median & IQR of the 10 key physical measurements for guardrail verification
    all_phys = extract_key_physics_metrics(X_pose, X_emb)
    phys_median = np.median(all_phys, axis=0)
    phys_iqr = np.percentile(all_phys, 75, axis=0) - np.percentile(all_phys, 25, axis=0) + 1e-4

    X_tr, X_val, y_core_tr, y_core_val, y_sub_tr, y_sub_val = train_test_split(
        X_feat, y_core, raw_y, test_size=0.2, random_state=42, stratify=y_core
    )

    n_core_classes = len(core_classes)
    print(f"\nTraining Bone-Normalized Post-Scaler Weighted Ensemble ({len(X_tr)} train / {len(X_val)} val)...")
    core_model = StackedEnsemble(n_features=n_features, n_classes=n_core_classes, n_splits=4)
    core_model.fit(X_tr, y_core_tr)

    val_preds = core_model.predict(X_val)
    val_probs = core_model.predict_proba(X_val)

    acc = accuracy_score(y_core_val, val_preds) * 100
    top2 = np.mean([
        y_core_val[i] in np.argsort(val_probs[i])[-2:]
        for i in range(len(y_core_val))
    ]) * 100

    print(f"\n{'='*60}")
    print(f"  Unseen Validation Core Accuracy : {acc:.1f}%")
    print(f"  Unseen Top-2 Accuracy           : {top2:.1f}%")
    print(f"{'='*60}")

    target_names = [core_classes[i] for i in sorted(np.unique(y_core_val))]
    print("\nPer-Class Unseen Validation Report:")
    print(classification_report(
        y_core_val, val_preds,
        labels=sorted(np.unique(y_core_val)),
        target_names=target_names,
        zero_division=0
    ))

    # Honest held-out accuracy for the harder 23-exact-shot task too (same split as above).
    # This number, not the full-dataset fit below, is the one to trust as "real-world" accuracy.
    n_sub_classes = len(orig_classes)
    sub_model_val = StackedEnsemble(n_features=n_features, n_classes=n_sub_classes, n_splits=3)
    sub_model_val.fit(X_tr, y_sub_tr)
    sub_val_preds = sub_model_val.predict(X_val)
    sub_acc = accuracy_score(y_sub_val, sub_val_preds) * 100
    print(f"\n  Unseen Validation 23-Shot Accuracy : {sub_acc:.1f}%  (this is the number that matters)")
    print(f"{'='*60}\n")

    print("Fitting Production 23-Shot & Core Models on Full Dataset...")
    final_core = StackedEnsemble(n_features=n_features, n_classes=n_core_classes, n_splits=4)
    final_core.fit(X_feat, y_core)

    sub_model = StackedEnsemble(n_features=n_features, n_classes=n_sub_classes, n_splits=3)
    sub_model.fit(X_feat, raw_y)

    train_sub_acc = accuracy_score(raw_y, sub_model.predict(X_feat)) * 100
    print(f"Full-dataset fit accuracy (training data, NOT a real accuracy estimate): {train_sub_acc:.1f}%")

    feat_templates = {
        orig_classes[i]: np.mean(X_feat[raw_y == i], axis=0)
        for i in range(len(orig_classes))
    }

    joblib.dump({
        "core_model": final_core,
        "sub_model": sub_model,
        "core_classes": core_classes,
        "sub_classes": orig_classes,
        "sub_display_names": sub_display_names,
        "feat_templates": feat_templates,
        "phys_median": phys_median,
        "phys_iqr": phys_iqr,
    }, MODEL_PKL_PATH)
    print(f"\nSaved Production Model to '{MODEL_PKL_PATH}'!")


if __name__ == "__main__":
    train()