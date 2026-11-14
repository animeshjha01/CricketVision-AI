import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from config import clean_label
from train_model import map_to_core_shot

# Load current YOLO11 data
d = np.load("cricket_data.npz", allow_pickle=True)
X_pose = d["X"]
X_emb = d["X_emb"]
raw_y = d["y"]
classes = [str(c) for c in d["classes"]]
mapped_core = [map_to_core_shot(classes[i]) for i in raw_y]
core_classes = sorted(list(set(mapped_core)))
y_core = np.array([core_classes.index(m) for m in mapped_core], dtype=np.int64)

# Original feature builder (clean, no bloated noise features)
ANGLE_TRIPLETS = [
    (5, 7, 9), (6, 8, 10), (11, 5, 7), (12, 6, 8),
    (5, 11, 13), (6, 12, 14), (11, 13, 15), (12, 14, 16)
]

def build_original_features(X_pose, X_emb):
    N, T, _ = X_pose.shape
    pts = X_pose.reshape(N, T, 17, 2)
    vel = np.diff(pts, axis=1, prepend=pts[:, :1])

    angles = []
    for a, b, c in ANGLE_TRIPLETS:
        v1 = pts[:, :, a, :] - pts[:, :, b, :]
        v2 = pts[:, :, c, :] - pts[:, :, b, :]
        cos_t = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-6)
        angles.append(np.arccos(np.clip(cos_t, -1.0, 1.0)))
    angles = np.stack(angles, axis=-1)

    wrists = 0.5 * (pts[:, :, 9, :] + pts[:, :, 10, :])
    shldrs = 0.5 * (pts[:, :, 5, :] + pts[:, :, 6, :])
    wrist_rel_shldr = wrists - shldrs

    wrist_vel = 0.5 * (vel[:, :, 9, :] + vel[:, :, 10, :])
    vx_mean = np.mean(wrist_vel[:, :, 0], axis=1, keepdims=True)
    vy_mean = np.mean(wrist_vel[:, :, 1], axis=1, keepdims=True)
    vx_abs = np.mean(np.abs(wrist_vel[:, :, 0]), axis=1, keepdims=True)
    vy_abs = np.mean(np.abs(wrist_vel[:, :, 1]), axis=1, keepdims=True)
    horiz_vert_ratio = vx_abs / (vy_abs + 0.05)

    swing_delta = np.mean(wrists[:, 10:, :], axis=1) - np.mean(wrists[:, :6, :], axis=1)
    min_wrist_y = np.min(wrist_rel_shldr[:, :, 1], axis=1, keepdims=True)

    pose_feats = np.concatenate([
        np.mean(pts.reshape(N, T, 34), axis=1),
        np.std(pts.reshape(N, T, 34), axis=1),
        np.mean(pts[:, :6].reshape(N, 6, 34), axis=1),
        np.mean(pts[:, 6:11].reshape(N, 5, 34), axis=1),
        np.mean(pts[:, 11:].reshape(N, 5, 34), axis=1),
        np.mean(vel.reshape(N, T, 34), axis=1),
        np.max(vel.reshape(N, T, 34), axis=1),
        np.mean(angles, axis=1),
        np.max(angles, axis=1),
        np.min(angles, axis=1),
        vx_mean, vy_mean, vx_abs, vy_abs, horiz_vert_ratio, swing_delta, min_wrist_y,
    ], axis=-1)

    flow_part = X_emb[:, :144] * 2.5
    cnn_part = X_emb[:, 144:656] * 0.75
    pose_part = pose_feats * 2.2

    return np.nan_to_num(np.concatenate([flow_part, pose_part, cnn_part], axis=-1), nan=0.0).astype(np.float32)

X_orig = build_original_features(X_pose, X_emb)
print(f"Original features shape: {X_orig.shape}")

# Also let's test what features actually carry signal:
# Flow only (144), Pose only (283), CNN only (512)
flow_only = X_emb[:, :144]
pose_only = X_orig[:, 144:144+283]
cnn_only = X_emb[:, 144:656]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def eval_clf(clf_factory, X_data, y_data):
    scores = []
    top2_scores = []
    for train_idx, test_idx in skf.split(X_data, y_data):
        clf = clf_factory()
        clf.fit(X_data[train_idx], y_data[train_idx])
        preds = clf.predict(X_data[test_idx])
        probs = clf.predict_proba(X_data[test_idx])
        scores.append(accuracy_score(y_data[test_idx], preds))
        top2 = np.mean([y_data[test_idx][i] in np.argsort(probs[i])[-2:] for i in range(len(test_idx))])
        top2_scores.append(top2)
    return np.mean(scores)*100, np.std(scores)*100, np.mean(top2_scores)*100

print("\n--- Evaluating Classifiers on Core Shots (5-Fold CV) ---")
clfs = [
    ("LogisticRegression (C=0.1, L2)", lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=42))),
    ("Linear SVM (C=0.1)", lambda: make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.1, probability=True, random_state=42))),
    ("RBF SVM (C=1.0)", lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True, random_state=42))),
    ("ExtraTrees (200 trees)", lambda: make_pipeline(StandardScaler(), ExtraTreesClassifier(n_estimators=200, random_state=42))),
    ("RandomForest (200 trees)", lambda: make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42))),
    ("PCA(60) + LogisticRegression", lambda: make_pipeline(StandardScaler(), PCA(n_components=60), LogisticRegression(C=0.5, max_iter=1000, random_state=42))),
    ("PCA(60) + MLP(128,64)", lambda: make_pipeline(StandardScaler(), PCA(n_components=60), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1.0, max_iter=500, random_state=42))),
    ("Original Soft Voting", lambda: make_pipeline(
        StandardScaler(),
        VotingClassifier(
            estimators=[
                ("et", ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
                ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
                ("lr", make_pipeline(PCA(n_components=60, random_state=42), LogisticRegression(C=0.2, max_iter=1000, random_state=42))),
                ("mlp", make_pipeline(PCA(n_components=60, random_state=42), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1.2, max_iter=500, random_state=42))),
            ],
            voting="soft",
            weights=[2, 1, 2, 2],
            n_jobs=-1,
        )
    ))
]

for name, factory in clfs:
    mean_acc, std_acc, top2 = eval_clf(factory, X_orig, y_core)
    print(f"{name:32s}: Acc = {mean_acc:5.1f}% +/- {std_acc:4.1f}% | Top-2 = {top2:5.1f}%")
