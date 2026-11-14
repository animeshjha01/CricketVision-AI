import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from train_model import map_to_core_shot, build_hybrid_features

d = np.load("cricket_data.npz", allow_pickle=True)
X_pose = d["X"]
X_emb = d["X_emb"]
y = d["y"]
classes = [str(c) for c in d["classes"]]
core_classes = sorted(list(set(map_to_core_shot(c) for c in classes)))
y_core = np.array([core_classes.index(map_to_core_shot(classes[yi])) for yi in y])

X_feat = build_hybrid_features(X_pose, X_emb)

# Biomechanical constraints extraction:
# 1. Horizontal flow bias: Off-side flow (columns 0..72) vs Leg-side flow (columns 72..144)
flow_feats = X_emb[:, :144]
# In 3x3 grid * 4 phases, off_side is col 0, 4, 8, 12... leg_side is col 1, 5, 9, 13...
off_energy = np.mean(flow_feats[:, 0::4], axis=1)  # High for offside shots (cover drive, cut, punch)
leg_energy = np.mean(flow_feats[:, 1::4], axis=1)  # High for legside shots (flick, on-drive, pull)
off_vs_leg = (off_energy - leg_energy) / (off_energy + leg_energy + 1e-5)  # +1 = strong offside, -1 = strong legside

# 2. Vertical swing ratio
up_energy = np.mean(flow_feats[:, 2::4], axis=1)
down_energy = np.mean(flow_feats[:, 3::4], axis=1)
vert_vs_horiz = (up_energy + down_energy) / (off_energy + leg_energy + 1e-5)

print("Biomechanical indicators computed:")
print("Off vs Leg range:", np.min(off_vs_leg), np.max(off_vs_leg))
print("Vert vs Horiz range:", np.min(vert_vs_horiz), np.max(vert_vs_horiz))

# Evaluate 5-fold CV with Metric Learning (Prototype Distance + Ridge + ExtraTrees + Biomechanical Physics)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_smart_model():
    fold_core_accs = []
    fold_top2_accs = []
    fold_sub_accs = []

    for tr_idx, val_idx in skf.split(X_feat, y):
        X_tr, y_tr = X_feat[tr_idx], y[tr_idx]
        y_c_tr = y_core[tr_idx]
        X_val, y_val = X_feat[val_idx], y[val_idx]
        y_c_val = y_core[val_idx]

        # 1. Compute per-class centroids on train fold
        templates = np.array([
            np.mean(X_tr[y_tr == ci], axis=0) if np.sum(y_tr == ci) > 0 else np.zeros(X_tr.shape[1])
            for ci in range(len(classes))
        ])

        # Cosine similarity to templates as rich meta-features
        tr_sims = cosine_similarity(X_tr, templates)
        val_sims = cosine_similarity(X_val, templates)

        # 2. Classifier on augmented feature representation (Features + Template Similarities)
        X_tr_augmented = np.hstack([X_tr, tr_sims * 5.0])
        X_val_augmented = np.hstack([X_val, val_sims * 5.0])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_augmented)
        X_val_s = scaler.transform(X_val_augmented)

        # Core model
        clf_core = LogisticRegression(C=0.08, class_weight="balanced", max_iter=1000, random_state=42)
        clf_core.fit(X_tr_s, y_c_tr)
        core_probs = clf_core.predict_proba(X_val_s)

        # Sub model
        clf_sub = LogisticRegression(C=0.05, class_weight="balanced", max_iter=1000, random_state=42)
        clf_sub.fit(X_tr_s, y_tr)
        sub_probs = clf_sub.predict_proba(X_val_s)

        # Physics rule adjustment for each validation sample:
        # If off_vs_leg > 0.15 (strong offside): flick_shot probability gets penalized
        # If vert_vs_horiz > 1.2 (vertical bat): scoop & sweep get penalized
        for i, vi in enumerate(val_idx):
            ovl = off_vs_leg[vi]
            vvh = vert_vs_horiz[vi]
            if ovl > 0.12:  # Definitely offside! (Cover drive, Late cut, Backfoot punch)
                flick_idx = core_classes.index("flick_shot")
                core_probs[i, flick_idx] *= 0.15
            elif ovl < -0.15:  # Definitely legside! (Flick shot, on-drive, pull)
                cover_idx = core_classes.index("cover_drive")
                punch_idx = core_classes.index("backfoot_punch")
                core_probs[i, cover_idx] *= 0.20
                core_probs[i, punch_idx] *= 0.20

            if vvh > 1.25:  # Strongly vertical bat (Drive, Defense, Punch)
                scoop_idx = core_classes.index("scoop_or_sweep")
                core_probs[i, scoop_idx] *= 0.10

            core_probs[i] /= np.sum(core_probs[i]) + 1e-8

        # Combine core & sub
        pred_core = np.argmax(core_probs, axis=1)
        fold_core_accs.append(accuracy_score(y_c_val, pred_core))

        top2 = np.mean([y_c_val[i] in np.argsort(core_probs[i])[-2:] for i in range(len(y_c_val))])
        fold_top2_accs.append(top2)

    print(f"Physics + Template Regularized Core Accuracy: {np.mean(fold_core_accs)*100:.2f}%")
    print(f"Physics + Template Regularized Top-2 Accuracy: {np.mean(fold_top2_accs)*100:.2f}%")

evaluate_smart_model()
