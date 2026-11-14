import sys
from pathlib import Path
import numpy as np
import joblib
from config import clean_label
from train_model import map_to_core_shot, StackedEnsemble, build_hybrid_features

sys.modules["__main__"].StackedEnsemble = StackedEnsemble

bundle = joblib.load("shot_model.pkl")
core_model = bundle["core_model"]
sub_model = bundle["sub_model"]
core_classes = list(bundle["core_classes"])
sub_classes = list(bundle["sub_classes"])

data = np.load("cricket_data.npz", allow_pickle=True)
X = data["X"]
X_emb = data["X_emb"]
y = data["y"]
classes = list(data["classes"])

print(f"Dataset: {len(y)} samples, {len(classes)} classes")

X_feat = build_hybrid_features(X, X_emb)
print(f"Features: {X_feat.shape}")

# Core evaluation
y_core = np.array([core_classes.index(map_to_core_shot(classes[yi])) for yi in y])
core_probs = core_model.predict_proba(X_feat)
core_preds = np.argmax(core_probs, axis=1)
core_acc = np.mean(core_preds == y_core) * 100.0

# Sub evaluation
sub_probs = sub_model.predict_proba(X_feat)
sub_preds = np.argmax(sub_probs, axis=1)
sub_acc = np.mean(sub_preds == y) * 100.0

# Hierarchical combination exactly as in app.py
combined_scores = np.zeros((len(y), len(sub_classes)), dtype=np.float32)
for si, sk in enumerate(sub_classes):
    ci = core_classes.index(map_to_core_shot(sk))
    combined_scores[:, si] = (core_probs[:, ci] ** 0.60) * (sub_probs[:, si] ** 0.40)
combined_preds = np.argmax(combined_scores, axis=1)
combined_acc = np.mean(combined_preds == y) * 100.0

print(f"\n==========================================")
print(f"Core Model Train Accuracy:     {core_acc:.2f}%")
print(f"Sub Model Train Accuracy:      {sub_acc:.2f}%")
print(f"Hierarchical Combined Accuracy: {combined_acc:.2f}%")
print(f"==========================================\n")

print("--- Breakdown by Folder / Class ---")
for ci, cname in enumerate(classes):
    mask = (y == ci)
    count = np.sum(mask)
    if count == 0:
        continue
    c_core_acc = np.mean(core_preds[mask] == y_core[mask]) * 100.0
    c_sub_acc = np.mean(sub_preds[mask] == ci) * 100.0
    c_comb_acc = np.mean(combined_preds[mask] == ci) * 100.0
    print(f"{cname:28s} (N={count:2d}): Core={c_core_acc:5.1f}% | Sub={c_sub_acc:5.1f}% | Comb={c_comb_acc:5.1f}%")
