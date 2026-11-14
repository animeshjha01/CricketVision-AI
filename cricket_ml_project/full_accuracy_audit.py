import numpy as np
import joblib
from collections import defaultdict
from train_model import map_to_core_shot, StackedEnsemble, build_hybrid_features

# Ensure pickle compatibility
import sys
sys.modules["__main__"].StackedEnsemble = StackedEnsemble

bundle = joblib.load("shot_model.pkl")
core_model = bundle["core_model"]
sub_model = bundle["sub_model"]
core_classes = list(bundle["core_classes"])
sub_classes = list(bundle["sub_classes"])
sub_display_names = bundle["sub_display_names"]

data = np.load("cricket_data.npz", allow_pickle=True)
X = data["X"]
X_emb = data["X_emb"]
y = data["y"]
classes = list(data["classes"])

X_feat = build_hybrid_features(X, X_emb)

core_probs = core_model.predict_proba(X_feat)
sub_probs = sub_model.predict_proba(X_feat)

combined_scores = np.zeros((len(y), len(sub_classes)), dtype=np.float32)
for si, sk in enumerate(sub_classes):
    ci = core_classes.index(map_to_core_shot(sk))
    combined_scores[:, si] = (core_probs[:, ci] ** 0.60) * (sub_probs[:, si] ** 0.40)

pred_sub_indices = np.argmax(combined_scores, axis=1)
pred_sub_keys = [sub_classes[i] for i in pred_sub_indices]
pred_core_keys = [map_to_core_shot(sk) for sk in pred_sub_keys]

true_sub_keys = [classes[yi] for yi in y]
true_core_keys = [map_to_core_shot(sk) for sk in true_sub_keys]

# Top-1 accuracy
exact_sub_acc = np.mean(np.array(pred_sub_keys) == np.array(true_sub_keys)) * 100.0
exact_core_acc = np.mean(np.array(pred_core_keys) == np.array(true_core_keys)) * 100.0

# Top-2 accuracy
top2_sub_correct = 0
for i in range(len(y)):
    top2_idx = np.argsort(combined_scores[i])[-2:]
    if y[i] in top2_idx:
        top2_sub_correct += 1
top2_sub_acc = (top2_sub_correct / len(y)) * 100.0

# Top-3 accuracy
top3_sub_correct = 0
for i in range(len(y)):
    top3_idx = np.argsort(combined_scores[i])[-3:]
    if y[i] in top3_idx:
        top3_sub_correct += 1
top3_sub_acc = (top3_sub_correct / len(y)) * 100.0

print("=" * 60)
print(f"CRICKETVISION AI — OFFICIAL DATASET ACCURACY AUDIT")
print(f"Total Videos Analyzed: {len(y)}")
print("=" * 60)
print(f"  Exact 23-Shot Sub-Category Accuracy : {exact_sub_acc:.2f}%")
print(f"  Core Biomechanical Family Accuracy  : {exact_core_acc:.2f}%")
print(f"  Top-2 DRS Accuracy                  : {top2_sub_acc:.2f}%")
print(f"  Top-3 DRS Repertoire Accuracy       : {top3_sub_acc:.2f}%")
print("=" * 60)

print("\n--- Key Demonstration Categories Accuracy ---")
demo_shots = ["cover_drive", "backfoot_punch", "front_foot_defense", "flick_shot", "pull_shot", "square_cut", "upper_cut", "sweep", "scoop"]
for ds in demo_shots:
    if ds in classes:
        ci = classes.index(ds)
        mask = (y == ci)
        acc = np.mean(np.array(pred_sub_keys)[mask] == ds) * 100.0
        print(f"  {sub_display_names.get(ds, ds):22s} ({np.sum(mask):2d} clips) : {acc:.1f}%")
