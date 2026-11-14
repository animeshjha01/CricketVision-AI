import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from train_model import map_to_core_shot, build_hybrid_features

d = np.load("cricket_data.npz", allow_pickle=True)
X_pose = d["X"]
X_emb = d["X_emb"]
y = d["y"]
classes = [str(c) for c in d["classes"]]

X_feat = build_hybrid_features(X_pose, X_emb)

# Compute mean template per class from all samples
templates = np.array([np.mean(X_feat[y == ci], axis=0) for ci in range(len(classes))])

# Cosine similarity matrix
sims = cosine_similarity(X_feat, templates)  # (299, 23)
template_preds = np.argmax(sims, axis=1)

print(f"Nearest Template Exact 23-Shot Accuracy: {np.mean(template_preds == y)*100:.2f}%")

# Core accuracy
core_classes = sorted(list(set(map_to_core_shot(c) for c in classes)))
y_core = np.array([core_classes.index(map_to_core_shot(classes[yi])) for yi in y])
pred_core = np.array([core_classes.index(map_to_core_shot(classes[pi])) for pi in template_preds])

print(f"Nearest Template Core Shot Accuracy:     {np.mean(pred_core == y_core)*100:.2f}%")
