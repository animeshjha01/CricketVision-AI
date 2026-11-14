import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from train_model import map_to_core_shot

d_v8 = np.load("cricket_data_v8_backup.npz", allow_pickle=True)
X_v8 = d_v8["X"]
X_emb_v8 = d_v8["X_emb"]
y_v8 = d_v8["y"]
classes_v8 = [str(c) for c in d_v8["classes"]]
mapped_core_v8 = [map_to_core_shot(classes_v8[i]) for i in y_v8]
core_classes_v8 = sorted(list(set(mapped_core_v8)))
y_core_v8 = np.array([core_classes_v8.index(m) for m in mapped_core_v8], dtype=np.int64)

print("=== EVALUATING V8 BACKUP DATASET ===")
print("X_v8 shape:", X_v8.shape)
print("X_emb_v8 shape:", X_emb_v8.shape)

# Let's test optical flow, pose, and cnn on v8 data
flow_v8 = X_emb_v8[:, :144]
cnn_v8 = X_emb_v8[:, 144:656] if X_emb_v8.shape[1] >= 656 else X_emb_v8[:, 144:]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for feat_name, feat_matrix in [
    ("v8 Pose Flat", X_v8.reshape(299, -1)),
    ("v8 Flow", flow_v8),
    ("v8 Entire X_emb", X_emb_v8),
]:
    scores = []
    for train_idx, test_idx in skf.split(feat_matrix, y_core_v8):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=42))
        clf.fit(feat_matrix[train_idx], y_core_v8[train_idx])
        preds = clf.predict(feat_matrix[test_idx])
        scores.append(accuracy_score(y_core_v8[test_idx], preds))
    print(f"{feat_name:25s}: Mean CV = {np.mean(scores)*100:.1f}%")
