import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from train_model import map_to_core_shot

d = np.load("cricket_data.npz", allow_pickle=True)
X_pose = d["X"]
X_emb = d["X_emb"]
raw_y = d["y"]
classes = [str(c) for c in d["classes"]]
mapped_core = [map_to_core_shot(classes[i]) for i in raw_y]
core_classes = sorted(list(set(mapped_core)))
y_core = np.array([core_classes.index(m) for m in mapped_core], dtype=np.int64)

# Test 1: Pose alone (flat: 16 * 34 = 544)
X_pose_flat = X_pose.reshape(len(X_pose), -1)

# Test 2: Optical flow alone (144)
X_flow = X_emb[:, :144]

# Test 3: 3D-CNN alone (512)
X_cnn = X_emb[:, 144:656]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, feat in [("Pose (544)", X_pose_flat), ("Flow (144)", X_flow), ("CNN (512)", X_cnn), ("All (1200)", np.hstack([X_pose_flat, X_flow, X_cnn]))]:
    fold_preds = np.zeros(len(y_core), dtype=int)
    for tr, val in skf.split(feat, y_core):
        clf = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(feat[tr])
        X_val = scaler.transform(feat[val])
        clf.fit(X_tr, y_core[tr])
        fold_preds[val] = clf.predict(X_val)
    acc = np.mean(fold_preds == y_core) * 100.0
    print(f"\nFeature: {name} -> 5-Fold CV Accuracy: {acc:.2f}%")

print("\n--- Confusion Matrix with Combined Features (LogisticRegression C=0.05) ---")
cm = confusion_matrix(y_core, fold_preds)
print("Classes:", core_classes)
for i, c in enumerate(core_classes):
    print(f"{c:20s}: {cm[i]}")
