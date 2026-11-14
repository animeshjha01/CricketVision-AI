import sys
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from train_model import map_to_core_shot, StackedEnsemble, build_hybrid_features, _make_ensemble

data = np.load("cricket_data.npz", allow_pickle=True)
X = data["X"]
X_emb = data["X_emb"]
y = data["y"]
classes = list(data["classes"])

X_feat = build_hybrid_features(X, X_emb)
core_classes = sorted(list(set(map_to_core_shot(c) for c in classes)))
y_core = np.array([core_classes.index(map_to_core_shot(classes[yi])) for yi in y])

print(f"Features shape: {X_feat.shape}")
print(f"Number of samples: {len(y)}")
print(f"Number of core classes: {len(core_classes)}")

# Let's run a clean 5-fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, clf_maker in [
    ("Stacked Ensemble (current)", lambda: _make_ensemble(X_feat.shape[1], len(core_classes))),
]:
    fold_accs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat, y_core)):
        clf = clf_maker()
        clf.fit(X_feat[train_idx], y_core[train_idx])
        preds = clf.predict(X_feat[val_idx])
        acc = accuracy_score(y_core[val_idx], preds) * 100.0
        fold_accs.append(acc)
        print(f"  Fold {fold+1}: {acc:.1f}%")
    print(f"{model_name} Mean 5-Fold CV Accuracy: {np.mean(fold_accs):.2f}% +/- {np.std(fold_accs):.2f}%")
