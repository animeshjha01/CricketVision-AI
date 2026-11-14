import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from train_model import map_to_core_shot, build_hybrid_features

d = np.load("cricket_data.npz", allow_pickle=True)
X_pose = d["X"]
X_emb = d["X_emb"]
y = d["y"]
classes = [str(c) for c in d["classes"]]
core_classes = sorted(list(set(map_to_core_shot(c) for c in classes)))
y_core = np.array([core_classes.index(map_to_core_shot(classes[yi])) for yi in y])

X_feat = build_hybrid_features(X_pose, X_emb)
print(f"X_feat shape: {X_feat.shape}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k_features in [40, 60, 80, 120, 200, 350, 500, X_feat.shape[1]]:
    fold_accs = []
    for tr, val in skf.split(X_feat, y_core):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_feat[tr])
        X_val = scaler.transform(X_feat[val])

        if k_features < X_feat.shape[1]:
            selector = SelectKBest(f_classif, k=k_features)
            X_tr = selector.fit_transform(X_tr, y_core[tr])
            X_val = selector.transform(X_val)

        clf = RidgeClassifierCV(alphas=np.logspace(-2, 3, 20))
        clf.fit(X_tr, y_core[tr])
        preds = clf.predict(X_val)
        fold_accs.append(accuracy_score(y_core[val], preds))

    print(f"k={k_features:4d} features -> Ridge CV Acc: {np.mean(fold_accs)*100:.2f}% +/- {np.std(fold_accs)*100:.2f}%")
