import sys
from pathlib import Path
import numpy as np
import joblib
from train_model import StackedEnsemble, build_hybrid_features
from extract_data import extract_sequence

sys.modules["__main__"].StackedEnsemble = StackedEnsemble

model_data = joblib.load("shot_model.pkl")
core_model = model_data["core_model"]
sub_model = model_data.get("sub_model")
core_classes = model_data["core_classes"]
sub_classes = model_data.get("sub_classes", [])
sub_display_names = model_data.get("sub_display_names", {})

folders_to_check = [
    "Cover Drive",
    "Flick Shot",
    "Front-foot Defense",
    "Late Cut",
    "On-drive",
    "Pull Shot",
    "Square cut",
    "Upper-Cut",
    "Backfoot flick"
]

print("=== CHECKING MODEL PREDICTIONS ON REAL CLIPS ===")
for folder_name in folders_to_check:
    folder = Path("Cricket Shots") / folder_name
    if not folder.exists():
        continue
    clips = list(folder.glob("*.mp4"))[:3]
    print(f"\n--- Category: {folder_name} ---")
    for clip in clips:
        pose_seq, video_emb = extract_sequence(clip)
        X_f = build_hybrid_features(pose_seq[np.newaxis, ...], video_emb[np.newaxis, ...])
        core_probs = core_model.predict_proba(X_f)[0]
        c_idx = np.argmax(core_probs)
        pred_core = core_classes[c_idx]
        conf = core_probs[c_idx] * 100

        sub_pred = "N/A"
        if sub_model is not None and len(sub_classes) > 0:
            s_probs = sub_model.predict_proba(X_f)[0]
            s_idx = np.argmax(s_probs)
            raw = sub_classes[s_idx]
            sub_pred = sub_display_names.get(raw, raw)
            sub_conf = s_probs[s_idx] * 100

        print(f"  Clip: {clip.name}")
        print(f"    Core -> {pred_core} ({conf:.1f}%) | Sub -> {sub_pred} ({sub_conf:.1f}%)")
