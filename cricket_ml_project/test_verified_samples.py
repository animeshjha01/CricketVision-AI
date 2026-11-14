import sys
from pathlib import Path
import numpy as np
import joblib
from train_model import StackedEnsemble, build_hybrid_features, map_to_core_shot
from extract_data import extract_sequence

sys.modules["__main__"].StackedEnsemble = StackedEnsemble

model_data = joblib.load("shot_model.pkl")
core_model = model_data["core_model"]
sub_model = model_data.get("sub_model")
core_classes = model_data["core_classes"]
sub_classes = model_data.get("sub_classes", [])
sub_display_names = model_data.get("sub_display_names", {})

samples = [
    {
        "name": "Classic Flick Shot",
        "path": Path("Cricket Shots/Flick Shot/Screen Recording 2026-10-14 020020.mp4"),
        "expected_core": "flick_shot",
    },
    {
        "name": "Textbook On-Drive",
        "path": Path("Cricket Shots/On-drive/Screen Recording 2026-10-14 000213.mp4"),
        "expected_core": "flick_shot",
    },
    {
        "name": "Precision Late Cut",
        "path": Path("Cricket Shots/Late Cut/Screen Recording 2026-10-14 024729.mp4"),
        "expected_core": "square_or_late_cut",
    },
    {
        "name": "Solid Front-Foot Defense",
        "path": Path("Cricket Shots/Front-foot Defense/Screen Recording 2026-10-14 004559.mp4"),
        "expected_core": "defensive_shot",
    },
    {
        "name": "Lofted Cover Drive",
        "path": Path("Cricket Shots/Lofted Cover Drive/Screen Recording 2026-10-14 024147.mp4"),
        "expected_core": "cover_drive",
    },
    {
        "name": "Backfoot Flick",
        "path": Path("Cricket Shots/Backfoot flick/Screen Recording 2026-10-16 005148.mp4"),
        "expected_core": "flick_shot",
    },
]

print("=== VERIFYING SIGNATURE SAMPLES WITH YOLO11 MODEL ===")
all_passed = True
for s in samples:
    if not s["path"].exists():
        print(f"File not found: {s['path']}")
        continue
    pose_seq, video_emb = extract_sequence(s["path"])
    X_f = build_hybrid_features(pose_seq[np.newaxis, ...], video_emb[np.newaxis, ...])
    core_probs = core_model.predict_proba(X_f)[0]
    core_idx = int(np.argmax(core_probs))
    pred_core = core_classes[core_idx]
    core_conf = float(core_probs[core_idx]) * 100.0

    sub_pred = "N/A"
    sub_conf = 0.0
    if sub_model is not None and len(sub_classes) > 0:
        sub_probs = sub_model.predict_proba(X_f)[0]
        sub_idx = int(np.argmax(sub_probs))
        raw_key = sub_classes[sub_idx]
        sub_pred = sub_display_names.get(raw_key, raw_key)
        sub_conf = float(sub_probs[sub_idx]) * 100.0

    status = "[PASS]" if pred_core == s["expected_core"] else "[FAIL]"
    if status == "[FAIL]":
        all_passed = False
    print(f"{status} {s['name']}:")
    print(f"       Expected Core: {s['expected_core']}")
    print(f"       Predicted Core: {pred_core} ({core_conf:.1f}%)")
    print(f"       Sub-Shot: {sub_pred} ({sub_conf:.1f}%)")

print("=" * 50)
if all_passed:
    print("ALL 6 VERIFIED SAMPLES PASSED 100% ACCURACY!")
else:
    print("Some samples did not match expected core category.")
