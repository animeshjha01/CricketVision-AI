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

test_samples = [
    {
        "title": "🏏 Classic Flick Shot — MS Dhoni Style",
        "path": Path("Cricket Shots/Flick Shot/Screen Recording 2026-10-14 013130.mp4"),
        "expected_sub": "Flick Shot"
    },
    {
        "title": "🏏 Textbook On-Drive — Sachin Tendulkar Style",
        "path": Path("Cricket Shots/On-drive/Screen Recording 2026-10-14 003036.mp4"),
        "expected_sub": "On-drive"
    },
    {
        "title": "🪵 Precision Late Cut — Brian Lara Style",
        "path": Path("Cricket Shots/Late Cut/Screen Recording 2026-10-14 011439.mp4"),
        "expected_sub": "Late Cut"
    },
    {
        "title": "🧤 Solid Front-Foot Defense — Rahul Dravid Style",
        "path": Path("Cricket Shots/Front-foot Defense/Rahul-Dravid-Front-SnapYT.App.mp4"),
        "expected_sub": "Front-foot Defense"
    },
    {
        "title": "🏏 Classic Cover Drive — Virat Kohli Style",
        "path": Path("Cricket Shots/Cover Drive/Screen Recording 2026-10-14 003430.mp4"),
        "expected_sub": "Cover Drive"
    },
    {
        "title": "🔴 Backfoot Flick — Rohit Sharma Style",
        "path": Path("Cricket Shots/Backfoot flick/Screen Recording 2026-10-14 013858.mp4"),
        "expected_sub": "Backfoot flick"
    },
    {
        "title": "💥 Fierce Pull Shot — Ricky Ponting Style",
        "path": Path("Cricket Shots/Pull Shot/Screen Recording 2026-10-13 233206.mp4"),
        "expected_sub": "Pull Shot"
    },
    {
        "title": "🚀 Audacious Upper-Cut — Virender Sehwag Style",
        "path": Path("Cricket Shots/Upper-Cut/Screen Recording 2026-10-16 024259.mp4"),
        "expected_sub": "Upper-Cut"
    }
]

print("=== VERIFYING ACCURATELY CLASSIFIED SAMPLES ===")
for s in test_samples:
    pose_seq, video_emb = extract_sequence(s["path"])
    X_f = build_hybrid_features(pose_seq[np.newaxis, ...], video_emb[np.newaxis, ...])
    sub_probs = sub_model.predict_proba(X_f)[0]
    s_idx = np.argmax(sub_probs)
    raw = sub_classes[s_idx]
    pred_sub = sub_display_names.get(raw, raw)
    conf = sub_probs[s_idx] * 100
    match = pred_sub.lower() == s["expected_sub"].lower()
    print(f"[{'PASS' if match else 'FAIL'}] {s['title']}")
    print(f"       File: {s['path']}")
    print(f"       Expected: {s['expected_sub']} | Got: {pred_sub} ({conf:.1f}%)\n")
