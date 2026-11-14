import os, sys
import joblib, numpy as np
from extract_data import extract_sequence
from train_model import MODEL_PKL_PATH, build_hybrid_features, map_to_core_shot, StackedEnsemble

sys.modules["__main__"].StackedEnsemble = StackedEnsemble

bundle = joblib.load(MODEL_PKL_PATH)
core_model = bundle['core_model']
sub_model = bundle['sub_model']
core_classes = bundle['core_classes']
sub_classes = bundle['sub_classes']
sub_display_names = bundle['sub_display_names']

base = r"c:\cricket_ml_project\Cricket Shots"
folders = os.listdir(base)

results = []

for folder in folders:
    folder_path = os.path.join(base, folder)
    if not os.path.isdir(folder_path):
        continue
    mp4s = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    if not mp4s:
        continue
    
    # Test first 2 videos
    for v_name in mp4s[:2]:
        v_path = os.path.join(folder_path, v_name)
        try:
            pose_seq, video_emb = extract_sequence(v_path)
            if pose_seq is None or video_emb is None:
                continue
            feat = build_hybrid_features(np.expand_dims(pose_seq, axis=0), np.expand_dims(video_emb, axis=0))
            core_probs = core_model.predict_proba(feat)[0]
            sub_probs = sub_model.predict_proba(feat)[0]
            
            combined_scores = np.zeros(len(sub_classes), dtype=np.float32)
            for i, sub_k in enumerate(sub_classes):
                c_idx = core_classes.index(map_to_core_shot(sub_k))
                combined_scores[i] = (core_probs[c_idx] ** 0.60) * (sub_probs[i] ** 0.40)
            
            best_idx = int(np.argmax(combined_scores))
            pred_sub = sub_classes[best_idx]
            pred_name = sub_display_names.get(pred_sub, pred_sub).title()
            
            results.append({
                "folder": folder,
                "file": v_name,
                "path": v_path,
                "pred": pred_name,
                "score": float(combined_scores[best_idx])
            })
            print(f"[{folder}] -> Predicted: {pred_name} (score: {combined_scores[best_idx]:.3f})")
        except Exception as e:
            print(f"Error on {v_path}: {e}")

print("\n--- Summary of matches ---")
for r in results:
    if r['folder'].lower().replace('_', ' ').replace('-', ' ') in r['pred'].lower().replace('_', ' ').replace('-', ' ') or r['pred'].lower() in r['folder'].lower():
        print(f"MATCH! Folder: {r['folder']} | Video: {r['file']} | Predicted: {r['pred']}")
