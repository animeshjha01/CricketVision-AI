import sys
import joblib
import numpy as np
from extract_data import extract_sequence
from train_model import MODEL_PKL_PATH, build_hybrid_features, map_to_core_shot, StackedEnsemble

sys.modules["__main__"].StackedEnsemble = StackedEnsemble

bundle = joblib.load(MODEL_PKL_PATH)
core_model = bundle['core_model']
sub_model = bundle['sub_model']
core_classes = bundle['core_classes']
sub_classes = bundle['sub_classes']
feat_templates = bundle['feat_templates']

vids = [
    ('Cover Drive', r'Cricket Shots\Cover Drive\Screen Recording 2026-10-14 003430.mp4'),
    ('Pull Shot', r'Cricket Shots\Pull Shot\Screen Recording 2026-10-13 233206.mp4'),
    ('Square Cut', r'Cricket Shots\Square cut\Screen Recording 2026-10-14 013531.mp4'),
    ('Front Defense', r'Cricket Shots\Front-foot Defense\FD3.mp4'),
    ('Sweep', r'Cricket Shots\Sweep\Screen Recording 2026-10-16 020723.mp4'),
]

for label, vid in vids:
    pose_seq, video_emb, vis = extract_sequence(vid, return_vis=True)
    feat = build_hybrid_features(np.expand_dims(pose_seq, axis=0), np.expand_dims(video_emb, axis=0))
    c_p = core_model.predict_proba(feat)[0]
    s_p = sub_model.predict_proba(feat)[0]

    best_c = core_classes[np.argmax(c_p)]
    best_s = sub_classes[np.argmax(s_p)]

    # Also compute template cosine similarity for all sub classes!
    template_sims = {}
    for sk, tmpl in feat_templates.items():
        sim = float(np.dot(feat[0], tmpl) / (np.linalg.norm(feat[0]) * np.linalg.norm(tmpl) + 1e-8))
        template_sims[sk] = sim
    sorted_sims = sorted(template_sims.items(), key=lambda x: x[1], reverse=True)[:3]

    print(f"=== {label} ===")
    print(f"  Core argmax: {best_c} ({np.max(c_p):.3f})")
    print(f"  Sub argmax:  {best_s} ({np.max(s_p):.3f})")
    print(f"  Top templates: {sorted_sims}")
    top3_c = np.argsort(c_p)[::-1][:3]
    print(f"  Top 3 core: {[core_classes[i] + f': {c_p[i]:.2f}' for i in top3_c]}")
    print()
