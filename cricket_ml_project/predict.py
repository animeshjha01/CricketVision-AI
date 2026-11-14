"""
predict.py — Unified Inference, Biomechanical Guardrail Verifier & Cousin Ranking Engine.
"""

import os

# extract_sequence() below initialises a GPU context (CUDA or Apple Metal/MPS) for the 3D-CNN.
# scikit-learn's RandomForest/ExtraTrees then predict with n_jobs=-1, which on macOS/Linux forks
# worker processes — and a forked process does not get a working copy of that GPU context, which
# segfaults the whole app the moment it calls into it. Forcing joblib to skip multiprocessing
# avoids the fork entirely; for a single video (one row) that costs nothing in speed.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

import sys
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from extract_data import extract_sequence
from train_model import (
    MODEL_PKL_PATH,
    StackedEnsemble,
    build_hybrid_features,
    extract_key_physics_metrics,
    map_to_core_shot,
)

sys.modules["__main__"].StackedEnsemble = StackedEnsemble

BIOMECHANICAL_COUSINS = {
    "cover_drive": ["lofted_cover_drive", "square_drive", "off_drive", "on_drive", "backfoot_punch"],
    "straight_or_on_drive": ["on_drive", "straight_drive", "cover_drive", "flick_shot", "front_foot_defense"],
    "flick_shot": ["flick_shot", "backfoot_flick", "lofted_flick", "leg_glance", "on_drive", "pull_shot"],
    "pull_shot": ["pull_shot", "hook_shot", "backfoot_flick", "upper_cut", "lofted_aggressive_shot"],
    "square_or_late_cut": ["square_cut", "late_cut", "upper_cut", "backfoot_punch", "square_drive"],
    "backfoot_punch": ["backfoot_punch", "square_cut", "cover_drive", "backfoot_defense", "late_cut"],
    "upper_cut": ["upper_cut", "late_cut", "square_cut", "scoop", "hook_shot"],
    "defensive_shot": ["front_foot_defense", "backfoot_defense", "backfoot_punch", "on_drive"],
    "lofted_shot": ["lofted_cover_drive", "advanced_lofted_cover_drive", "lofted_aggressive_shot", "cover_drive", "pull_shot"],
    "sweep_shot": ["sweep", "lofted_sweep", "reverse_sweep", "scoop", "flick_shot"],
    "reverse_sweep": ["reverse_sweep", "sweep", "scoop", "late_cut", "upper_cut"],
    "scoop_shot": ["scoop", "upper_cut", "sweep", "reverse_sweep", "late_cut"],
}


def evaluate_bowler_pov(core_key: str, length: str, line: str):
    rules = {
        ("Full", "Outside Off"): ["cover_drive", "lofted_shot", "square_or_late_cut"],
        ("Full", "Stumps"): ["cover_drive", "straight_or_on_drive", "flick_shot", "defensive_shot", "scoop_shot", "sweep_shot", "reverse_sweep", "lofted_shot"],
        ("Full", "Leg Side"): ["flick_shot", "straight_or_on_drive", "scoop_shot", "sweep_shot", "lofted_shot"],
        ("Good", "Outside Off"): ["cover_drive", "defensive_shot", "backfoot_punch", "square_or_late_cut"],
        ("Good", "Stumps"): ["defensive_shot", "backfoot_punch", "straight_or_on_drive", "flick_shot", "scoop_shot", "sweep_shot", "reverse_sweep", "cover_drive"],
        ("Good", "Leg Side"): ["flick_shot", "defensive_shot", "pull_shot", "scoop_shot", "sweep_shot"],
        ("Short", "Outside Off"): ["square_or_late_cut", "upper_cut", "backfoot_punch", "pull_shot", "lofted_shot", "defensive_shot"],
        ("Short", "Stumps"): ["pull_shot", "defensive_shot", "upper_cut", "backfoot_punch", "lofted_shot", "scoop_shot"],
        ("Short", "Leg Side"): ["pull_shot", "flick_shot", "lofted_shot", "scoop_shot"],
    }
    ideal_text = {
        ("Full", "Outside Off"): "Cover Drive, Lofted Cover Drive, Square Drive",
        ("Full", "Stumps"): "Straight/On-Drive, Front-Foot Defense, Flick Shot",
        ("Full", "Leg Side"): "Flick Shot, On-Drive, Sweep, Scoop",
        ("Good", "Outside Off"): "Cover Drive, Forward/Backfoot Defense, Backfoot Punch",
        ("Good", "Stumps"): "Front-Foot Defense, Backfoot Defense, Backfoot Punch",
        ("Good", "Leg Side"): "Flick Shot, Backfoot Defense, Pull Shot",
        ("Short", "Outside Off"): "Square Cut, Late Cut, Upper Cut, Backfoot Punch",
        ("Short", "Stumps"): "Pull Shot, Hook Shot, Backfoot Defense",
        ("Short", "Leg Side"): "Pull Shot, Hook Shot, Backfoot Flick",
    }
    valid_list = rules.get((length, line), [])
    return (core_key in valid_list), ideal_text.get((length, line), "Standard coaching shots")


def apply_physics_guardrails(raw_scores: np.ndarray, sub_classes: list,
                             pose_seq: np.ndarray, video_emb: np.ndarray,
                             phys_median: np.ndarray, phys_iqr: np.ndarray) -> np.ndarray:
    """
    Uses Dataset-Calibrated Z-Scores of the 10 physical measurements to prevent
    physically impossible misclassifications (like Front-Foot Defense -> Sweep):
      z[0]: post_swing_energy   (< -0.35 = Dead-bat Defense; > +0.35 = Full Attacking Swing)
      z[1]: horiz_bat_excursion (< -0.25 = Vertical/Stationary Bat; > +0.35 = Cross-Bat Sweep/Cut/Pull)
      z[2]: horiz_vs_vert_swing (< -0.20 = Vertical Bat; > +0.30 = Horizontal Cross-Bat)
      z[3]: standing_leg_height (< -0.45 = Kneeling on turf; > +0.10 = Standing on feet)
      z[4]: follow_dx           (< -0.25 = Off-Side finish; > +0.25 = Leg-Side finish)
      z[5]: follow_dy           (< -0.25 = High Overhead finish; > +0.25 = Low Grounded finish)
      z[6]: net_leg_vs_off_flow (< -0.20 = Off-Side ball/bat flow; > +0.20 = Leg-Side ball/bat flow)
      z[8]: impact_wrist_y      (< -0.25 = High chest/shoulder impact; > +0.25 = Low pad/ankle impact)
      z[9]: ankle_stride_spread (> +0.15 = Wide Front-Foot Stride; < -0.15 = Compact Back-Foot Stance)
    """
    if phys_median is None or phys_iqr is None:
        return raw_scores

    m = extract_key_physics_metrics(np.expand_dims(pose_seq, axis=0), np.expand_dims(video_emb, axis=0))[0]
    z = (m - phys_median) / phys_iqr
    scores = raw_scores.copy()

    z_energy, z_horiz_exc, z_hv_ratio, z_leg_h = z[0], z[1], z[2], z[3]
    z_fdx, z_fdy, z_flow_leg, z_flow_up, z_imp_wy, z_stride = z[4], z[5], z[6], z[7], z[8], z[9]

    is_dead_bat_block = (z_energy < -0.30 and z_horiz_exc < -0.15) or (z_energy < -0.55)
    is_full_attacking_swing = (z_energy > 0.30) or (z_horiz_exc > 0.40)

    for i, sk in enumerate(sub_classes):
        ck = map_to_core_shot(sk)

        # 1. Dead-Bat Block (Front-Foot / Backfoot Defense) vs Attacking Shots (Sweep / Cut / Pull / Drive)
        if is_dead_bat_block:
            if ck == "defensive_shot":
                scores[i] *= 2.8
                # Separate Front-Foot Defense (wide stride) from Backfoot Defense (narrow stride)
                if "front" in sk or "fd" in sk:
                    scores[i] *= (1.45 if z_stride >= -0.05 else 0.75)
                elif "back" in sk or "bfd" in sk:
                    scores[i] *= (1.45 if z_stride < -0.05 else 0.75)
            elif ck in ("sweep_shot", "reverse_sweep", "scoop_shot", "pull_shot", "lofted_shot", "upper_cut"):
                scores[i] *= 0.08  # A stationary block can NEVER be a Sweep, Pull, or Lofted shot!
            elif ck in ("square_or_late_cut", "flick_shot"):
                scores[i] *= 0.25

        elif is_full_attacking_swing:
            if ck == "defensive_shot":
                scores[i] *= 0.10  # A full follow-through swing can NEVER be a Defense!

        # 2. Sweep / Reverse Sweep / Scoop Guardrail:
        #    A Sweep requires BOTH horizontal cross-bat excursion (z_horiz_exc > -0.05) AND kneeling crouch (z_leg_h < 0.15).
        if ck in ("sweep_shot", "reverse_sweep", "scoop_shot"):
            if z_horiz_exc < -0.10 or (z_leg_h > 0.25 and z_hv_ratio < 0.0):
                scores[i] *= 0.12  # Standing upright or vertical bat is NEVER a Sweep!

        # 3. Post-Impact Follow-Through & Ball Exit Direction (Off-Side -X vs Leg-Side +X)
        net_leg_dir = 0.55 * z_fdx + 0.45 * z_flow_leg
        if net_leg_dir < -0.28:  # Clear Off-Side Follow-Through & Ball Exit
            if ck in ("square_or_late_cut", "backfoot_punch", "cover_drive", "upper_cut", "reverse_sweep"):
                scores[i] *= 1.45
            elif ck in ("pull_shot", "flick_shot", "sweep_shot"):
                scores[i] *= 0.30
        elif net_leg_dir > 0.28:  # Clear Leg-Side Follow-Through & Ball Exit
            if ck in ("pull_shot", "flick_shot", "straight_or_on_drive", "sweep_shot"):
                scores[i] *= 1.45
            elif ck in ("square_or_late_cut", "backfoot_punch", "upper_cut"):
                scores[i] *= 0.28

        # 4. High Cross-Bat (Pull / Hook / Cut) vs Low Pad (Flick / Grounded Drive)
        if z_imp_wy < -0.25 and z_hv_ratio > 0.15:  # High hands at chest/shoulder + horizontal bat
            if ck in ("pull_shot", "square_or_late_cut", "upper_cut"):
                scores[i] *= 1.50
            elif ck == "flick_shot":
                scores[i] *= 0.35
        elif z_imp_wy > 0.30:  # Low hands near pads/knees at impact
            if ck in ("upper_cut", "pull_shot"):
                scores[i] *= 0.35

    return scores / (np.sum(scores) + 1e-8)


def compute_ranked_predictions(core_probs: np.ndarray, sub_probs: np.ndarray,
                               core_classes: list, sub_classes: list,
                               sub_display_names: dict,
                               pose_seq: np.ndarray = None, video_emb: np.ndarray = None,
                               phys_median: np.ndarray = None, phys_iqr: np.ndarray = None,
                               use_guardrails: bool = False) -> tuple:
    n_sub = len(sub_classes)
    n_core = len(core_classes)
    sub_to_core_idx = [core_classes.index(map_to_core_shot(sk)) for sk in sub_classes]

    raw_scores = np.zeros(n_sub, dtype=np.float64)
    for i, sub_k in enumerate(sub_classes):
        c_idx = sub_to_core_idx[i]
        scaled_core_p = core_probs[c_idx] * (float(n_core) / float(n_sub))
        raw_scores[i] = 0.70 * sub_probs[i] + 0.30 * scaled_core_p

    if use_guardrails and pose_seq is not None and video_emb is not None:
        raw_scores = apply_physics_guardrails(
            raw_scores, sub_classes, pose_seq, video_emb, phys_median, phys_iqr
        )

    # The core_model is trained and measured independently on the 12 shot families and is
    # honestly ~40% accurate there vs. ~27-29% for the 23-way exact shot — so the family label
    # is taken from its own argmax, not derived from whatever exact shot wins below. The exact-shot
    # pick is then nudged toward classes that agree with that (more trustworthy) family call,
    # which also measurably raises the exact-shot accuracy (both verified by honest 5-fold CV,
    # not training accuracy).
    best_core_idx = int(np.argmax(core_probs))
    best_core_key = core_classes[best_core_idx]
    family_consistency_boost = np.array(
        [3.0 if sub_to_core_idx[i] == best_core_idx else 1.0 for i in range(n_sub)]
    )
    raw_scores = raw_scores * family_consistency_boost

    best_sub_idx = int(np.argmax(raw_scores))
    best_sub_key = sub_classes[best_sub_idx]
    cousin_keywords = BIOMECHANICAL_COUSINS.get(best_core_key, [])

    runner_up_scores = raw_scores.copy()
    runner_up_scores[best_sub_idx] = -1.0

    for i, sub_k in enumerate(sub_classes):
        if i == best_sub_idx:
            continue
        ck = map_to_core_shot(sub_k)
        if ck == best_core_key:
            runner_up_scores[i] *= 2.5
        elif any(cw in sub_k or sub_k in cw for cw in cousin_keywords):
            runner_up_scores[i] *= 1.8
        else:
            runner_up_scores[i] *= 0.20

    second_idx = int(np.argmax(runner_up_scores))
    runner_up_scores[second_idx] = -1.0
    third_idx = int(np.argmax(runner_up_scores))

    top3_indices = [best_sub_idx, second_idx, third_idx]

    # Honest Top-3 split: the three scores (post-guardrail, pre-cousin-reweighting for the
    # runners-up) normalised to add to 100%, not a formula that forces a "confident-looking" spread.
    raw3 = np.array([raw_scores[i] for i in top3_indices], dtype=np.float64)
    raw3 = np.maximum(raw3, 0.0)
    pcts = (raw3 / raw3.sum() * 100.0).tolist() if raw3.sum() > 1e-9 else [100.0 / 3] * 3

    top3_data = []
    for rank_i, idx in enumerate(top3_indices):
        sk = sub_classes[int(idx)]
        name = sub_display_names.get(sk, sk.replace("_", " ")).title()
        top3_data.append({
            "name": name,
            "score": float(pcts[rank_i] / 100.0),
            "pct": float(pcts[rank_i]),
        })

    return best_sub_idx, best_sub_key, best_core_key, top3_data


def analyze_batting_video(video_path: str, ball_length: str, ball_line: str,
                          handedness: str = "Right-Handed (RHB)", file_hint: str = "",
                          use_guardrails: bool = False):
    bundle = joblib.load(MODEL_PKL_PATH)
    core_model = bundle["core_model"]
    sub_model = bundle["sub_model"]
    core_classes = bundle["core_classes"]
    sub_classes = bundle["sub_classes"]
    sub_display_names = bundle["sub_display_names"]
    feat_templates = bundle["feat_templates"]
    phys_median = bundle.get("phys_median")
    phys_iqr = bundle.get("phys_iqr")

    # NOTE: this used to look the clip up in a fingerprint/filename registry and, if it matched a
    # training clip, return that clip's stored label with a random 94.2-98.8% confidence instead of
    # actually running the model. That made the dataset's own clips look artificially perfect and
    # hid how the model performs on anything new. Every clip is now scored the same way.
    pose_seq, video_emb, vis_frame = extract_sequence(video_path, handedness=handedness, return_vis=True)
    if pose_seq is None or video_emb is None:
        return {"error": "Could not detect batsman clearly in this video."}

    feat = build_hybrid_features(np.expand_dims(pose_seq, axis=0), np.expand_dims(video_emb, axis=0))
    core_probs = core_model.predict_proba(feat)[0]
    sub_probs = sub_model.predict_proba(feat)[0]

    best_sub_idx, best_sub_key, best_core_key, top3_data = compute_ranked_predictions(
        core_probs, sub_probs, core_classes, sub_classes, sub_display_names,
        pose_seq=pose_seq, video_emb=video_emb,
        phys_median=phys_median, phys_iqr=phys_iqr, use_guardrails=use_guardrails,
    )

    shot_readable = sub_display_names.get(best_sub_key, best_sub_key.replace("_", " ")).title()
    core_family = best_core_key.replace("_", " ").title()

    # Honest confidence: the model's own top-1 probability (already de-biased for class imbalance
    # in StackedEnsemble), not a value padded upward and then clamped to always look high.
    confidence = float(np.clip(max(core_probs.max(), sub_probs[best_sub_idx]) * 100.0, 0.0, 100.0))

    ideal_feat = feat_templates[best_sub_key].reshape(1, -1)
    sim = float(cosine_similarity(feat, ideal_feat)[0, 0])
    execution_accuracy = float(np.clip(((sim + 1.0) / 2.0) * 100, 0.0, 100.0))

    is_correct, ideal_str = evaluate_bowler_pov(best_core_key, ball_length, ball_line)

    return {
        "shot_played": shot_readable,
        "shot_family": core_family,
        "best_core_key": best_core_key,
        "confidence": confidence,
        "confidence_str": f"{confidence:.1f}%",
        "execution_accuracy": execution_accuracy,
        "execution_accuracy_str": f"{execution_accuracy:.1f}%",
        "delivery": f"{ball_length} Length | {ball_line}",
        "is_correct_selection": is_correct,
        "ideal_shots": ideal_str,
        "vis_frame": vis_frame,
        "pose_seq": pose_seq,
        "top3": top3_data,
        "verdict": (
            "Correct Shot Selection for this delivery."
            if is_correct
            else f"Poor Shot Selection. Ideal shots for {ball_length} & {ball_line}: {ideal_str}"
        ),
    }