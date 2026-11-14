import re
from pathlib import Path

DATASET_DIR = Path("Cricket Shots")
DATA_NPZ_PATH = Path("cricket_data.npz")
MODEL_PATH = Path("shot_model.pth")

SEQ_LEN = 30
VALID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def clean_label(name: str) -> str:
    """Converts 'Advanced On-Drive' -> 'advanced_on_drive' for consistent matching."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")

# Bowler POV Matrix mapped to your exact Google Drive subfolder names
SELECTION_RULES = {
    ("Full", "Outside Off"): [
        "cover_drive",
        "advanced_lofted_cover_drive",
    ],
    ("Full", "Stumps"): [
        "advanced_on_drive",
        "flick_shot",
        "cover_drive",
        "front_foot_defense",
    ],
    ("Full", "Leg Side"): [
        "flick_shot",
        "advanced_on_drive",
    ],
    ("Good", "Outside Off"): [
        "cover_drive",
        "backfoot_punch",
        "backfoot_defense",
        "front_foot_defense",
    ],
    ("Good", "Stumps"): [
        "backfoot_defense",
        "front_foot_defense",
        "backfoot_flick",
        "backfoot_punch",
    ],
    ("Good", "Leg Side"): [
        "flick_shot",
        "backfoot_flick",
    ],
    ("Short", "Outside Off"): [
        "backfoot_punch",
        "backfoot_lofted_shot",
        "backfoot_defense",
    ],
    ("Short", "Stumps"): [
        "backfoot_defense",
        "backfoot_flick",
        "backfoot_lofted_shot",
    ],
    ("Short", "Leg Side"): [
        "backfoot_flick",
        "backfoot_lofted_shot",
    ],
}