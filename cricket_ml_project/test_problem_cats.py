import sys
from pathlib import Path
import numpy as np
import app

categories = ["Cover Drive", "Backfoot Punch", "Front-foot Defense"]

for cat in categories:
    folder = Path("Cricket Shots") / cat
    vids = sorted(list(folder.glob("*.mp4")))
    print(f"\n==========================================")
    print(f"TESTING ALL VIDEOS IN: {cat} (Total {len(vids)})")
    print(f"==========================================")
    for v in vids:
        res = app.analyze_video(str(v), "Good", "Stumps")
        if "error" in res:
            print(f"  {v.name}: ERROR -> {res['error']}")
        else:
            print(f"  {v.name:45s} -> Shot: {res['shot_played']:22s} | Family: {res['shot_family']:20s} | Conf: {res['confidence']}")
