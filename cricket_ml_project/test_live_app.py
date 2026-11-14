import app
from pathlib import Path

test_vids = [
    ("Cover Drive 1", "Cricket Shots/Cover Drive/Screen Recording 2026-10-14 003430.mp4"),
    ("Cover Drive 2", "Cricket Shots/Cover Drive/Screen Recording 2026-10-14 013231.mp4"),
    ("Backfoot Punch 1", "Cricket Shots/Backfoot Punch/Screen Recording 2026-10-14 002446.mp4"),
    ("Backfoot Punch 2", "Cricket Shots/Backfoot Punch/Screen Recording 2026-10-14 015850.mp4"),
    ("Front-foot Defense 1", "Cricket Shots/Front-foot Defense/FD3.mp4"),
    ("Front-foot Defense 2", "Cricket Shots/Front-foot Defense/Rahul-Dravid-Front-SnapYT.App.mp4"),
    ("Front-foot Defense 3", "Cricket Shots/Front-foot Defense/Screen Recording 2026-10-13 224107.mp4"),
    ("Flick Shot 1", "Cricket Shots/Flick Shot/Screen Recording 2026-10-14 013130.mp4"),
    ("Flick Shot 2", "Cricket Shots/Flick Shot/Screen Recording 2026-10-14 013318.mp4"),
    ("Pull Shot 1", "Cricket Shots/Pull Shot/Screen Recording 2026-10-13 233206.mp4"),
    ("Square cut 1", "Cricket Shots/Square cut/Screen Recording 2026-10-14 013531.mp4"),
    ("Upper-Cut 1", "Cricket Shots/Upper-Cut/Screen Recording 2026-10-16 024259.mp4"),
    ("Sweep 1", "Cricket Shots/Sweep/Screen Recording 2026-10-16 020723.mp4"),
    ("Scoop 1", "Cricket Shots/Scoop/Screen Recording 2026-10-13 235252.mp4"),
]

print("=== APP.PY LIVE PREDICTIONS TEST ===")
for name, p in test_vids:
    res = app.analyze_video(p, "Good", "Stumps")
    if "error" in res:
        print(f"[{name}] ERROR: {res['error']}")
    else:
        print(f"[{name:22s}] -> Shot: {res['shot_played']:22s} | Family: {res['shot_family']:20s} | Conf: {res['confidence']}")
