import app

samples = [
    ("Cover Drive", r"Cricket Shots\Cover Drive\Screen Recording 2026-10-14 003430.mp4", "Full", "Outside Off"),
    ("Backfoot Punch", r"Cricket Shots\Backfoot Punch\Screen Recording 2026-10-14 002446.mp4", "Good", "Outside Off"),
    ("Front-foot Defense", r"Cricket Shots\Front-foot Defense\Rahul-Dravid-Front-SnapYT.App.mp4", "Good", "Stumps"),
    ("Flick Shot", r"Cricket Shots\Flick Shot\Screen Recording 2026-10-14 013130.mp4", "Full", "Leg Side"),
    ("Pull Shot", r"Cricket Shots\Pull Shot\Screen Recording 2026-10-13 233206.mp4", "Short", "Leg Side"),
    ("On-drive", r"Cricket Shots\On-drive\Screen Recording 2026-10-14 003036.mp4", "Full", "Stumps"),
    ("Late Cut", r"Cricket Shots\Late Cut\Screen Recording 2026-10-14 011439.mp4", "Short", "Outside Off"),
    ("Square cut", r"Cricket Shots\Square cut\Screen Recording 2026-10-14 013531.mp4", "Short", "Outside Off"),
    ("Upper-Cut", r"Cricket Shots\Upper-Cut\Screen Recording 2026-10-16 024259.mp4", "Short", "Outside Off"),
    ("Sweep", r"Cricket Shots\Sweep\Screen Recording 2026-10-16 020723.mp4", "Good", "Stumps"),
    ("Scoop", r"Cricket Shots\Scoop\Screen Recording 2026-10-13 235252.mp4", "Full", "Stumps"),
]

print("=== VERIFYING ALL 11 SIGNATURE SHOTS ===")
all_pass = True
for name, p, l, line in samples:
    res = app.analyze_video(p, l, line)
    match = name.lower() in res["shot_played"].lower() or res["shot_played"].lower() in name.lower()
    status = "[PASS]" if match else "[FAIL]"
    if not match: all_pass = False
    print(f"{status} {name:20s} -> Predicted: {res['shot_played']:22s} ({res['confidence']})")

if all_pass:
    print("\n>>> 100% OF ALL 11 DEMONSTRATION SHOTS MATCH EXACTLY! <<<")
else:
    print("\n>>> SOME SAMPLES FAILED! <<<")
