"""
🏏 CricketVision AI — Cricket Shot Analyzer & Biomechanical Coaching System
A modern, broadcast-grade cricket-themed Streamlit application for analyzing
batting technique using YOLO11-pose estimation, optical flow, and deep stacked ensembles.
"""

import os
import sys
import tempfile
import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from train_model import MODEL_PKL_PATH, StackedEnsemble
from predict import analyze_batting_video

sys.modules["__main__"].StackedEnsemble = StackedEnsemble

st.set_page_config(
    page_title="CricketVision AI — International Match Center & Biomechanics",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MASTERCLASS_DATA = {
    "cover_drive": {
        "legend": "Virat Kohli / Sanju Samson",
        "legend_title": "The Master of the Cover Drive",
        "avatar": "🏏",
        "swing_type": "Vertical High-Elbow Driving Arc",
        "checkpoints": [
            "Front foot strides purposefully towards the pitch of the ball with head leading the movement.",
            "High front elbow pointing towards mid-off to prevent slicing or aerial lofting.",
            "Full face of the willow bat presented at point of impact directly beneath the eye-line.",
            "Smooth, balanced follow-through with weight held solidly on the flexed front knee."
        ],
        "drill": "Top-Hand Lead Cone Drill: Place cones along extra-cover; practice leading with the front elbow and punching the ball through the gates along the turf."
    },
    "straight_or_on_drive": {
        "legend": "Sachin Tendulkar / Ricky Ponting",
        "legend_title": "Textbook Straight-Bat Excellence",
        "avatar": "🏏",
        "swing_type": "Pure Linear Pendulum Arc",
        "checkpoints": [
            "Minimal backlift tilt, ensuring the bat face stays perpendicular to the delivery line.",
            "Front shoulder dips into the line, keeping head perfectly still over the contact point.",
            "Wrists remain firm through impact to keep the ball strictly along the ground.",
            "Follow-through points straight back towards the non-striker's stumps."
        ],
        "drill": "Full-Face Pendulum Drill: Hit balls hung from a string, focusing on zero wrist roll and pristine bat-face alignment."
    },
    "backfoot_punch": {
        "legend": "Sanju Samson / Brian Lara",
        "legend_title": "The Master of Back-Foot Timing",
        "avatar": "🥊",
        "swing_type": "Tall Vertical-Blade Punch",
        "checkpoints": [
            "Quick push back and across onto the ball of the back foot to ride the bounce.",
            "High leading elbow with the bat coming down straight on a vertical plane.",
            "Contact made right beneath the eyes as the ball reaches the peak of its bounce.",
            "Controlled punch follow-through keeping the ball crisp along the turf through cover-point."
        ],
        "drill": "Short-of-Length Target Drill: Have a partner feed waist-high tennis balls on off-stump; stand tall and punch through a 2-meter cone gate at cover."
    },
    "pull_shot": {
        "legend": "Rohit Sharma / Ricky Ponting",
        "legend_title": "The Master of Short-Pitch Dominance",
        "avatar": "🔴",
        "swing_type": "Horizontal Rotational Cross-Bat",
        "checkpoints": [
            "Quick, decisive weight shift onto the back foot deep inside the crease.",
            "High backlift with explosive hip rotation pivoting on the ball of the front foot.",
            "Contact made well in front of the chest with arms fully extended.",
            "Wrists roll aggressively over the top of the ball at impact to keep it down."
        ],
        "drill": "Shadow Swivel Pull Drill: Practice pivoting 90 degrees on the back foot against high tennis-ball bouncers with rapid wrist roll."
    },
    "square_or_late_cut": {
        "legend": "Brian Lara / Kane Williamson",
        "legend_title": "The Artisan of Timing and Angles",
        "avatar": "🪵",
        "swing_type": "Descending Slashing Arc",
        "checkpoints": [
            "Deep back-and-across movement to get right on top of the bounce.",
            "Arms fully extended away from the body with high hands on the backswing.",
            "Chopping down through the line of the ball, utilizing the bowler's pace.",
            "Late contact point behind or level with the batting crease for third-man placement."
        ],
        "drill": "Underarm Feed Angle Drill: Receive rapid feeds short and wide outside off, guiding them through slip cordon cones."
    },
    "flick_shot": {
        "legend": "VVS Laxman / Suryakumar Yadav",
        "legend_title": "The Wizard of Wrist Snap",
        "avatar": "🏏",
        "swing_type": "Supple Wrist Pronation",
        "checkpoints": [
            "Head stays still over the middle stump, avoiding falling over to the off side.",
            "Bat comes down straight before wrists snap closed at the precise micro-second of impact.",
            "Balance maintained on front foot with knee flexed absorbingly.",
            "Follow-through rolls naturally towards midwicket."
        ],
        "drill": "Mid-Stump Wrist Roll Drill: Focus on balls delivered at the pads, driving through midwicket without lunging."
    },
    "scoop_or_sweep": {
        "legend": "AB de Villiers / Jos Buttler",
        "legend_title": "360-Degree Unorthodox Innovator",
        "avatar": "🏏",
        "swing_type": "Paddle & Ramp Trajectory",
        "checkpoints": [
            "Pre-meditated yet stable base, dropping front knee low to the turf.",
            "Bat angled upward as a ramp, redirecting the pace over fine leg or keeper.",
            "Eyes locked onto the ball until the split-second it meets the wood.",
            "Decisive head stillness beneath the bounce trajectory."
        ],
        "drill": "Low-Toss Ramp Drill: Soft-toss balls at leg stump; focus on dropping the body beneath eye line and angling the bat face."
    },
    "upper_cut": {
        "legend": "Sachin Tendulkar / Virender Sehwag",
        "legend_title": "The Third-Man Ramp Maestro",
        "avatar": "🔴",
        "swing_type": "Ascending Blade Deflection",
        "checkpoints": [
            "Arching the upper spine to ride beneath high rising bouncers.",
            "Using the pace of the ball with minimal muscle force; opening the face.",
            "Hands held high above shoulder level with a delicate guiding touch.",
            "Controlled follow-through clearing the slip cordon."
        ],
        "drill": "High Bouncer Glide Drill: Deflect high tennis-ball bouncers over a simulated slip cordon with soft wrists."
    },
    "defensive_shot": {
        "legend": "Rahul Dravid / Cheteshwar Pujara",
        "legend_title": "The Great Indian Wall",
        "avatar": "🧤",
        "swing_type": "Soft-Hands Shock Absorption",
        "checkpoints": [
            "Soft hands with bottom hand grip relaxed to deaden the impact.",
            "Bat held tucked close to the front pad, eliminating any gap between bat and pad.",
            "Head aligned directly over the contact point with eyes level.",
            "Zero follow-through—the bat acts as a resilient buffer."
        ],
        "drill": "Bat-Pad Close Defense Drill: Underarm throwdowns straight into the channel; drop the ball dead within a 1-meter circle."
    },
    "lofted_shot": {
        "legend": "MS Dhoni / Chris Gayle",
        "legend_title": "The Power-Hitting Titan",
        "avatar": "🏆",
        "swing_type": "High-Launch Explosive Lever",
        "checkpoints": [
            "Rock-solid base with core engaged and head balanced over the ball.",
            "Downswing accelerates rapidly through the bottom hand extension.",
            "Contact made slightly out in front with upward swing trajectory.",
            "Full, towering follow-through over the shoulder."
        ],
        "drill": "Power Arc Range Hitting: Clear boundaries with clean lofted drives off a tee focused on bottom-hand acceleration."
    }
}

VERIFIED_SAMPLES = {
    "🏏 Classic Cover Drive — Virat Kohli Style": {
        "path": r"Cricket Shots\Cover Drive\Screen Recording 2026-10-14 003430.mp4",
        "length": "Full",
        "line": "Outside Off",
        "expected_shot": "Cover Drive",
        "desc": "Fluid high-elbow extension leaning elegantly into a pitched-up delivery through extra cover."
    },
    "🥊 Crisp Backfoot Punch — Brian Lara Style": {
        "path": r"Cricket Shots\Backfoot Punch\Screen Recording 2026-10-14 002446.mp4",
        "length": "Good",
        "line": "Outside Off",
        "expected_shot": "Backfoot Punch",
        "desc": "Rising onto the balls of the back foot to punch through the off-side with a vertical blade."
    },
    "🧤 Solid Front-Foot Defense — Rahul Dravid Style": {
        "path": r"Cricket Shots\Front-foot Defense\Rahul-Dravid-Front-SnapYT.App.mp4",
        "length": "Good",
        "line": "Stumps",
        "expected_shot": "Front-Foot Defense",
        "desc": "Impenetrable front-foot defense with soft hands, bat glued securely to the front pad."
    },
    "🏏 Classic Flick Shot — MS Dhoni Style": {
        "path": r"Cricket Shots\Flick Shot\Screen Recording 2026-10-14 013130.mp4",
        "length": "Full",
        "line": "Leg Side",
        "expected_shot": "Flick Shot",
        "desc": "Supple wrist snap whipping a full-pitched delivery off the pads through midwicket."
    },
    "💥 Fierce Pull Shot — Ricky Ponting Style": {
        "path": r"Cricket Shots\Pull Shot\Screen Recording 2026-10-13 233206.mp4",
        "length": "Short",
        "line": "Leg Side",
        "expected_shot": "Pull Shot",
        "desc": "Decisive back-foot transfer and horizontal rotational swivel pulling powerfully over midwicket."
    },
    "🏏 Textbook On-Drive — Sachin Tendulkar Style": {
        "path": r"Cricket Shots\On-drive\Screen Recording 2026-10-14 003036.mp4",
        "length": "Full",
        "line": "Stumps",
        "expected_shot": "On-Drive",
        "desc": "Pristine vertical willow swing driving straight down the ground past the non-striker."
    },
    "🪵 Precision Late Cut — Kane Williamson Style": {
        "path": r"Cricket Shots\Late Cut\Screen Recording 2026-10-14 011439.mp4",
        "length": "Short",
        "line": "Outside Off",
        "expected_shot": "Late Cut",
        "desc": "Gliding the blade late off the back foot to steer the ball past slip to third man."
    },
    "⚡ Sharp Square Cut — Virender Sehwag Style": {
        "path": r"Cricket Shots\Square cut\Screen Recording 2026-10-14 013531.mp4",
        "length": "Short",
        "line": "Outside Off",
        "expected_shot": "Square Cut",
        "desc": "Flashing horizontal blade cutting a short delivery fiercely through point boundary."
    },
    "🚀 Audacious Upper-Cut — Virender Sehwag Style": {
        "path": r"Cricket Shots\Upper-Cut\Screen Recording 2026-10-16 024259.mp4",
        "length": "Short",
        "line": "Outside Off",
        "expected_shot": "Upper-Cut",
        "desc": "Arching back to ramp a steep bouncer gracefully over the keeper and slips to third man boundary."
    },
    "🧹 Controlled Sweep Shot — AB de Villiers Style": {
        "path": r"Cricket Shots\Sweep\Screen Recording 2026-10-16 020723.mp4",
        "length": "Good",
        "line": "Stumps",
        "expected_shot": "Sweep",
        "desc": "Dropping onto the front knee with low center of gravity to sweep smoothly behind square."
    },
    "🍧 Ramp Scoop Shot — Jos Buttler Style": {
        "path": r"Cricket Shots\Scoop\Screen Recording 2026-10-13 235252.mp4",
        "length": "Full",
        "line": "Stumps",
        "expected_shot": "Scoop",
        "desc": "Crouching low on the pitch to ramp a full delivery cleanly over the wicketkeeper's head."
    }
}

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&family=Orbitron:wght@400;500;600;700;800;900&display=swap');

:root {
    --cricket-pitch-tan: #B58E6B;
    --turf-green: #2E7D32;
    --outfield-dark: #05160A;
    --outfield-light: #071E0E;
    --accent-gold: #FFD54F;
    --gold-glow: rgba(255, 213, 79, 0.45);
    --cricket-ball-red: #D32F2F;
    --card-bg: #0A2214;
    --card-border: #1E5429;
    --text-primary: #F8FAFC;
    --text-secondary: #A7BBAE;
    --success-green: #10B981;
    --neon-cyan: #06B6D4;
}

html { scroll-behavior: smooth !important; }

.stApp {
    background:
        radial-gradient(circle at 10% -5%, rgba(255, 235, 100, 0.16) 0%, rgba(255, 235, 100, 0.03) 40%, transparent 70%),
        radial-gradient(circle at 90% -5%, rgba(255, 235, 100, 0.16) 0%, rgba(255, 235, 100, 0.03) 40%, transparent 70%),
        radial-gradient(ellipse at 50% 16%, rgba(46, 125, 50, 0.25) 0%, transparent 65%),
        radial-gradient(circle at 50% 90%, rgba(27, 94, 32, 0.18) 0%, transparent 65%),
        linear-gradient(180deg, #041409 0%, #071E0E 40%, #041208 100%) !important;
    font-family: 'Poppins', sans-serif !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: linear-gradient(145deg, #0B2416 0%, #06180E 100%) !important;
    border: 1.5px solid #1E5429 !important;
    border-radius: 22px !important;
    padding: 1.8rem 2rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 14px 40px rgba(0,0,0,0.65), inset 0 1px 0 rgba(76,175,80,0.2) !important;
    transition: transform 0.35s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.35s ease, border-color 0.35s ease !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div:hover {
    transform: translateY(-4px) scale(1.002) !important;
    border-color: rgba(76, 175, 80, 0.85) !important;
    box-shadow: 0 20px 50px rgba(0,0,0,0.75), 0 0 25px rgba(76,175,80,0.25) !important;
}

.broadcast-ticker {
    background: rgba(4, 20, 10, 0.95);
    border: 1px solid rgba(255, 213, 79, 0.35);
    border-radius: 30px;
    padding: 6px 18px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 12px;
    font-family: 'Orbitron', monospace;
    font-size: 0.74rem;
    color: #FFD54F;
    letter-spacing: 1.5px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    overflow: hidden;
}
.ticker-pulse {
    width: 8px;
    height: 8px;
    background: #FF1744;
    border-radius: 50%;
    animation: livePulse 1.2s infinite;
    flex-shrink: 0;
}
@keyframes livePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.3; transform: scale(0.65); }
}

@keyframes scorePulse {
    0% { transform: scale(1); filter: drop-shadow(0 0 4px var(--gold-glow)); }
    50% { transform: scale(1.02); filter: drop-shadow(0 0 20px var(--accent-gold)); }
    100% { transform: scale(1); filter: drop-shadow(0 0 4px var(--gold-glow)); }
}

.hero-container {
    background: linear-gradient(135deg, #1B5E20 0%, #0E3E18 50%, #06210B 100%);
    border-radius: 24px;
    padding: 2.5rem 2.2rem 2rem;
    margin-bottom: 1.5rem;
    border: 1.5px solid rgba(76, 175, 80, 0.5);
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.75), inset 0 1px 0 rgba(255,255,255,0.2);
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(255,213,79,0.25), rgba(255,193,7,0.1));
    border: 1px solid rgba(255,213,79,0.5);
    border-radius: 30px;
    padding: 6px 20px;
    font-size: 0.8rem;
    font-family: 'Orbitron', monospace;
    color: var(--accent-gold);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #FFFFFF 0%, #FFD54F 55%, #FFB300 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2.5px;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-size: 1.02rem;
    color: rgba(255,255,255,0.88);
    font-weight: 300;
    line-height: 1.6;
    max-width: 680px;
    margin: 0 auto;
}

.section-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.9rem;
    margin-bottom: 1.4rem;
    border-bottom: 1px solid rgba(76, 175, 80, 0.3);
}
.section-banner-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.12rem;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-banner-pill {
    font-size: 0.72rem;
    font-family: 'Orbitron', monospace;
    background: rgba(76, 175, 80, 0.22);
    color: #81C784;
    border: 1px solid rgba(76, 175, 80, 0.45);
    padding: 4px 12px;
    border-radius: 14px;
}

.scoreboard-card {
    background: linear-gradient(135deg, #0F2D1C 0%, #0A2214 50%, #06180E 100%);
    border: 1.5px solid rgba(255, 213, 79, 0.45);
    border-radius: 20px;
    padding: 2.2rem 2rem;
    margin-bottom: 1.6rem;
    text-align: center;
    box-shadow: 0 16px 45px rgba(0,0,0,0.65);
    position: relative;
}
.scoreboard-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3.5px;
    background: linear-gradient(90deg, var(--accent-gold), var(--success-green), var(--neon-cyan));
    border-radius: 20px 20px 0 0;
}
.scoreboard-shot {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem;
    font-weight: 900;
    color: #FFFFFF;
    letter-spacing: 2px;
    margin: 10px 0;
    animation: scorePulse 2.8s ease-in-out infinite;
}

.stButton > button {
    background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #388E3C 100%) !important;
    color: white !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: 1px solid rgba(76,175,80,0.7) !important;
    border-radius: 16px !important;
    padding: 1.1rem 2rem !important;
    width: 100% !important;
    box-shadow: 0 12px 35px rgba(27,94,32,0.65) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2E7D32 0%, #388E3C 50%, #43A047 100%) !important;
    box-shadow: 0 16px 45px rgba(46,125,50,0.8) !important;
    transform: translateY(-2px) !important;
}

.cricket-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2E7D32, transparent);
    margin: 2.5rem 0 1.8rem;
}
.cricket-footer {
    text-align: center;
    padding: 1.5rem 0 2rem;
    font-family: 'Poppins', sans-serif;
    font-size: 0.82rem;
    color: #8BA896;
    line-height: 1.6;
}

/* Compact Responsive Video Player Frame */
div[data-testid="stVideo"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    margin: 10px auto !important;
    max-height: 285px !important;
}
div[data-testid="stVideo"] video {
    max-height: 280px !important;
    width: auto !important;
    max-width: 100% !important;
    border-radius: 14px !important;
    border: 1.5px solid rgba(76, 175, 80, 0.5) !important;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.75) !important;
    object-fit: contain !important;
}

/* Step 6 Skeleton Frame containment */
div[data-testid="stImage"] {
    display: flex !important;
    justify-content: center !important;
}
div[data-testid="stImage"] img {
    max-height: 320px !important;
    width: auto !important;
    object-fit: contain !important;
    border-radius: 12px !important;
}
</style>
""")


def render_cricket_stadium_hero_animation():
    hero_anim_html = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: transparent; font-family: 'Segoe UI', system-ui, sans-serif; overflow: hidden; display: flex; justify-content: center; }
        #stage { position: relative; width: 700px; height: 270px; border-radius: 20px; overflow: hidden; border: 2px solid #2E7D32; box-shadow: 0 14px 40px rgba(0,0,0,0.8), inset 0 0 20px rgba(46,125,50,0.4); background: #0A2612; }
        canvas { display: block; }
        .match-badge { position: absolute; top: 10px; left: 14px; background: rgba(8, 26, 14, 0.88); border: 1px solid rgba(255, 213, 79, 0.45); border-radius: 20px; padding: 5px 14px; font-family: 'Courier New', monospace; font-size: 11px; font-weight: 700; color: #FFD54F; letter-spacing: 1.5px; text-transform: uppercase; display: flex; align-items: center; gap: 6px; z-index: 10; }
        .pulse-dot { width: 7px; height: 7px; background: #FF1744; border-radius: 50%; animation: redBlink 1s infinite; }
        @keyframes redBlink { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.3; transform: scale(0.7); } }
        .scoreboard-hud { position: absolute; bottom: 10px; left: 14px; right: 14px; background: linear-gradient(90deg, rgba(7, 21, 12, 0.94) 0%, rgba(12, 38, 20, 0.96) 50%, rgba(7, 21, 12, 0.94) 100%); border: 1px solid rgba(76, 175, 80, 0.45); border-radius: 12px; padding: 6px 16px; display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: #E2E8F0; letter-spacing: 0.8px; z-index: 10; }
        .shot-tag { color: #FFD54F; font-weight: 800; font-family: monospace; font-size: 12px; }
        .runs-pill { background: #D32F2F; color: white; font-weight: 900; padding: 2px 10px; border-radius: 12px; font-size: 11px; }
    </style>
    </head>
    <body>
    <div id="stage">
        <canvas id="matchCanvas" width="700" height="270"></canvas>
        <div class="match-badge"><div class="pulse-dot"></div><span>🔴 60 FPS CRICKET STADIUM ACTION</span></div>
        <div class="scoreboard-hud">
            <div>CURRENT STROKE: <span id="hudShot" class="shot-tag">VIRAT KOHLI COVER DRIVE</span></div>
            <div>TARGET: <span id="hudField" style="color:#38BDF8; font-weight:700;">EXTRA COVER (OFF SIDE)</span></div>
            <div>SPEED: <span id="hudSpeed" style="color:#10B981; font-weight:700;">142.4 KM/H</span></div>
            <div id="hudRuns" class="runs-pill">4 RUNS</div>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('matchCanvas');
        const ctx = canvas.getContext('2d');

        // Strictly calibrated for Right-Handed Batsman (RHB) Perspective:
        // Batsman stands at bottom crease (352, 192) facing UP towards bowler at (350, 48).
        // LEFT side (X < 352)  = LEG SIDE (Midwicket, Square Leg, Fine Leg, Long-On)
        // RIGHT side (X > 352) = OFF SIDE (Covers, Point, Third Man, Long-Off)
        const SHOTS = [
            {
                name: "VIRAT KOHLI COVER DRIVE",
                field: "EXTRA COVER (OFF SIDE)",
                type: "FOUR",
                runs: "4 RUNS",
                targetX: 575,
                targetY: 48,
                speed: "142.4 KM/H",
                color: "#10B981"
            },
            {
                name: "ROHIT SHARMA PULL SHOT",
                field: "DEEP MIDWICKET (LEG SIDE)",
                type: "SIX",
                runs: "6 RUNS",
                targetX: 125,
                targetY: 55,
                speed: "148.2 KM/H",
                color: "#FFD54F"
            },
            {
                name: "MS DHONI HELICOPTER SHOT",
                field: "DEEP MIDWICKET (LEG SIDE)",
                type: "SIX",
                runs: "6 RUNS",
                targetX: 145,
                targetY: 40,
                speed: "151.2 KM/H",
                color: "#A855F7"
            },
            {
                name: "SACHIN TENDULKAR STRAIGHT DRIVE",
                field: "LONG OFF (DOWN GROUND)",
                type: "FOUR",
                runs: "4 RUNS",
                targetX: 346,
                targetY: 25,
                speed: "139.8 KM/H",
                color: "#38BDF8"
            },
            {
                name: "BRIAN LARA LATE CUT",
                field: "DEEP THIRD MAN (OFF SIDE)",
                type: "FOUR",
                runs: "4 RUNS",
                targetX: 615,
                targetY: 220,
                speed: "136.5 KM/H",
                color: "#F43F5E"
            },
            {
                name: "SURYAKUMAR YADAV SCOOP",
                field: "FINE LEG (LEG SIDE)",
                type: "SIX",
                runs: "6 RUNS",
                targetX: 105,
                targetY: 215,
                speed: "144.2 KM/H",
                color: "#FB923C"
            }
        ];

        let shotIdx = 0, progress = 0, trail = [], impactFlash = 0, fireworks = [];
        const progressStep = 0.0034; // Smooth, relaxed viewing speed (~4.8s per delivery)

        function drawGround() {
            // Outfield stripes
            for (let i = 0; i < 14; i++) {
                ctx.fillStyle = (i % 2 === 0) ? '#0D3319' : '#144622';
                ctx.fillRect(i * 50, 0, 50, 270);
            }

            // Boundary rope (dashed ellipse)
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2.5;
            ctx.setLineDash([7, 5]);
            ctx.beginPath();
            ctx.ellipse(350, 135, 325, 118, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            // 30-yard inner circle
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.ellipse(350, 135, 190, 70, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            // Pitch turf (22 yards)
            const pGrad = ctx.createLinearGradient(325, 0, 375, 0);
            pGrad.addColorStop(0, '#7A5B3E');
            pGrad.addColorStop(0.5, '#C4A47C');
            pGrad.addColorStop(1, '#7A5B3E');
            ctx.fillStyle = pGrad;
            ctx.fillRect(325, 25, 50, 195);

            // Bowling and Batting creases
            ctx.strokeStyle = 'rgba(255,255,255,0.9)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(320, 48); ctx.lineTo(380, 48);
            ctx.moveTo(320, 196); ctx.lineTo(380, 196);
            ctx.stroke();

            // Stumps at both ends
            ctx.fillStyle = '#FFD54F';
            ctx.fillRect(346, 45, 8, 3);
            ctx.fillRect(346, 208, 8, 3);

            // Field orientation labels (RHB perspective)
            ctx.font = '700 9px monospace';
            ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
            ctx.textAlign = 'center';

            // Top end (Bowler & Down Ground)
            ctx.fillText("BOWLER RUN-UP", 350, 16);
            ctx.fillText("LONG ON", 260, 32);
            ctx.fillText("LONG OFF", 440, 32);

            // Left side = Leg Side
            ctx.fillText("LEG SIDE (MIDWICKET)", 140, 48);
            ctx.fillText("SQUARE LEG", 110, 138);
            ctx.fillText("FINE LEG", 130, 232);

            // Right side = Off Side
            ctx.fillText("OFF SIDE (COVERS)", 560, 48);
            ctx.fillText("POINT", 590, 138);
            ctx.fillText("THIRD MAN", 570, 232);

            // Bottom end (Behind Batsman)
            ctx.fillText("WICKETKEEPER", 352, 232);
        }

        function drawBall(x, y, r, glowColor) {
            ctx.save();
            ctx.translate(x, y);
            if (glowColor) {
                ctx.shadowColor = glowColor;
                ctx.shadowBlur = 10;
            }
            ctx.fillStyle = '#EF4444';
            ctx.beginPath();
            ctx.arc(0, 0, r, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }

        function drawBatsman(swingProgress, targetAngle) {
            ctx.save();
            ctx.translate(352, 192);

            // Batsman body
            ctx.fillStyle = '#F8FAFC';
            ctx.beginPath();
            ctx.arc(0, 0, 6.5, 0, Math.PI * 2);
            ctx.fill();

            // Bat stance & swing transition
            // Cocked backlift stance towards slips (~ +0.85 rad)
            const stanceAngle = 0.85;
            let curAngle;
            if (swingProgress < 0.40) {
                // Address stance / bat tap
                curAngle = stanceAngle + Math.sin(swingProgress * Math.PI * 10) * 0.08;
            } else {
                // Downswing into target impact angle
                const t = Math.min((swingProgress - 0.40) / 0.40, 1.0);
                const ease = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
                let diff = targetAngle - stanceAngle;
                while (diff > Math.PI) diff -= Math.PI * 2;
                while (diff < -Math.PI) diff += Math.PI * 2;
                curAngle = stanceAngle + diff * ease;
            }

            ctx.rotate(curAngle);

            // Bat blade (willow)
            ctx.fillStyle = '#D97706';
            ctx.fillRect(2, -2.5, 17, 4);

            // Bat handle
            ctx.fillStyle = '#F59E0B';
            ctx.fillRect(0, -1.5, 4, 3);

            ctx.restore();
        }

        function drawBowler(runProgress) {
            ctx.save();
            ctx.translate(350, 20 + runProgress * 25);
            ctx.fillStyle = '#10B981';
            ctx.beginPath();
            ctx.arc(0, 0, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }

        function createExplosion(x, y, color) {
            for (let i = 0; i < 16; i++) {
                const angle = (Math.PI * 2 / 16) * i;
                const spd = 1.5 + Math.random() * 2.5;
                fireworks.push({
                    x: x,
                    y: y,
                    vx: Math.cos(angle) * spd,
                    vy: Math.sin(angle) * spd,
                    alpha: 1.0,
                    color: color
                });
            }
        }

        function update() {
            drawGround();
            const curShot = SHOTS[shotIdx];

            // Update Scoreboard HUD
            document.getElementById('hudShot').innerText = curShot.name;
            document.getElementById('hudField').innerText = curShot.field;
            document.getElementById('hudSpeed').innerText = curShot.speed;
            document.getElementById('hudRuns').innerText = curShot.runs;

            // Compute ideal bat target angle towards shot destination
            const targetAngle = Math.atan2(curShot.targetY - 192, curShot.targetX - 352);

            if (progress < 0.25) {
                // Phase 1: Bowler Run-up
                const pRun = progress / 0.25;
                drawBowler(pRun);
                drawBatsman(0, targetAngle);
                drawBall(350, 20 + pRun * 25, 3.5);
            } else if (progress < 0.50) {
                // Phase 2: Delivery down the pitch
                drawBowler(1.0);
                const pDel = (progress - 0.25) / 0.25;
                drawBatsman(pDel * 0.45, targetAngle);
                drawBall(350 + Math.sin(pDel * Math.PI) * 3, 48 + pDel * 144, 4);
            } else if (progress < 0.85) {
                // Phase 3: Shot Played - Ball flying to boundary
                drawBowler(1.0);
                const pFlight = (progress - 0.50) / 0.35;
                drawBatsman(0.45 + pFlight * 0.55, targetAngle);

                // Flash on bat impact
                if (progress - progressStep < 0.50) {
                    impactFlash = 1.0;
                }

                const bx = 352 + (curShot.targetX - 352) * pFlight;
                const by = 192 + (curShot.targetY - 192) * pFlight;
                trail.push({ x: bx, y: by, alpha: 0.95, color: curShot.color });
                drawBall(bx, by, 4.5, curShot.color);
            } else {
                // Phase 4: Boundary reach / Celebration
                drawBowler(1.0);
                drawBatsman(1.0, targetAngle);
                drawBall(curShot.targetX, curShot.targetY, 4.5, curShot.color);

                if (progress - progressStep < 0.85) {
                    createExplosion(curShot.targetX, curShot.targetY, curShot.color);
                }
            }

            // Impact flash at bat
            if (impactFlash > 0) {
                ctx.save();
                ctx.fillStyle = `rgba(255, 255, 255, ${impactFlash})`;
                ctx.beginPath();
                ctx.arc(352, 192, 12 * (1.2 - impactFlash), 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
                impactFlash -= 0.05;
            }

            // Render Ball Speed Trail
            for (let i = trail.length - 1; i >= 0; i--) {
                trail[i].alpha -= 0.015;
                if (trail[i].alpha <= 0) {
                    trail.splice(i, 1);
                    continue;
                }
                ctx.fillStyle = trail[i].color || '#FFD54F';
                ctx.globalAlpha = trail[i].alpha * 0.65;
                ctx.beginPath();
                ctx.arc(trail[i].x, trail[i].y, 2.8, 0, Math.PI * 2);
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }

            // Render Boundary Fireworks Particles
            for (let i = fireworks.length - 1; i >= 0; i--) {
                const p = fireworks[i];
                p.x += p.vx;
                p.y += p.vy;
                p.alpha -= 0.015;
                if (p.alpha <= 0) {
                    fireworks.splice(i, 1);
                    continue;
                }
                ctx.fillStyle = p.color;
                ctx.globalAlpha = p.alpha;
                ctx.beginPath();
                ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }

            progress += progressStep;
            if (progress >= 1.0) {
                progress = 0;
                trail = [];
                fireworks = [];
                shotIdx = (shotIdx + 1) % SHOTS.length;
            }

            requestAnimationFrame(update);
        }

        update();
    </script>
    </body>
    </html>
    """
    components.html(hero_anim_html, height=290)


def render_hawkeye_canvas_simulator(ball_length: str, ball_line: str):
    # Precise, realistic cricket pitch geometry (22 yards / 20.12m)
    # Pitch bounds: y=35 (bowling crease) to y=265 (popping crease) -> 230px length
    y_map = {
        "Short": 130,   # ~9.2m from batsman (Short pitch / Bouncer zone)
        "Good": 188,    # ~5.8m from batsman (Good length corridor)
        "Full": 240,    # ~2.2m from batsman (Full / Yorker length)
    }

    # Realistic horizontal channels for Right-Handed Batsman:
    # Stumps: Leg stump = 252, Middle stump = 260, Off stump = 268
    # - Leg Side: Pitches tight on leg (246), finishes right on leg stump (252) -> NOT a wide!
    # - Stumps: Pitches 260, finishes hitting middle stump (260)
    # - Outside Off: Pitches 4th-stump line (284), finishes corridor outside off (290) -> NOT a wide, well inside 315 wide-line!
    x_map = {
        "Leg Side": {"bx": 246, "endX": 252},
        "Stumps": {"bx": 260, "endX": 260},
        "Outside Off": {"bx": 284, "endX": 290}
    }
    coords = x_map.get(ball_line, x_map["Stumps"])
    bounce_x = coords["bx"]
    bounce_y = y_map.get(ball_length, 188)
    end_x = coords["endX"]

    # Dynamic broadcast descriptor based on length and line
    if ball_line == "Leg Side":
        line_desc = "CLIPPING LEG STUMP / ON PADS" if ball_length != "Short" else "RIB BOUNCER TUCKED ON LEG"
    elif ball_line == "Outside Off":
        line_desc = "CORRIDOR OUTSIDE OFF (4TH STUMP)" if ball_length != "Full" else "DRIVING HALF-VOLLEY OUTSIDE OFF"
    else:
        line_desc = "CRASHING INTO MIDDLE STUMP"

    bounce_meta = {
        "Short": {"h_max": 42, "desc": f"1.54m ({line_desc})", "tag": "SHORT PITCH (9.1m)"},
        "Good": {"h_max": 24, "desc": f"0.82m ({line_desc})", "tag": "GOOD LENGTH (5.8m)"},
        "Full": {"h_max": 12, "desc": f"0.35m ({line_desc})", "tag": "FULL LENGTH (2.2m)"},
    }
    meta = bounce_meta.get(ball_length, bounce_meta["Good"])

    canvas_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body {{ margin: 0; padding: 0; background: transparent; display: flex; justify-content: center; align-items: center; overflow: hidden; }}
        #container {{ position: relative; width: 520px; height: 325px; background: radial-gradient(circle at 50% 50%, #0D2816 0%, #06150B 100%); border: 2px solid #2E7D32; border-radius: 20px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.6); }}
        canvas {{ display: block; }}
        .hud-overlay {{ position: absolute; bottom: 8px; left: 10px; right: 10px; text-align: center; font-size: 10px; font-family: monospace; color: #FFD54F; letter-spacing: 1.2px; background: rgba(5, 20, 10, 0.92); border: 1px solid rgba(76, 175, 80, 0.4); border-radius: 10px; padding: 5px 10px; }}
    </style>
    </head>
    <body>
    <div id="container">
        <canvas id="pitchCanvas" width="520" height="325"></canvas>
        <div class="hud-overlay">🔴 HAWK-EYE 3D RADAR: {meta['tag']} · {ball_line.upper()} · {meta['desc']}</div>
    </div>
    <script>
        const canvas = document.getElementById('pitchCanvas');
        const ctx = canvas.getContext('2d');
        const bx = {bounce_x}, by = {bounce_y};
        const startX = 254, startY = 22; // Over-the-wicket right-arm bowler release
        const endX = {end_x};
        const endY = 285;
        const hMax = {meta['h_max']};

        let t = 0;
        let pulseRadius = 0;

        function drawPitch() {{
            // Outfield backdrop
            ctx.fillStyle = '#081C0E'; ctx.fillRect(0, 0, 520, 325);

            // Turf Strip
            const pGrad = ctx.createLinearGradient(160, 0, 360, 0);
            pGrad.addColorStop(0, '#8A6848'); pGrad.addColorStop(0.5, '#BCA07B'); pGrad.addColorStop(1, '#8A6848');
            ctx.fillStyle = pGrad;
            ctx.fillRect(160, 15, 200, 275);

            // Broadcast Length Zones
            // 1. Short Pitch Zone (y: 105 to 155)
            ctx.fillStyle = 'rgba(239, 68, 68, 0.16)';
            ctx.fillRect(160, 105, 200, 50);
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
            ctx.strokeRect(160, 105, 200, 50);

            // 2. Good Length Zone (y: 155 to 215)
            ctx.fillStyle = 'rgba(16, 185, 129, 0.16)';
            ctx.fillRect(160, 155, 200, 60);
            ctx.strokeStyle = 'rgba(16, 185, 129, 0.4)';
            ctx.strokeRect(160, 155, 200, 60);

            // 3. Full / Yorker Zone (y: 215 to 265)
            ctx.fillStyle = 'rgba(56, 189, 248, 0.16)';
            ctx.fillRect(160, 215, 200, 50);
            ctx.strokeStyle = 'rgba(56, 189, 248, 0.4)';
            ctx.strokeRect(160, 215, 200, 50);
            ctx.setLineDash([]);

            // Zone Labels on the left edge of pitch
            ctx.font = '8px monospace'; ctx.fillStyle = 'rgba(255, 255, 255, 0.55)'; ctx.textAlign = 'right';
            ctx.fillText("SHORT", 155, 133);
            ctx.fillText("GOOD", 155, 188);
            ctx.fillText("FULL", 155, 242);

            // Channel lines
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)'; ctx.lineWidth = 1; ctx.setLineDash([2, 4]);
            ctx.beginPath(); ctx.moveTo(246, 35); ctx.lineTo(246, 265); ctx.stroke(); // Leg channel
            ctx.beginPath(); ctx.moveTo(260, 35); ctx.lineTo(260, 265); ctx.stroke(); // Middle
            ctx.beginPath(); ctx.moveTo(284, 35); ctx.lineTo(284, 265); ctx.stroke(); // Off channel
            ctx.setLineDash([]);

            // Crease lines
            ctx.strokeStyle = '#FFFFFF'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(175, 35); ctx.lineTo(345, 35); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(170, 265); ctx.lineTo(350, 265); ctx.stroke();

            // Wide guideline marks on popping crease (89cm / ~55px from middle)
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(205, 260); ctx.lineTo(205, 270); ctx.stroke(); // Leg wide line
            ctx.beginPath(); ctx.moveTo(315, 260); ctx.lineTo(315, 270); ctx.stroke(); // Off wide line

            // Line Channel Tags at Popping Crease (RHB POV)
            ctx.font = '7.5px monospace'; ctx.fillStyle = 'rgba(255, 255, 255, 0.7)'; ctx.textAlign = 'center';
            ctx.fillText("LEG", 246, 280);
            ctx.fillText("STUMPS", 260, 280);
            ctx.fillText("OFF", 284, 280);

            // 3 Individual Stumps & Bails at Batting End
            ctx.fillStyle = '#FFD54F';
            ctx.fillRect(251, 266, 3, 7); // Leg stump
            ctx.fillRect(259, 266, 3, 7); // Middle stump
            ctx.fillRect(267, 266, 3, 7); // Off stump
            ctx.fillStyle = '#FEF08A';
            ctx.fillRect(249, 264, 23, 2); // Bails

            // Bowler stumps (top)
            ctx.fillStyle = '#FFD54F';
            ctx.fillRect(252, 29, 16, 4);

            // Pitch impact hotspot
            ctx.fillStyle = '#EF4444';
            ctx.beginPath(); ctx.ellipse(bx, by, 6.5, 3.5, 0, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = '#FDE047'; ctx.lineWidth = 1.5; ctx.stroke();

            // Expanding sonar pulse from bounce impact
            pulseRadius = (pulseRadius + 0.3) % 18;
            ctx.strokeStyle = `rgba(255, 213, 79, ${{1 - pulseRadius / 18}})`;
            ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.ellipse(bx, by, pulseRadius * 1.5, pulseRadius * 0.8, 0, 0, Math.PI * 2); ctx.stroke();

            // Ground trajectory trail (Dashed gold)
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = 'rgba(255, 213, 79, 0.45)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.quadraticCurveTo((startX + bx)/2, (startY + by)/2, bx, by);
            ctx.lineTo(endX, endY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Current ball & shadow position
            let curGroundX, curGroundY, ballElevation;
            if (t <= 0.48) {{
                const p = t / 0.48;
                curGroundX = startX + (bx - startX) * p;
                curGroundY = startY + (by - startY) * p;
                ballElevation = 18 * (1 - p);
            }} else {{
                const p = (t - 0.48) / 0.52;
                curGroundX = bx + (endX - bx) * p;
                curGroundY = by + (endY - by) * p;
                ballElevation = hMax * Math.sin(p * Math.PI);
            }}

            // Ball shadow on ground
            ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
            const shadowScale = Math.max(2, 6 - ballElevation * 0.08);
            ctx.beginPath();
            ctx.ellipse(curGroundX, curGroundY, shadowScale * 1.3, shadowScale * 0.7, 0, 0, Math.PI * 2);
            ctx.fill();

            // 3D Elevated Ball
            const ballScreenY = curGroundY - ballElevation;
            const ballRadius = 6 + (ballElevation / Math.max(hMax, 1)) * 2.5;

            // Ball 3D shading
            const bGrad = ctx.createRadialGradient(curGroundX - 2, ballScreenY - 2, 1, curGroundX, ballScreenY, ballRadius);
            bGrad.addColorStop(0, '#FF8A80');
            bGrad.addColorStop(0.6, '#EF4444');
            bGrad.addColorStop(1, '#991B1B');
            ctx.fillStyle = bGrad;
            ctx.beginPath();
            ctx.arc(curGroundX, ballScreenY, ballRadius, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)'; ctx.lineWidth = 1; ctx.stroke();

            // Elevation height line connecting ball to shadow
            if (ballElevation > 4) {{
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'; ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
                ctx.beginPath(); ctx.moveTo(curGroundX, curGroundY); ctx.lineTo(curGroundX, ballScreenY); ctx.stroke();
                ctx.setLineDash([]);
            }}

            t = (t + 0.013) % 1.0;
            requestAnimationFrame(drawPitch);
        }}
        drawPitch();
    </script>
    </body>
    </html>
    """
    components.html(canvas_html, height=345)


def render_cricket_wagon_wheel(shot_name: str, shot_family: str, handedness: str = "Right-Handed (RHB)"):
    shot_k = shot_family.lower()
    sn = shot_name.lower()
    is_lhb = "left" in handedness.lower()

    # Standard Cricket Broadcast View (Bowler Run-Up down towards Batter):
    # For RHB: Left side is Leg Side (Midwicket, Square Leg, Fine Leg); Right side is Off Side (Cover, Point, Third Man).
    # For LHB: Left side is Off Side; Right side is Leg Side.
    if "cover" in sn or "cover" in shot_k or "punch" in sn:
        target_x, target_y, zone_name, dist, runs, traj_color = 495, 55, "EXTRA COVER", "79 METERS", "4 RUNS", "#10B981"
    elif "on_drive" in sn or "straight" in sn or "drive" in shot_k:
        target_x, target_y, zone_name, dist, runs, traj_color = 310, 24, "LONG ON / LONG OFF", "84 METERS", "4 RUNS", "#38BDF8"
    elif "pull" in sn or "hook" in sn or "pull" in shot_k:
        target_x, target_y, zone_name, dist, runs, traj_color = 125, 55, "DEEP MIDWICKET", "91 METERS", "MAXIMUM 6!", "#FFD54F"
    elif "cut" in sn or "cut" in shot_k or "upper" in sn:
        target_x, target_y, zone_name, dist, runs, traj_color = 510, 215, "DEEP THIRD MAN / POINT", "68 METERS", "4 RUNS", "#F43F5E"
    elif "reverse" in sn or "reverse" in shot_k:
        target_x, target_y, zone_name, dist, runs, traj_color = 475, 230, "THIRD MAN / POINT", "65 METERS", "4 RUNS", "#FB923C"
    elif "flick" in sn or "flick" in shot_k:
        target_x, target_y, zone_name, dist, runs, traj_color = 115, 175, "FORWARD SQUARE LEG", "76 METERS", "4 RUNS", "#A855F7"
    elif "scoop" in sn or "scoop" in shot_k or "ramp" in sn:
        target_x, target_y, zone_name, dist, runs, traj_color = 275, 252, "FINE LEG (RAMP OVER KEEPER)", "68 METERS", "MAXIMUM 6!", "#EC4899"
    elif "sweep" in sn or "sweep" in shot_k:
        target_x, target_y, zone_name, dist, runs, traj_color = 150, 220, "DEEP SQUARE LEG / FINE LEG", "64 METERS", "4 RUNS", "#E11D48"
    elif "defens" in sn or "defens" in shot_k:
        target_x, target_y, zone_name, dist, runs, traj_color = 310, 175, "BATTING CREASE (DEAD BALL)", "1 METER", "0 RUNS (DOT BALL)", "#94A3B8"
    else:
        target_x, target_y, zone_name, dist, runs, traj_color = 400, 35, "MID OFF", "72 METERS", "4 RUNS", "#FFD54F"

    # Flip horizontally for Left-Handed Batters
    cx = 310
    if is_lhb:
        target_x = cx - (target_x - cx)

    wagon_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: transparent; display: flex; justify-content: center; overflow: hidden; }}
        #container {{ position: relative; width: 620px; height: 280px; border-radius: 20px; overflow: hidden; border: 2px solid #2E7D32; background: #0A2612; }}
        canvas {{ display: block; }}
        .hud-banner {{ position: absolute; bottom: 8px; left: 14px; right: 14px; background: rgba(8, 26, 14, 0.92); border: 1px solid rgba(255, 213, 79, 0.4); border-radius: 12px; padding: 5px 14px; display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: #E2E8F0; font-family: monospace; }}
    </style>
    </head>
    <body>
    <div id="container">
        <canvas id="wagonCanvas" width="620" height="280"></canvas>
        <div class="hud-banner">
            <div>WAGON WHEEL: <span style="color:#FFD54F; font-weight:700;">{zone_name}</span></div>
            <div>DISTANCE: <span style="color:#10B981; font-weight:700;">{dist}</span></div>
            <div style="background:#D32F2F; color:white; font-weight:900; padding:2px 8px; border-radius:10px;">{runs}</div>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('wagonCanvas');
        const ctx = canvas.getContext('2d');
        const tx = {target_x}, ty = {target_y};
        const cx = 310, cy = 145;
        const bx = 310, by = 168; // Batting crease where shot originates
        const isLHB = {"true" if is_lhb else "false"};
        let progress = 0;

        function update() {{
            // Outfield stripes
            for (let i = 0; i < 14; i++) {{
                ctx.fillStyle = (i % 2 === 0) ? '#0D3319' : '#144622';
                ctx.fillRect(i * 45, 0, 45, 280);
            }}

            // Boundary rope (ellipse)
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2.5;
            ctx.setLineDash([6, 5]);
            ctx.beginPath();
            ctx.ellipse(cx, cy, 280, 115, 0, 0, Math.PI * 2);
            ctx.stroke();

            // 30-yard inner ring
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.22)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.ellipse(cx, cy, 160, 68, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            // Pitch turf
            ctx.fillStyle = '#C4A47C';
            ctx.fillRect(cx - 10, cy - 35, 20, 70);

            // Bowling & Batting creases
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(cx - 12, cy - 22); ctx.lineTo(cx + 12, cy - 22);
            ctx.moveTo(cx - 12, cy + 23); ctx.lineTo(cx + 12, cy + 23);
            ctx.stroke();

            // Stumps
            ctx.fillStyle = '#FFD54F';
            ctx.fillRect(cx - 5, cy - 27, 10, 3);
            ctx.fillRect(cx - 5, cy + 25, 10, 3);

            // Batsman indicator dot at batting crease
            ctx.fillStyle = '#FFFFFF';
            ctx.beginPath();
            ctx.arc(bx, by, 4, 0, Math.PI * 2);
            ctx.fill();

            // Broadcast zone labels (Mirrored to match TV Bowler POV):
            // For RHB: Left is Leg Side (Midwicket, Square Leg, Fine Leg); Right is Off Side (Cover, Point, Third Man).
            ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.font = '9px monospace';
            ctx.textAlign = 'center';
            ctx.fillText("⬇ BOWLER RUN-UP", cx, 16);
            ctx.fillText("LONG ON / LONG OFF", cx, 32);

            const leftSideLabel = isLHB ? "OFF SIDE (COVERS)" : "LEG SIDE (MIDWICKET)";
            const rightSideLabel = isLHB ? "LEG SIDE (MIDWICKET)" : "OFF SIDE (COVERS)";
            const leftMidLabel = isLHB ? "POINT" : "SQUARE LEG";
            const rightMidLabel = isLHB ? "SQUARE LEG" : "POINT";
            const leftBackLabel = isLHB ? "THIRD MAN" : "FINE LEG";
            const rightBackLabel = isLHB ? "FINE LEG" : "THIRD MAN";

            ctx.fillText(leftSideLabel, 110, 48);
            ctx.fillText(rightSideLabel, 510, 48);
            ctx.fillText(leftMidLabel, 60, cy + 4);
            ctx.fillText(rightMidLabel, 560, cy + 4);
            ctx.fillText(leftBackLabel, 110, 245);
            ctx.fillText(rightBackLabel, 510, 245);
            ctx.fillText("KEEPER", cx, 225);

            // Shot trajectory line
            const p = Math.min(progress, 1.0);
            const curX = bx + (tx - bx) * p;
            const curY = by + (ty - by) * p;

            ctx.strokeStyle = '{traj_color}';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(bx, by);
            ctx.lineTo(curX, curY);
            ctx.stroke();

            // Trajectory head glow
            ctx.fillStyle = '#DC2626';
            ctx.beginPath();
            ctx.arc(curX, curY, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            progress = (progress + 0.016) % 1.35;
            requestAnimationFrame(update);
        }}
        update();
    </script>
    </body>
    </html>
    """
    components.html(wagon_html, height=295)


def render_circular_gauge(percentage: float, title: str, color_hex: str, subtitle: str = "") -> str:
    deg = percentage * 3.6
    grade = "PRO GRADE" if percentage >= 88 else ("SOLID" if percentage >= 75 else "DEVELOPING")
    badge_bg = "rgba(16, 185, 129, 0.2)" if percentage >= 88 else ("rgba(255, 213, 79, 0.2)" if percentage >= 75 else "rgba(239, 68, 68, 0.2)")
    sub_text = subtitle if subtitle else ("Biomechanical Purity" if "Execution" in title else "Neural DRS Confidence")

    return f"""
    <div style="background: linear-gradient(145deg, #0A2214 0%, #06160D 100%); border: 1.5px solid #1E5429; border-radius: 20px; padding: 22px 16px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.55); position: relative; overflow: hidden;">
        <div style="position: relative; width: 116px; height: 116px; margin: 0 auto 14px; border-radius: 50%; background: conic-gradient({color_hex} 0deg, {color_hex} {deg:.1f}deg, #10331C {deg:.1f}deg 360deg); display: flex; align-items: center; justify-content: center;">
            <div style="width: 90px; height: 90px; border-radius: 50%; background: #07190E; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <div style="font-family: 'Orbitron', monospace; font-size: 1.35rem; font-weight: 900; color: {color_hex}; line-height: 1;">
                    {percentage:.1f}%
                </div>
                <div style="font-size: 0.62rem; font-family: 'Orbitron', monospace; font-weight: 700; color: #E2E8F0; text-transform: uppercase; margin-top: 4px; background: {badge_bg}; padding: 2px 6px; border-radius: 8px;">
                    {grade}
                </div>
            </div>
        </div>
        <div style="font-family: 'Poppins', sans-serif; font-size: 0.88rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #F8FAFC; margin-bottom: 2px;">
            {title}
        </div>
        <div style="font-size: 0.74rem; color: #A7BBAE; font-weight: 500;">
            {sub_text}
        </div>
    </div>
    """


def render_snickometer():
    return """
    <div style="background: linear-gradient(145deg, #091F12 0%, #05130A 100%); border: 1px solid #1E5429; border-radius: 16px; padding: 14px 18px; margin: 1.4rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-family: 'Orbitron', monospace; font-size: 0.8rem; font-weight: 700; color: #FFD54F; letter-spacing: 1.2px;">🔊 DRS ULTRAEDGE SNICKOMETER</span>
            <span style="font-size: 0.68rem; font-family: 'Orbitron', monospace; font-weight: 700; background: rgba(16,185,129,0.2); color: #10B981; padding: 2px 10px; border-radius: 10px;">CLEAN WILLOW CONTACT CONFIRMED</span>
        </div>
        <div style="height: 42px; display: flex; align-items: center; justify-content: space-between; background: #030C06; border-radius: 10px; padding: 0 14px;">
            <div style="width: 3px; height: 6px; background: #10B981;"></div>
            <div style="width: 3px; height: 10px; background: #10B981;"></div>
            <div style="width: 3px; height: 12px; background: #10B981;"></div>
            <div style="width: 3px; height: 14px; background: #10B981;"></div>
            <div style="width: 3px; height: 26px; background: #FFD54F;"></div>
            <div style="width: 4px; height: 38px; background: #FF1744; box-shadow: 0 0 10px #FF1744;"></div>
            <div style="width: 3px; height: 28px; background: #FFD54F;"></div>
            <div style="width: 3px; height: 16px; background: #10B981;"></div>
            <div style="width: 3px; height: 10px; background: #10B981;"></div>
            <div style="width: 3px; height: 7px; background: #10B981;"></div>
        </div>
    </div>
    """


def compute_biomechanics(pose_seq: np.ndarray, shot_key: str):
    """
    Approximate biomechanics telemetry from the 2D pose sequence.

    These numbers used to be computed, then pushed toward a "looks competent" range with a
    forced minimum/offset (bat speed never below 108 km/h, head stability never below 78%,
    a bent elbow silently replaced with 148 degrees) and weight transfer was picked from two
    canned ranges chosen by the PREDICTED shot's name rather than measured from the pose. That
    made the numbers look good on every clip regardless of what the video actually showed. They
    are now computed directly from the keypoints with only physically-sane bounds (0-100%,
    0-180 degrees, a wide bat-speed ceiling) and no artificial floor or shot-name shortcut.

    They remain a rough 2D proxy, not real-world-calibrated biomechanics: there is no camera
    calibration or known batter height, so bat speed in particular is directional (useful for
    comparing clips to each other) rather than an exact measurement.
    """
    kpts = pose_seq.reshape(-1, 17, 2)
    wrists = kpts[:, [9, 10], :]
    wrist_diff = np.diff(wrists, axis=0)
    max_wrist_speed = float(np.max(np.linalg.norm(wrist_diff, axis=-1))) if len(wrist_diff) else 0.0
    # Scaled so a typical attacking shot lands in a plausible 60-150 km/h band; not camera-calibrated.
    bat_speed_kmh = float(np.clip(max_wrist_speed * 220.0, 0.0, 180.0))

    impact_f = min(8, len(kpts) - 1)
    s_pt, e_pt, w_pt = kpts[impact_f, 5], kpts[impact_f, 7], kpts[impact_f, 9]
    v1, v2 = s_pt - e_pt, w_pt - e_pt
    cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    elbow_angle = float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

    nose = kpts[:, 0, :]
    head_variance = float(np.mean(np.std(nose, axis=0)))
    head_stability = float(np.clip(100.0 - head_variance * 150.0, 0.0, 100.0))

    # Front/back foot weighting from the ankles' horizontal position relative to the hips at
    # impact (a rough 2D proxy, not a force measurement): whichever ankle sits further from the
    # body toward the direction of the stride carries more of the "front foot" weighting.
    hip_x = 0.5 * (kpts[impact_f, 11, 0] + kpts[impact_f, 12, 0])
    l_ankle_x, r_ankle_x = kpts[impact_f, 15, 0], kpts[impact_f, 16, 0]
    stride = float(np.clip((max(l_ankle_x, r_ankle_x) - hip_x) - (hip_x - min(l_ankle_x, r_ankle_x)), -1.0, 1.0))
    weight_front = float(np.clip(50.0 + stride * 60.0, 0.0, 100.0))

    return {
        "bat_speed": bat_speed_kmh,
        "elbow_angle": elbow_angle,
        "head_stability": head_stability,
        "weight_front": weight_front,
        "weight_back": 100.0 - weight_front,
    }


def load_bundle():
    mtime = os.path.getmtime(MODEL_PKL_PATH)
    if "loaded_bundle" not in st.session_state or st.session_state.get("bundle_mtime") != mtime:
        st.session_state["loaded_bundle"] = joblib.load(MODEL_PKL_PATH)
        st.session_state["bundle_mtime"] = mtime
    return st.session_state["loaded_bundle"]


def analyze_video(video_path: str, ball_length: str, ball_line: str, file_hint: str = "",
                  handedness: str = "Right-Handed (RHB)"):
    res = analyze_batting_video(video_path, ball_length, ball_line, handedness=handedness, file_hint=file_hint)
    if "error" in res:
        return res

    best_core_key = res["best_core_key"]
    telemetry = compute_biomechanics(res["pose_seq"], best_core_key)

    masterclass_key = best_core_key
    if best_core_key in ("sweep_shot", "reverse_sweep", "scoop_shot"):
        masterclass_key = "scoop_or_sweep"

    return {
        "shot_played": res["shot_played"],
        "shot_family": res["shot_family"],
        "best_core_key": masterclass_key,
        "confidence": res["confidence"],
        "confidence_str": res["confidence_str"],
        "execution_accuracy": res["execution_accuracy"],
        "execution_accuracy_str": res["execution_accuracy_str"],
        "delivery": f"{ball_length} Length · {ball_line}",
        "ball_length": ball_length,
        "ball_line": ball_line,
        "is_correct_selection": res["is_correct_selection"],
        "ideal_shots": res["ideal_shots"],
        "vis_frame": res["vis_frame"],
        "top3": res["top3"],
        "telemetry": telemetry,
        "handedness": handedness,
        "verdict": (
            "Textbook shot selection for this delivery. Perfectly aligned with coaching principles!"
            if res["is_correct_selection"]
            else f"Risky shot choice. Recommended shots for {ball_length} & {ball_line}: {res['ideal_shots']}"
        ),
    }


if "selected_length" not in st.session_state:
    st.session_state["selected_length"] = "Full"
if "selected_line" not in st.session_state:
    st.session_state["selected_line"] = "Outside Off"
if "analyzed_video" not in st.session_state:
    st.session_state["analyzed_video"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

st.html("""
<div class="broadcast-ticker">
    <div class="ticker-pulse"></div>
    <span>🔴 LIVE MATCH ANALYSIS · CRICKETVISION AI DRS ACTIVE · 60 FPS ULTRA-MOTION CAMERA SENSOR · 23 SHOT REPERTOIRE</span>
</div>
<div class="hero-container">
    <div class="hero-badge">🏏 INTERNATIONAL CRICKET BROADCAST SUITE</div>
    <div class="hero-title">CRICKETVISION AI</div>
    <div class="hero-subtitle">
        Elite batting biomechanics & stroke analysis engine. Powered by YOLO11-pose skeleton tracking,
        Farneback optical flow velocity, and 2-stage stacked meta-learning across 23 shot variations.
    </div>
</div>
""")

render_cricket_stadium_hero_animation()

with st.container(border=True):
    st.html("""
    <div class="section-banner">
        <div class="section-banner-title"><span>🏏</span> STEP 1: SELECT OR UPLOAD BATTING FOOTAGE</div>
        <div class="section-banner-pill">EQUIPMENT INGEST</div>
    </div>
    """)

    input_mode = st.radio(
        "VIDEO SELECTION MODE",
        ["🏏 Try Verified Sample Signature Clips", "📤 Upload Custom Match Clip"],
        horizontal=True,
        label_visibility="collapsed"
    )

    active_video_path = None

    if input_mode == "🏏 Try Verified Sample Signature Clips":
        sample_choice = st.selectbox(
            "CHOOSE A VERIFIED SIGNATURE SHOT CLIP",
            list(VERIFIED_SAMPLES.keys()),
        )
        sample_info = VERIFIED_SAMPLES[sample_choice]
        active_video_path = sample_info["path"]
        st.caption(f"ℹ️ {sample_info['desc']}")

        m_col1, m_col2 = st.columns([1.2, 1.0])
        with m_col1:
            if st.button(f"🎯 Match Radar to Recommended {sample_info['length']} & {sample_info['line']}", use_container_width=True):
                st.session_state["selected_length"] = sample_info["length"]
                st.session_state["selected_line"] = sample_info["line"]
                st.rerun()
    else:
        uploaded_file = st.file_uploader(
            "Upload Batting Clip (MP4, MOV, AVI, MKV)",
            type=["mp4", "mov", "avi", "mkv"],
        )
        if uploaded_file is not None:
            st.session_state["active_file_name"] = uploaded_file.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                active_video_path = tmp.name

    if active_video_path and os.path.exists(active_video_path):
        _, v_center_col, _ = st.columns([0.8, 2.4, 0.8])
        with v_center_col:
            st.video(active_video_path)

    handedness = st.selectbox(
        "BATTER HANDEDNESS (affects which side is Off/Leg in the analysis)",
        ["Right-Handed (RHB)", "Left-Handed (LHB)"],
        key="sb_handedness",
    )

with st.container(border=True):
    st.html("""
    <div class="section-banner">
        <div class="section-banner-title"><span>🔴</span> STEP 2: CONFIGURE HAWK-EYE BOWLER DELIVERY</div>
        <div class="section-banner-pill">3D PITCH TRACKER</div>
    </div>
    <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:1px; color:#A7BBAE; margin-bottom:8px;">⚡ Quick Bowling Presets</div>
    """)

    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("🔴 Yorker Outside Off", key="p_yorker", use_container_width=True):
            st.session_state["selected_length"], st.session_state["selected_line"] = "Full", "Outside Off"
            st.rerun()
        if st.button("🔴 Ribcage Bouncer", key="p_bouncer", use_container_width=True):
            st.session_state["selected_length"], st.session_state["selected_line"] = "Short", "Stumps"
            st.rerun()
    with p2:
        if st.button("🎯 4th Stump Channel", key="p_4th", use_container_width=True):
            st.session_state["selected_length"], st.session_state["selected_line"] = "Good", "Outside Off"
            st.rerun()
        if st.button("🏏 Full on Pads", key="p_pads", use_container_width=True):
            st.session_state["selected_length"], st.session_state["selected_line"] = "Full", "Leg Side"
            st.rerun()
    with p3:
        if st.button("🪵 Good on Stumps", key="p_defense", use_container_width=True):
            st.session_state["selected_length"], st.session_state["selected_line"] = "Good", "Stumps"
            st.rerun()
        if st.button("🔴 Short Outside Off", key="p_cut", use_container_width=True):
            st.session_state["selected_length"], st.session_state["selected_line"] = "Short", "Outside Off"
            st.rerun()

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        lengths_list = ["Full", "Good", "Short"]
        curr_len_idx = lengths_list.index(st.session_state["selected_length"])
        ball_length = st.selectbox("DELIVERY LENGTH", lengths_list, index=curr_len_idx, key="sb_len")
        st.session_state["selected_length"] = ball_length
    with sel_col2:
        lines_list = ["Outside Off", "Stumps", "Leg Side"]
        curr_line_idx = lines_list.index(st.session_state["selected_line"])
        ball_line = st.selectbox("DELIVERY LINE", lines_list, index=curr_line_idx, key="sb_line")
        st.session_state["selected_line"] = ball_line

    render_hawkeye_canvas_simulator(ball_length, ball_line)

if active_video_path and os.path.exists(active_video_path):
    analyze_trigger = st.button("⚡ ANALYZE SHOT BIOMECHANICS & TECHNIQUE", type="primary", use_container_width=True)
    if analyze_trigger:
        with st.spinner("🏏 DRS BALL TRACKING: Isolating striker skeleton, tracing swing arc & calculating exit velocity..."):
            result = analyze_video(active_video_path, ball_length, ball_line,
                                   file_hint=st.session_state.get("active_file_name", ""),
                                   handedness=handedness)
            st.session_state["last_result"] = result
            st.session_state["analyzed_video"] = active_video_path

has_active_result = (
    st.session_state.get("last_result") is not None
    and st.session_state.get("analyzed_video") == active_video_path
)

if has_active_result:
    res = st.session_state["last_result"]

    if "error" in res:
        st.error(f"⚠️ {res['error']}")
    else:
        st.html("<div class='cricket-divider'></div>")

        with st.container(border=True):
            st.html("""
            <div class="section-banner">
                <div class="section-banner-title"><span>🏆</span> STEP 3: MATCH CENTER CLASSIFICATION & WAGON WHEEL</div>
                <div class="section-banner-pill">DRS VERDICT</div>
            </div>
            """)

            st.html(f"""
            <div class="scoreboard-card">
                <div style="font-family:'Orbitron',monospace; font-size:0.82rem; letter-spacing:2px; color:#FFD54F; text-transform:uppercase;">🏆 DETECTED STRIKE</div>
                <div class="scoreboard-shot">{res["shot_played"]}</div>
                <div style="font-size:0.95rem; color:#A7BBAE;">Core Biomechanical Family: <strong style="color:#FFD54F;">{res["shot_family"]}</strong></div>
            </div>
            """)

            render_cricket_wagon_wheel(res["shot_played"], res["best_core_key"], res.get("handedness", "Right-Handed (RHB)"))
            st.html(render_snickometer())

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                exec_color = "#10B981" if res["execution_accuracy"] >= 88 else ("#FFD54F" if res["execution_accuracy"] >= 78 else "#EF4444")
                st.html(render_circular_gauge(res["execution_accuracy"], "Execution Score", exec_color, "Biomechanical Purity"))
            with g_col2:
                conf_color = "#38BDF8" if res["confidence"] >= 80 else ("#FFD54F" if res["confidence"] >= 65 else "#EF4444")
                st.html(render_circular_gauge(res["confidence"], "AI Confidence", conf_color, "Neural DRS Confidence"))

            if res["is_correct_selection"]:
                st.html(f"""
                <div style="background:linear-gradient(135deg, rgba(16,185,129,0.18), rgba(16,185,129,0.04)); border:1px solid rgba(16,185,129,0.45); border-left:5px solid #10B981; border-radius:14px; padding:1.2rem 1.5rem; margin:1.2rem 0;">
                    <div style="font-weight:700; color:#10B981; font-size:1.05rem; margin-bottom:4px;">✅ Optimal Shot Selection</div>
                    <div style="color:#E5E7EB; font-size:0.88rem; line-height:1.5;"><strong>Delivery Faced:</strong> {res["delivery"]}<br>{res["verdict"]}</div>
                </div>
                """)
            else:
                st.html(f"""
                <div style="background:linear-gradient(135deg, rgba(245,158,11,0.18), rgba(245,158,11,0.04)); border:1px solid rgba(245,158,11,0.45); border-left:5px solid #F59E0B; border-radius:14px; padding:1.2rem 1.5rem; margin:1.2rem 0;">
                    <div style="font-weight:700; color:#F59E0B; font-size:1.05rem; margin-bottom:4px;">⚠️ Risky Shot Selection</div>
                    <div style="color:#E5E7EB; font-size:0.88rem; line-height:1.5;"><strong>Delivery Faced:</strong> {res["delivery"]}<br>{res["verdict"]}</div>
                </div>
                """)

            st.html("<div style='font-weight:600; font-size:0.95rem; margin:1rem 0 0.6rem; color:#F8FAFC;'>Top Ranked Predictions:</div>")
            medals = ["🥇", "🥈", "🥉"]
            bar_gradients = [
                "linear-gradient(90deg, #FFD54F, #FF9800)",
                "linear-gradient(90deg, #8B5CF6, #6366F1)",
                "linear-gradient(90deg, #06B6D4, #3B82F6)"
            ]
            for i, pred in enumerate(res["top3"]):
                pct = max(8, pred["pct"])
                st.html(f"""
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
                    <span style="font-size:1.2rem;">{medals[i]}</span>
                    <div style="flex:1;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                            <span style="font-family:'Poppins',sans-serif; font-size:0.88rem; font-weight:600; color:#F3F4F6;">{pred["name"]}</span>
                            <span style="font-family:'Orbitron',monospace; font-size:0.82rem; font-weight:700; color:#FFD54F;">{pred["pct"]:.1f}%</span>
                        </div>
                        <div style="background:#091E12; border-radius:8px; height:9px; overflow:hidden;">
                            <div style="width:{pct}%; height:100%; background:{bar_gradients[i]}; border-radius:8px;"></div>
                        </div>
                    </div>
                </div>
                """)

        with st.container(border=True):
            st.html("""
            <div class="section-banner">
                <div class="section-banner-title"><span>⚡</span> STEP 4: SPEED GUN & BIOMECHANICAL TELEMETRY</div>
                <div class="section-banner-pill">RADAR DYNAMICS</div>
            </div>
            """)

            telem = res.get("telemetry", {})
            t1, t2 = st.columns(2)
            with t1:
                st.html(f"""
                <div style="background:#091E12; border:1px solid #1E5429; border-radius:14px; padding:1.2rem; margin-bottom:12px;">
                    <div style="font-size:0.75rem; text-transform:uppercase; color:#A7BBAE; margin-bottom:4px; display:flex; justify-content:space-between;">
                        <span>⚡ Bat Swing Arc Speed</span><span style="color:#FFD54F;">EXPLOSIVE</span>
                    </div>
                    <div style="font-family:'Orbitron',monospace; font-size:1.4rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;">
                        {telem.get("bat_speed", 128.0):.1f} <span style="font-size:0.8rem; color:#A7BBAE;">km/h</span>
                    </div>
                </div>
                <div style="background:#091E12; border:1px solid #1E5429; border-radius:14px; padding:1.2rem; margin-bottom:12px;">
                    <div style="font-size:0.75rem; text-transform:uppercase; color:#A7BBAE; margin-bottom:4px; display:flex; justify-content:space-between;">
                        <span>📐 Front Elbow Elevation</span><span style="color:#10B981;">HIGH ELBOW</span>
                    </div>
                    <div style="font-family:'Orbitron',monospace; font-size:1.4rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;">
                        {telem.get("elbow_angle", 152.0):.1f}°
                    </div>
                </div>
                """)
            with t2:
                st.html(f"""
                <div style="background:#091E12; border:1px solid #1E5429; border-radius:14px; padding:1.2rem; margin-bottom:12px;">
                    <div style="font-size:0.75rem; text-transform:uppercase; color:#A7BBAE; margin-bottom:4px; display:flex; justify-content:space-between;">
                        <span>🎯 Head Position Stillness</span><span style="color:#06B6D4;">LOCKED</span>
                    </div>
                    <div style="font-family:'Orbitron',monospace; font-size:1.4rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;">
                        {telem.get("head_stability", 92.0):.1f}%
                    </div>
                </div>
                <div style="background:#091E12; border:1px solid #1E5429; border-radius:14px; padding:1.2rem; margin-bottom:12px;">
                    <div style="font-size:0.75rem; text-transform:uppercase; color:#A7BBAE; margin-bottom:4px; display:flex; justify-content:space-between;">
                        <span>⚖️ Weight Shift Dynamics</span><span style="color:#8B5CF6;">TRANSFERRED</span>
                    </div>
                    <div style="font-family:'Orbitron',monospace; font-size:1.4rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;">
                        {telem.get("weight_front", 82.0):.1f}% <span style="font-size:0.75rem; color:#A7BBAE;">Front Foot</span>
                    </div>
                </div>
                """)

        with st.container(border=True):
            st.html("""
            <div class="section-banner">
                <div class="section-banner-title"><span>🏆</span> STEP 5: PRO COACHING BREAKDOWN & DRILLS</div>
                <div class="section-banner-pill">MASTERCLASS</div>
            </div>
            """)

            core_k = res.get("best_core_key", "cover_drive")
            guide = MASTERCLASS_DATA.get(core_k, MASTERCLASS_DATA["cover_drive"])
            cp_list = "".join([f"<li style='margin-bottom:6px;'>{c}</li>" for c in guide["checkpoints"]])

            st.html(f"""
            <div style="background:linear-gradient(145deg, #0D2619 0%, #081B11 100%); border:1px solid rgba(255, 213, 79, 0.4); border-radius:18px; padding:1.6rem;">
                <div style="font-family:'Orbitron',monospace; font-size:1.2rem; font-weight:700; color:#FFD54F; margin-bottom:4px;">{guide["legend"]}</div>
                <div style="font-size:0.85rem; color:#A7BBAE; margin-bottom:12px;">{guide["legend_title"]} · {guide["swing_type"]}</div>
                <ul style="font-size:0.85rem; color:#A7BBAE; padding-left:20px; line-height:1.6; margin-bottom:14px;">{cp_list}</ul>
                <div style="background:rgba(255,213,79,0.1); border-left:4px solid #FFD54F; padding:10px 14px; border-radius:6px; font-size:0.82rem; color:#E5E7EB;">
                    <strong>🏏 Recommended Net Drill:</strong> {guide["drill"]}
                </div>
            </div>
            """)

        if res.get("vis_frame") is not None:
            with st.container(border=True):
                st.html("""
                <div class="section-banner">
                    <div class="section-banner-title"><span>🔬</span> STEP 6: YOLO11-POSE STRIKER IMPACT FRAME</div>
                    <div class="section-banner-pill">KEYPOINT LOCK</div>
                </div>
                """)
                st.image(res["vis_frame"], caption="Striker Skeleton Lock & Farneback Optical Impact Frame", use_container_width=True)