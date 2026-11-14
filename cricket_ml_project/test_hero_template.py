import streamlit as st
import streamlit.components.v1 as components

def get_cricket_stadium_hero_html():
    return """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: transparent;
            font-family: 'Segoe UI', system-ui, sans-serif;
            overflow: hidden;
            display: flex;
            justify-content: center;
        }
        #stage {
            position: relative;
            width: 700px;
            height: 270px;
            border-radius: 20px;
            overflow: hidden;
            border: 2px solid #2E7D32;
            box-shadow: 0 14px 40px rgba(0,0,0,0.8), inset 0 0 20px rgba(46,125,50,0.4);
            background: #0A2612;
        }
        canvas { display: block; }
        .match-badge {
            position: absolute;
            top: 10px;
            left: 14px;
            background: rgba(8, 26, 14, 0.88);
            border: 1px solid rgba(255, 213, 79, 0.45);
            border-radius: 20px;
            padding: 5px 14px;
            font-family: 'Courier New', monospace;
            font-size: 11px;
            font-weight: 700;
            color: #FFD54F;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            display: flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .pulse-dot {
            width: 7px;
            height: 7px;
            background: #FF1744;
            border-radius: 50%;
            animation: redBlink 1s infinite;
        }
        @keyframes redBlink {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.3; transform: scale(0.7); }
        }
        .scoreboard-hud {
            position: absolute;
            bottom: 10px;
            left: 14px;
            right: 14px;
            background: linear-gradient(90deg, rgba(7, 21, 12, 0.92) 0%, rgba(12, 38, 20, 0.95) 50%, rgba(7, 21, 12, 0.92) 100%);
            border: 1px solid rgba(76, 175, 80, 0.4);
            border-radius: 12px;
            padding: 6px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            color: #E2E8F0;
            letter-spacing: 1px;
        }
        .shot-tag {
            color: #FFD54F;
            font-weight: 800;
            font-family: monospace;
            font-size: 12px;
        }
        .runs-pill {
            background: #D32F2F;
            color: white;
            font-weight: 900;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 11px;
        }
    </style>
    </head>
    <body>
    <div id="stage">
        <canvas id="matchCanvas" width="700" height="270"></canvas>
        <div class="match-badge">
            <div class="pulse-dot"></div>
            <span>LIVE 60 FPS CRICKET STADIUM ACTION</span>
        </div>
        <div class="scoreboard-hud">
            <div>CURRENT STROKE: <span id="hudShot" class="shot-tag">COVER DRIVE</span></div>
            <div>SPEED: <span id="hudSpeed" style="color:#10B981; font-weight:700;">143.6 KM/H</span></div>
            <div id="hudRuns" class="runs-pill">4 RUNS</div>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('matchCanvas');
        const ctx = canvas.getContext('2d');

        const SHOTS = [
            { name: "VIRAT KOHLI COVER DRIVE", field: "EXTRA COVER", type: "FOUR", runs: "4 RUNS", targetX: 130, targetY: 45, batAngle: -0.7, color: "#10B981" },
            { name: "ROHIT SHARMA PULL SHOT", field: "DEEP MIDWICKET", type: "SIX", runs: "6 RUNS", targetX: 580, targetY: 55, batAngle: 1.1, color: "#FFD54F" },
            { name: "SACHIN TENDULKAR STRAIGHT DRIVE", field: "LONG OFF", type: "FOUR", runs: "4 RUNS", targetX: 300, targetY: 28, batAngle: -0.2, color: "#38BDF8" },
            { name: "BRIAN LARA LATE CUT", field: "DEEP THIRD MAN", type: "FOUR", runs: "4 RUNS", targetX: 95, targetY: 215, batAngle: -1.3, color: "#F43F5E" },
            { name: "MS DHONI HELICOPTER FLICK", field: "FORWARD SQUARE LEG", type: "SIX", runs: "6 RUNS", targetX: 590, targetY: 175, batAngle: 0.9, color: "#A855F7" }
        ];

        let shotIdx = 0;
        let progress = 0; // 0 to 1
        let dustParticles = [];
        let trail = [];
        let fireworks = [];
        let state = "runup"; // runup, bowl, strike, flight, boundary

        function drawGround() {
            // Mown Grass Turf with 10 alternating lawn stripes
            for (let i = 0; i < 14; i++) {
                ctx.fillStyle = (i % 2 === 0) ? '#0D3319' : '#144622';
                ctx.fillRect(i * 50, 0, 50, 270);
            }

            // Dual Stadium Floodlights
            const gLeft = ctx.createRadialGradient(50, 15, 10, 150, 100, 260);
            gLeft.addColorStop(0, 'rgba(255, 240, 120, 0.32)');
            gLeft.addColorStop(1, 'transparent');
            ctx.fillStyle = gLeft;
            ctx.fillRect(0, 0, 700, 270);

            const gRight = ctx.createRadialGradient(650, 15, 10, 550, 100, 260);
            gRight.addColorStop(0, 'rgba(255, 240, 120, 0.32)');
            gRight.addColorStop(1, 'transparent');
            ctx.fillStyle = gRight;
            ctx.fillRect(0, 0, 700, 270);

            // Stadium Boundary Rope (White Dashed Arc)
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2.5;
            ctx.setLineDash([7, 5]);
            ctx.beginPath();
            ctx.ellipse(350, 135, 325, 118, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            // 30-Yard Infield Ring
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.ellipse(350, 135, 190, 70, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]);

            // 22-Yard Pitch Strip
            const pGrad = ctx.createLinearGradient(325, 0, 375, 0);
            pGrad.addColorStop(0, '#7A5B3E');
            pGrad.addColorStop(0.2, '#A6845B');
            pGrad.addColorStop(0.5, '#C4A47C');
            pGrad.addColorStop(0.8, '#A6845B');
            pGrad.addColorStop(1, '#7A5B3E');
            ctx.fillStyle = pGrad;
            ctx.fillRect(325, 25, 50, 195);

            // Crease Lines
            ctx.strokeStyle = 'rgba(255,255,255,0.9)';
            ctx.lineWidth = 1.5;
            // Bowling crease (top)
            ctx.beginPath(); ctx.moveTo(320, 48); ctx.lineTo(380, 48); ctx.stroke();
            // Popping crease (top)
            ctx.beginPath(); ctx.moveTo(322, 60); ctx.lineTo(378, 60); ctx.stroke();
            // Popping crease (batsman bottom)
            ctx.beginPath(); ctx.moveTo(320, 196); ctx.lineTo(380, 196); ctx.stroke();
            // Bowling crease (batsman bottom)
            ctx.beginPath(); ctx.moveTo(322, 208); ctx.lineTo(378, 208); ctx.stroke();

            // Non-striker Stumps (Top)
            ctx.fillStyle = '#E2B16B';
            ctx.fillRect(346, 45, 8, 3);

            // Striker Stumps (Bottom)
            ctx.fillStyle = '#FFD54F';
            ctx.fillRect(346, 208, 8, 3);

            // Wicketkeeper (crouching behind stumps)
            ctx.fillStyle = '#1E3A8A';
            ctx.beginPath();
            ctx.arc(350, 224, 5, 0, Math.PI * 2);
            ctx.fill();
            // Keeper gloves
            ctx.fillStyle = '#FFFFFF';
            ctx.fillRect(345, 219, 4, 3);
            ctx.fillRect(351, 219, 4, 3);
        }

        function drawBall(x, y, r, seamAngle) {
            ctx.save();
            ctx.translate(x, y);

            // Ball shadow on pitch
            ctx.fillStyle = 'rgba(0,0,0,0.4)';
            ctx.beginPath();
            ctx.ellipse(2, 3, r * 1.1, r * 0.5, 0, 0, Math.PI * 2);
            ctx.fill();

            // Red Leather Cricket Ball
            const bGrad = ctx.createRadialGradient(-r * 0.3, -r * 0.3, r * 0.1, 0, 0, r);
            bGrad.addColorStop(0, '#EF4444');
            bGrad.addColorStop(0.7, '#B91C1C');
            bGrad.addColorStop(1, '#7F1D1D');
            ctx.fillStyle = bGrad;
            ctx.beginPath();
            ctx.arc(0, 0, r, 0, Math.PI * 2);
            ctx.fill();

            // White Stitched Seam
            ctx.rotate(seamAngle);
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 1.2;
            ctx.setLineDash([2, 1.5]);
            ctx.beginPath();
            ctx.arc(0, 0, r * 0.85, -0.6, 0.6);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.restore();
        }

        function drawBatsman(swingProgress, targetAngle) {
            ctx.save();
            ctx.translate(352, 192);

            // Batsman Body (Cricket Whites)
            ctx.fillStyle = '#F8FAFC';
            ctx.beginPath();
            ctx.arc(0, 0, 6, 0, Math.PI * 2); // Helmet / Head
            ctx.fill();

            // Helmet Peak
            ctx.fillStyle = '#0F172A';
            ctx.fillRect(-2, -7, 4, 2);

            // White Batting Pads
            ctx.fillStyle = '#E2E8F0';
            ctx.fillRect(-5, 3, 4, 8);
            ctx.fillRect(1, 3, 4, 8);

            // English Willow Bat
            const currentBatAngle = targetAngle * Math.min(swingProgress * 2, 1.0);
            ctx.rotate(currentBatAngle);
            ctx.fillStyle = '#D97706'; // Willow Wood
            ctx.fillRect(2, -2, 16, 3.5);
            // Bat Grip (Red Rubber)
            ctx.fillStyle = '#DC2626';
            ctx.fillRect(0, -1.5, 3, 2.5);

            ctx.restore();
        }

        function drawBowler(runProgress) {
            const by = 20 + runProgress * 25; // 20 to 45
            ctx.save();
            ctx.translate(350, by);

            // Bowler Head
            ctx.fillStyle = '#10B981';
            ctx.beginPath();
            ctx.arc(0, 0, 5, 0, Math.PI * 2);
            ctx.fill();

            // Bowler Body (Flurry arms)
            ctx.strokeStyle = '#F1F5F9';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(-4, 0); ctx.lineTo(4, 0);
            ctx.stroke();

            ctx.restore();
        }

        function update() {
            drawGround();

            const curShot = SHOTS[shotIdx];
            document.getElementById('hudShot').innerText = curShot.name;
            document.getElementById('hudRuns').innerText = curShot.runs;

            // Phase 1: Bowler Run-Up (t: 0 -> 0.25)
            if (progress < 0.25) {
                const pRun = progress / 0.25;
                drawBowler(pRun);
                drawBatsman(0, curShot.batAngle);
                // Holding ball
                drawBall(352, 20 + pRun * 25, 3.5, progress * 15);
            }
            // Phase 2: Delivery & Pitch Bounce (t: 0.25 -> 0.50)
            else if (progress < 0.50) {
                drawBowler(1.0);
                const pDel = (progress - 0.25) / 0.25; // 0 to 1
                const ballY = 48 + pDel * (192 - 48);
                const ballX = 350 + Math.sin(pDel * Math.PI) * 4; // slight swing

                // Batsman prepares stance
                drawBatsman(pDel * 0.4, curShot.batAngle);

                // Check pitch bounce at t ~ 0.38
                if (progress >= 0.37 && progress <= 0.40 && dustParticles.length === 0) {
                    for (let i = 0; i < 8; i++) {
                        dustParticles.push({
                            x: ballX, y: ballY,
                            vx: (Math.random() - 0.5) * 3,
                            vy: (Math.random() - 0.5) * 3,
                            alpha: 1.0
                        });
                    }
                }

                drawBall(ballX, ballY, 4, progress * 25);
            }
            // Phase 3: Bat Strike & Explosive Flight (t: 0.50 -> 0.85)
            else if (progress < 0.85) {
                drawBowler(1.0);
                const pFlight = (progress - 0.50) / 0.35; // 0 to 1

                // Batsman completes full follow-through
                drawBatsman(0.5 + pFlight * 0.5, curShot.batAngle);

                // Ball rockets towards target boundary
                const startX = 352;
                const startY = 192;
                const ballX = startX + (curShot.targetX - startX) * pFlight;
                const ballY = startY + (curShot.targetY - startY) * pFlight;

                // Add tracer trail
                trail.push({ x: ballX, y: ballY, alpha: 0.9 });

                drawBall(ballX, ballY, 4.5, progress * 35);

                // Strike flash at t=0.50
                if (progress >= 0.50 && progress <= 0.54) {
                    ctx.fillStyle = 'rgba(255, 235, 59, 0.7)';
                    ctx.beginPath();
                    ctx.arc(352, 192, 18, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
            // Phase 4: Boundary Celebration (t: 0.85 -> 1.0)
            else {
                drawBowler(1.0);
                drawBatsman(1.0, curShot.batAngle);

                // Ball resting at boundary
                drawBall(curShot.targetX, curShot.targetY, 4.5, 0);

                // Spawn celebration fireworks at boundary
                if (fireworks.length < 24) {
                    for (let k = 0; k < 6; k++) {
                        fireworks.push({
                            x: curShot.targetX, y: curShot.targetY,
                            vx: (Math.random() - 0.5) * 5,
                            vy: (Math.random() - 0.5) * 5,
                            color: (k % 2 === 0) ? '#FFD54F' : '#10B981',
                            alpha: 1.0
                        });
                    }
                }

                // Boundary Sign Banner on Canvas
                ctx.save();
                ctx.fillStyle = 'rgba(0,0,0,0.85)';
                ctx.strokeStyle = '#FFD54F';
                ctx.lineWidth = 2;
                ctx.roundRect(curShot.targetX - 55, curShot.targetY - 26, 110, 22, 6);
                ctx.fill();
                ctx.stroke();

                ctx.font = 'bold 11px monospace';
                ctx.fillStyle = '#FFD54F';
                ctx.textAlign = 'center';
                ctx.fillText(`🏏 ${curShot.type}! ${curShot.runs}`, curShot.targetX, curShot.targetY - 11);
                ctx.restore();
            }

            // Draw Dust Particles
            for (let i = dustParticles.length - 1; i >= 0; i--) {
                const p = dustParticles[i];
                p.x += p.vx;
                p.y += p.vy;
                p.alpha -= 0.04;
                if (p.alpha <= 0) {
                    dustParticles.splice(i, 1);
                    continue;
                }
                ctx.fillStyle = `rgba(235, 205, 160, ${p.alpha})`;
                ctx.beginPath();
                ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
                ctx.fill();
            }

            // Draw Tracer Trail
            for (let i = trail.length - 1; i >= 0; i--) {
                const tr = trail[i];
                tr.alpha -= 0.035;
                if (tr.alpha <= 0) {
                    trail.splice(i, 1);
                    continue;
                }
                ctx.fillStyle = `rgba(255, 213, 79, ${tr.alpha * 0.7})`;
                ctx.beginPath();
                ctx.arc(tr.x, tr.y, 2, 0, Math.PI * 2);
                ctx.fill();
            }

            // Draw Fireworks
            for (let i = fireworks.length - 1; i >= 0; i--) {
                const fw = fireworks[i];
                fw.x += fw.vx;
                fw.y += fw.vy;
                fw.alpha -= 0.03;
                if (fw.alpha <= 0) {
                    fireworks.splice(i, 1);
                    continue;
                }
                ctx.fillStyle = fw.color;
                ctx.globalAlpha = fw.alpha;
                ctx.beginPath();
                ctx.arc(fw.x, fw.y, 2.2, 0, Math.PI * 2);
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }

            // Advance Loop
            progress += 0.007; // ~4.5 seconds per full delivery cycle
            if (progress >= 1.0) {
                progress = 0;
                trail = [];
                fireworks = [];
                dustParticles = [];
                shotIdx = (shotIdx + 1) % SHOTS.length;
            }

            requestAnimationFrame(update);
        }

        update();
    </script>
    </body>
    </html>
    """

print("HTML Hero Animation template verified!")
