import os

html_test = """
<!DOCTYPE html>
<html>
<head>
<style>
    body { margin:0; padding:0; background:#05140A; font-family:'Segoe UI', sans-serif; display:flex; justify-content:center; align-items:center; }
    #canvas-container { position:relative; width:680px; height:260px; border-radius:18px; overflow:hidden; border:2px solid #2E7D32; box-shadow:0 12px 35px rgba(0,0,0,0.7); }
    canvas { display:block; }
    .banner { position:absolute; top:8px; left:12px; font-family:monospace; font-size:10px; font-weight:700; color:#FFD54F; letter-spacing:1.5px; background:rgba(0,0,0,0.65); padding:4px 10px; border-radius:10px; border:1px solid rgba(255,213,79,0.3); }
</style>
</head>
<body>
<div id="canvas-container">
    <canvas id="heroCanvas" width="680" height="260"></canvas>
    <div class="banner">🔴 LIVE CRICKET STADIUM MATCH ACTION</div>
</div>
<script>
    const canvas = document.getElementById('heroCanvas');
    const ctx = canvas.getContext('2d');

    const SHOTS = [
        { name: "CLASSIC COVER DRIVE", boundary: "EXTRA COVER", type: "FOUR", targetX: 110, targetY: 45, shotAngle: -2.3, runs: "4️⃣" },
        { name: "AUTHORITATIVE PULL SHOT", boundary: "DEEP MIDWICKET", type: "SIX", targetX: 580, targetY: 55, shotAngle: 2.2, runs: "6️⃣" },
        { name: "EXQUISITE STRAIGHT DRIVE", boundary: "LONG OFF", type: "FOUR", targetX: 280, targetY: 25, shotAngle: -1.7, runs: "4️⃣" },
        { name: "ARTISTIC LATE CUT", boundary: "THIRD MAN", type: "FOUR", targetX: 80, targetY: 220, shotAngle: -2.8, runs: "4️⃣" },
        { name: "WRISTY FLICK SHOT", boundary: "FORWARD SQUARE LEG", type: "FOUR", targetX: 590, targetY: 170, shotAngle: 1.8, runs: "4️⃣" }
    ];

    let shotIdx = 0;
    let animTime = 0;
    let dustPuff = [];
    let fireworks = [];
    let state = "bowling"; // "bowling", "shot", "boundary", "reset"

    function drawField() {
        // Mown Grass Outfield
        ctx.fillStyle = '#0F3818';
        ctx.fillRect(0, 0, 680, 260);

        // Mown Turf Alternating Bands
        for (let x = 0; x < 680; x += 55) {
            ctx.fillStyle = (Math.floor(x / 55) % 2 === 0) ? '#0D3115' : '#113E1B';
            ctx.fillRect(x, 0, 55, 260);
        }

        // Floodlight Beams from top corners
        const gradL = ctx.createRadialGradient(40, 10, 5, 120, 80, 220);
        gradL.addColorStop(0, 'rgba(255, 235, 59, 0.28)');
        gradL.addColorStop(1, 'transparent');
        ctx.fillStyle = gradL;
        ctx.fillRect(0, 0, 680, 260);

        const gradR = ctx.createRadialGradient(640, 10, 5, 560, 80, 220);
        gradR.addColorStop(0, 'rgba(255, 235, 59, 0.28)');
        gradR.addColorStop(1, 'transparent');
        ctx.fillStyle = gradR;
        ctx.fillRect(0, 0, 680, 260);

        // Boundary Oval
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([8, 6]);
        ctx.beginPath();
        ctx.ellipse(340, 130, 315, 115, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // 30-Yard Circle
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.ellipse(340, 130, 180, 65, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Pitch Strip (Clay / Khaki Tan)
        const pGrad = ctx.createLinearGradient(300, 0, 380, 0);
        pGrad.addColorStop(0, '#8A6844');
        pGrad.addColorStop(0.5, '#BFA175');
        pGrad.addColorStop(1, '#8A6844');
        ctx.fillStyle = pGrad;
        ctx.fillRect(315, 30, 50, 190);

        // Crease Lines
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.5;
        // Bowling Crease (Top)
        ctx.beginPath(); ctx.moveTo(310, 50); ctx.lineTo(370, 50); ctx.stroke();
        // Popping Crease (Top)
        ctx.beginPath(); ctx.moveTo(312, 62); ctx.lineTo(368, 62); ctx.stroke();
        // Popping Crease (Batsman - Bottom)
        ctx.beginPath(); ctx.moveTo(310, 195); ctx.lineTo(370, 195); ctx.stroke();
        // Bowling Crease (Batsman - Bottom)
        ctx.beginPath(); ctx.moveTo(312, 207); ctx.lineTo(368, 207); ctx.stroke();

        // Non-Striker Stumps
        ctx.fillStyle = '#E8A84A';
        ctx.fillRect(336, 46, 8, 3);

        // Striker Stumps
        ctx.fillStyle = '#FFD54F';
        ctx.fillRect(336, 208, 8, 3);
    }

    function loop() {
        drawField();
        animTime += 0.016;
        requestAnimationFrame(loop);
    }
    loop();
</script>
</body>
</html>
"""

with open("test_hero_anim.html", "w") as f:
    f.write(html_test)
print("Hero anim test written successfully!")
