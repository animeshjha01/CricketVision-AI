import streamlit.components.v1 as components

def get_hawkeye_canvas_html(ball_length: str, ball_line: str) -> str:
    # Coordinates on a 540x300 canvas
    # Pitch width: 220px (from x=160 to x=380), center=270
    # Length: Short=80, Good=150, Full=225
    y_map = {"Short": 85, "Good": 150, "Full": 225}
    x_map = {"Outside Off": 200, "Stumps": 270, "Leg Side": 340}

    bounce_x = x_map.get(ball_line, 270)
    bounce_y = y_map.get(ball_length, 150)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }}
        #container {{
            position: relative;
            width: 520px;
            height: 310px;
            background: radial-gradient(circle at 50% 50%, #0F2D18 0%, #08170D 100%);
            border: 2px solid #2E7D32;
            border-radius: 18px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.6), inset 0 0 40px rgba(0,0,0,0.5);
            overflow: hidden;
        }}
        canvas {{
            display: block;
        }}
        .hud-overlay {{
            position: absolute;
            bottom: 8px;
            left: 0;
            right: 0;
            text-align: center;
            font-size: 11px;
            font-family: monospace;
            color: #FFD54F;
            letter-spacing: 1.5px;
            text-shadow: 0 0 8px rgba(255,213,79,0.5);
        }}
    </style>
    </head>
    <body>
    <div id="container">
        <canvas id="pitchCanvas" width="520" height="310"></canvas>
        <div class="hud-overlay">
            🔴 HAWK-EYE BALL TRACKING: {ball_length.upper()} LENGTH · {ball_line.upper()}
        </div>
    </div>
    <script>
        const canvas = document.getElementById('pitchCanvas');
        const ctx = canvas.getContext('2d');

        const bx = {bounce_x};
        const by = {bounce_y};
        const startX = 270, startY = 25;
        const endX = 270, endY = 285;

        let t = 0; // 0 to 1
        let dustRings = [];
        let zingFlash = 0;

        function drawPitch() {{
            ctx.clearRect(0, 0, 520, 310);

            // Mown grass pattern background
            ctx.fillStyle = '#0F2615';
            ctx.fillRect(0, 0, 520, 310);

            // Outfield mowing stripes
            for (let i = -100; i < 600; i += 40) {{
                ctx.fillStyle = 'rgba(255, 255, 255, 0.025)';
                ctx.beginPath();
                ctx.moveTo(i, 0);
                ctx.lineTo(i + 20, 0);
                ctx.lineTo(i + 80, 310);
                ctx.lineTo(i + 60, 310);
                ctx.closePath();
                ctx.fill();
            }}

            // 22-Yard Pitch Strip (Clay / Turf hybrid)
            const pitchGrad = ctx.createLinearGradient(160, 0, 380, 0);
            pitchGrad.addColorStop(0, '#7D6148');
            pitchGrad.addColorStop(0.15, '#9C7A5B');
            pitchGrad.addColorStop(0.5, '#B58E6B');
            pitchGrad.addColorStop(0.85, '#9C7A5B');
            pitchGrad.addColorStop(1, '#7D6148');

            ctx.fillStyle = pitchGrad;
            ctx.fillRect(160, 15, 200, 275);
            ctx.strokeStyle = '#4CAF50';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(160, 15, 200, 275);

            // Length Grid Lines
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
            ctx.lineWidth = 1;

            // Short length line
            ctx.beginPath(); ctx.moveTo(160, 95); ctx.lineTo(360, 95); ctx.stroke();
            // Good length line
            ctx.beginPath(); ctx.moveTo(160, 165); ctx.lineTo(360, 165); ctx.stroke();

            // Line corridors
            ctx.beginPath(); ctx.moveTo(225, 15); ctx.lineTo(225, 290); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(315, 15); ctx.lineTo(315, 290); ctx.stroke();
            ctx.setLineDash([]);

            // Bowler Crease (Top)
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(180, 35); ctx.lineTo(340, 35); ctx.stroke();

            // Bowler Stumps
            ctx.fillStyle = '#FFD54F';
            ctx.fillRect(265, 22, 10, 6);

            // Batsman Popping Crease (Bottom)
            ctx.beginPath(); ctx.moveTo(175, 265); ctx.lineTo(345, 265); ctx.stroke();

            // Batsman Stumps & Zing Bails
            ctx.fillStyle = zingFlash > 0 ? '#FF1744' : '#FFD54F';
            ctx.fillRect(263, 272, 14, 5); // bails
            ctx.fillRect(264, 277, 3, 10);
            ctx.fillRect(268, 277, 3, 10);
            ctx.fillRect(272, 277, 3, 10);

            if (zingFlash > 0) {{
                ctx.fillStyle = 'rgba(255, 23, 68, ' + (zingFlash * 0.4) + ')';
                ctx.beginPath();
                ctx.arc(270, 275, 20, 0, Math.PI * 2);
                ctx.fill();
                zingFlash -= 0.03;
            }}

            // Labels
            ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.font = '9px monospace';
            ctx.fillText('SHORT', 165, 65);
            ctx.fillText('GOOD', 165, 130);
            ctx.fillText('FULL', 165, 210);

            ctx.fillText('OFF', 190, 280);
            ctx.fillStyle = '#FFD54F';
            ctx.fillText('STUMPS', 255, 258);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.fillText('LEG', 325, 280);

            // Trajectory Arc Path (Dashed)
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = 'rgba(255, 213, 79, 0.4)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.quadraticCurveTo((startX + bx) / 2 + 10, (startY + by) / 2, bx, by);
            ctx.quadraticCurveTo((bx + endX) / 2, (by + endY) / 2, endX, endY);
            ctx.stroke();
            ctx.setLineDash([]);

            // Pitch impact shockwaves
            for (let i = dustRings.length - 1; i >= 0; i--) {{
                let r = dustRings[i];
                ctx.strokeStyle = 'rgba(255, 213, 79, ' + r.alpha + ')';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.ellipse(r.x, r.y, r.radius * 1.4, r.radius * 0.8, 0, 0, Math.PI * 2);
                ctx.stroke();
                r.radius += 1.2;
                r.alpha -= 0.03;
                if (r.alpha <= 0) dustRings.splice(i, 1);
            }}

            // Ball bounce target marker
            ctx.fillStyle = 'rgba(255, 213, 79, 0.2)';
            ctx.beginPath();
            ctx.ellipse(bx, by, 12, 7, 0, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#FFD54F';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Ball Position along 2-phase trajectory (Bowler->Bounce, Bounce->Batsman)
            let curX, curY, ballScale = 1;
            if (t < 0.6) {{
                // Phase 1: In the air from bowler to pitch
                let p = t / 0.6;
                let cX = (startX + bx) / 2 + 10;
                let cY = (startY + by) / 2;
                curX = (1 - p) * (1 - p) * startX + 2 * (1 - p) * p * cX + p * p * bx;
                curY = (1 - p) * (1 - p) * startY + 2 * (1 - p) * p * cY + p * p * by;
                ballScale = 1.2 - 0.4 * p; // shrinks slightly as it moves away towards pitch
            }} else {{
                // Phase 2: Off the pitch towards stumps
                let p = (t - 0.6) / 0.4;
                let cX = (bx + endX) / 2;
                let cY = (by + endY) / 2;
                curX = (1 - p) * (1 - p) * bx + 2 * (1 - p) * p * cX + p * p * endX;
                curY = (1 - p) * (1 - p) * by + 2 * (1 - p) * p * cY + p * p * endY;
                ballScale = 0.8 + 0.4 * p;
            }}

            // Draw Red Leather Cricket Ball with Stitched White Seam
            ctx.save();
            ctx.translate(curX, curY);
            ctx.rotate(t * Math.PI * 4); // Spinning seam
            let r = 7 * ballScale;

            // Ball shadow
            ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
            ctx.beginPath();
            ctx.ellipse(2, 4, r, r * 0.5, 0, 0, Math.PI * 2);
            ctx.fill();

            // Red leather sphere
            const ballGrad = ctx.createRadialGradient(-r * 0.3, -r * 0.3, r * 0.1, 0, 0, r);
            ballGrad.addColorStop(0, '#EF5350');
            ballGrad.addColorStop(0.7, '#C62828');
            ballGrad.addColorStop(1, '#8E0000');
            ctx.fillStyle = ballGrad;
            ctx.beginPath();
            ctx.arc(0, 0, r, 0, Math.PI * 2);
            ctx.fill();

            // White Stitched Seam
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 1.4;
            ctx.setLineDash([2, 1.5]);
            ctx.beginPath();
            ctx.arc(0, 0, r * 0.9, -0.6, 0.6);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.restore();

            // Advance time
            t += 0.016;
            if (t >= 0.59 && t <= 0.61 && dustRings.length === 0) {{
                dustRings.push({{ x: bx, y: by, radius: 4, alpha: 0.9 }});
            }}
            if (t >= 0.98) {{
                zingFlash = 1.0; // Flash bails
                t = 0;
            }}

            requestAnimationFrame(drawPitch);
        }}

        drawPitch();
    </script>
    </body>
    </html>
    """
    return html_code
