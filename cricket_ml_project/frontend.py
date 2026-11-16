"""Presentation for the CricketVision practice workspace."""
import streamlit as st
import base64
import re
from pathlib import Path


@st.cache_data(show_spinner=False)
def _background_data(path, modified):
    """Cache the local photo while refreshing it when the asset changes."""
    return base64.b64encode(Path(path).read_bytes()).decode('ascii')


def apply_stadium_background():
    photo = Path(__file__).parent / 'assets' / 'stadium.jpg'
    if not photo.is_file():
        return
    encoded = _background_data(str(photo), photo.stat().st_mtime_ns)
    st.html("""<style>
    .stApp {
      background:linear-gradient(rgba(4,16,11,.68),rgba(4,16,11,.82)),
      url('data:image/jpeg;base64,""" + encoded + """') center center / cover fixed!important;
    }
    .block-container {background:rgba(7,20,14,.35);border-radius:16px;}
    [data-testid="stVerticalBlockBorderWrapper"] > div {background:rgba(12,29,21,.92);}
    .cv-nav {border-bottom-color:#607263;}
    .cv-footer {color:#c0cdbf;}
    </style>""")


def apply_theme():
    st.html("""<style>
    :root {color-scheme:dark;}
    .stApp {background:#101c18;color:#f2f0e5;font-family:Arial,sans-serif;}
    [data-testid="stHeader"] {display:none;}
    .cv-nav,.cv-hero,.cv-steps,.cv-eyebrow,.cv-empty,.cv-footer,.section-banner {font-family:Arial,sans-serif;}
    .block-container {max-width:1240px;padding:2rem 2.5rem 3rem;}
    h1,h2,h3,p,label,button {font-family:Arial,sans-serif;}
    [data-testid="stMarkdownContainer"] p,[data-testid="stWidgetLabel"] p {color:#dce3da;}
    [data-testid="stCaptionContainer"] p {color:#a8b7ab;line-height:1.6;}
    [data-testid="stVerticalBlockBorderWrapper"] > div,
    [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlockBorderWrapper"] {border-color:#34463c!important;border-radius:16px;}
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {background:#1b2c23;color:#f2f0e5;border-color:#425449;border-radius:8px;}
    [data-testid="stFileUploaderDropzone"] {background:#1b2c23;border:1px dashed #728878;border-radius:12px;}
    .stButton button,.stDownloadButton button {background:#20372a;color:#eeeede;border:1px solid #47604e;border-radius:8px;min-height:43px;font-size:.88rem;transition:background .2s;}
    .stButton button:hover,.stDownloadButton button:hover {background:#304c38;border-color:#d6e7a3;color:#fff;}
    .stButton button[kind="primary"] {background:#d6e7a3;color:#182719;border:0;font-weight:700;min-height:52px;}
    .stButton button:focus-visible,a:focus-visible {outline:3px solid #e1b66f;outline-offset:4px;}
    [data-testid="stRadio"] label {padding:4px 8px 4px 0;}
    [data-testid="stVideo"] video {max-height:300px;border-radius:10px;background:#0b120e;}
    .st-key-pose_review [data-testid="stImage"] {max-width:100%;margin-inline:auto;text-align:center;}
    .st-key-pose_review [data-testid="stImage"] img {display:block;width:auto!important;max-width:100%!important;height:auto!important;max-height:min(400px,52vh)!important;object-fit:contain;margin-inline:auto;border-radius:8px;}
    @media(max-width:760px){.st-key-pose_review [data-testid="stImage"] img{max-height:min(320px,48vh)!important;}}
    .cv-nav {display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #34463c;padding:0 0 22px;margin-bottom:26px;gap:20px;}
    .cv-brand {display:flex;align-items:center;gap:11px;color:#f2f0e5;font-weight:800;font-size:20px;letter-spacing:-.5px;}
    .cv-mark {display:grid;place-items:center;width:37px;height:37px;background:#d6e7a3;border-radius:50%;color:#173221;font-size:25px;}
    .cv-nav-links {display:flex;gap:28px;align-items:center;font-size:13px;}
    .cv-nav a {color:#bfccbf;text-decoration:none;}.cv-nav a:hover {color:#d6e7a3;}
    .cv-tag {color:#c7d5bf;border:1px solid #4c6251;border-radius:30px;padding:7px 12px;font-size:11px;letter-spacing:1px;}
    .cv-hero {display:grid;grid-template-columns:1.1fr 1fr;min-height:340px;border:1px solid #3a5141;border-radius:18px;overflow:hidden;background:#1b3225;margin-bottom:20px;}
    .cv-copy {padding:38px 38px 34px;position:relative;z-index:1;}
    .cv-eyebrow {font-size:10px;font-weight:700;letter-spacing:2.5px;color:#d6e7a3;text-transform:uppercase;margin-bottom:18px;}
    .cv-copy h1 {font-family:Georgia,serif;font-size:clamp(38px,4.8vw,62px);font-weight:400;line-height:1.04;letter-spacing:-2px;margin:0 0 18px;color:#f4f0df;}
    .cv-copy h1 em {color:#d6e7a3;font-weight:400;}
    .cv-copy p {color:#bacbbd;max-width:380px;font-size:14px;line-height:1.7;margin:0 0 23px;}
    .cv-cta {display:inline-block;padding:12px 18px;border-radius:7px;background:#d6e7a3;color:#1b2c1c!important;text-decoration:none;font-size:13px;font-weight:700;}
    .cv-art {position:relative;min-height:340px;background:radial-gradient(ellipse at center,#2d5136,#172c21);overflow:hidden;}
    .cv-art img {width:100%;height:100%;position:absolute;inset:0;object-fit:contain;}
    .cv-art-label {position:absolute;bottom:22px;left:24px;color:#d6e7a3;font-size:10px;letter-spacing:2px;}
    .cv-steps {display:grid;grid-template-columns:repeat(3,1fr);border-bottom:1px solid #34463c;padding:8px 0 24px;margin-bottom:26px;gap:20px;}
    .cv-step {display:flex;align-items:center;gap:12px;font-size:13px;color:#e0e6d7;}
    .cv-step b {color:#d6e7a3;font-family:Georgia,serif;font-size:25px;font-weight:400;}
    .cv-step small {display:block;color:#92a896;margin-top:5px;font-size:11px;}
    .section-banner {display:flex;align-items:center;justify-content:space-between;gap:12px;border-bottom:1px solid #34463c;padding:5px 0 16px;margin-bottom:15px;}
    .section-banner-title {font-size:18px;font-weight:600;color:#f2f0e5;}
    .section-banner-pill {font-size:10px;letter-spacing:1px;color:#cfdda8;white-space:nowrap;}
    .cv-note {border-left:2px solid #d6e7a3;padding:2px 0 2px 14px;margin:16px 0;color:#adbfaf;font-size:12px;line-height:1.8;}
    .cv-empty {padding:28px;border:1px dashed #48604e;border-radius:12px;text-align:center;color:#a9bcad;margin:20px 0;}
    .cv-empty strong {display:block;color:#e8ecdd;font-family:Georgia,serif;font-size:24px;font-weight:400;margin-bottom:8px;}
    .scoreboard-card {background:#203b2b;border:1px solid #61734e;border-radius:14px;padding:26px;margin-bottom:20px;}
    .scoreboard-shot {font-family:Georgia,serif;font-size:42px;color:#f4f0df;margin:10px 0;}
    .cricket-divider {height:1px;background:#34463c;margin:25px 0;}
    .cv-footer {display:flex;justify-content:space-between;gap:20px;border-top:1px solid #34463c;padding-top:22px;margin-top:35px;color:#8ea492;font-size:11px;}
    :root {--cv-panel:#14281e;--cv-border:#405848;--cv-muted:#b5c4b5;}
    .cv-result-card,.cv-metric-card,.cv-coach-card {background:var(--cv-panel)!important;border:1px solid var(--cv-border)!important;border-radius:12px!important;box-shadow:none!important;}
    .cv-metric-card {min-height:112px;padding:20px!important;}
    .cv-coach-card {padding:24px!important;}
    .cv-verdict {border-radius:12px!important;box-shadow:none!important;}
    .cv-review-nav {display:flex;align-items:center;flex-wrap:wrap;gap:9px;padding:15px 18px;background:#162c21;border:1px solid #405848;border-radius:12px;margin-bottom:15px;font:13px Arial,sans-serif;}
    .cv-review-nav strong {color:#d6e7a3;margin-right:auto;}
    .cv-review-nav a {color:#e5eadb;text-decoration:none;padding:8px 12px;border:1px solid #4a6351;border-radius:7px;transition:background 160ms ease;}
    .cv-review-nav a:hover {background:#2c4835;}
    .cv-anchor {scroll-margin-top:18px;outline-offset:4px;}
    .cv-skip {position:fixed;left:12px;top:-100px;z-index:99999;background:#d6e7a3;color:#17291a!important;padding:12px 20px;border-radius:7px;font:700 14px Arial,sans-serif;}
    .cv-skip:focus {top:12px;}
    a:focus-visible,button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible,[role="combobox"]:focus-visible,[role="radio"]:focus-visible,[tabindex]:focus-visible {outline:3px solid #e1b66f!important;outline-offset:3px!important;}
    .stButton button {transition:background 160ms ease,border-color 160ms ease;}
    .cv-hero,.cv-review-nav {animation:cv-reveal 220ms ease-out;}
    @keyframes cv-reveal {from{opacity:.6;transform:translateY(3px)}to{opacity:1;transform:translateY(0)}}
    @media(min-width:900px) and (max-height:850px){.block-container{padding-left:2rem;padding-right:2rem;}.cv-copy{padding:23px 28px!important;}.cv-copy h1{font-size:50px!important;}.cv-art{min-height:260px!important;}.cv-hero{min-height:260px!important;}}
    @media(max-width:760px){.block-container{padding:1.2rem 1rem 2rem;}.cv-hero{grid-template-columns:1fr;}.cv-copy{padding:28px;}.cv-art{min-height:250px;}.cv-nav-links{gap:12px;}.cv-tag{display:none;}.cv-steps{gap:10px;}.cv-step{align-items:flex-start;font-size:11px;}.cv-step small{line-height:1.5;}.section-banner-title{font-size:16px;}.cv-footer{flex-direction:column;}.cv-brand{font-size:17px;}}
    @media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important;}}
    </style>""")
    apply_stadium_background()


def render_header(compact=False):
    st.html("""<style>
    .stApp {background:radial-gradient(ellipse at 85% 0%,#193b30 0,transparent 44%),#0b1713;}
    .block-container {max-width:1380px;padding-top:1.3rem;}
    .cv-nav {padding-bottom:14px;margin-bottom:16px;}
    .cv-mark {background:#ba4c39;color:#fff4dc;box-shadow:inset -7px -3px 0 #85382f;position:relative;}
    .cv-mark:after {content:"";position:absolute;inset:4px 13px;border-left:1px dashed #f9dbc1;border-right:1px dashed #f9dbc1;border-radius:50%;transform:rotate(25deg);}
    .cv-hero {grid-template-columns:1.1fr 1fr;background:#122d21;min-height:290px;border-color:#355746;position:relative;margin-bottom:0;border-radius:14px 14px 0 0;}
    .cv-copy {padding:27px 32px;}
    .cv-copy h1 {font-family:Impact,'Arial Narrow',Arial,sans-serif;font-size:clamp(38px,4.8vw,64px);font-weight:900;letter-spacing:-1.8px;text-transform:uppercase;line-height:.98;margin-bottom:13px;}
    .cv-copy h1 em {font-style:normal;color:#dcf49b;font-weight:900;}
    .cv-copy p {max-width:440px;font-size:13px;margin-bottom:17px;line-height:1.65;}
    .cv-eyebrow {margin-bottom:12px;color:#d2b981;}
    .cv-art {min-height:290px;background:radial-gradient(ellipse at 75% 10%,#51795755,transparent 70%),#10271e;}
    .cv-art img {object-fit:cover;}
    .cv-art-label {left:auto;right:18px;bottom:15px;background:#0c201bdc;border:1px solid #48654b;padding:8px 11px;border-radius:4px;font-size:9px;}
    .cv-cta {background:#dcf49b;padding:10px 16px;}
    .cv-steps {border:1px solid #355746;border-top:0;background:#14271e;border-radius:0 0 12px 12px;padding:15px 25px;margin-bottom:18px;}
    .cv-step b {font:700 21px monospace;color:#d2b981;}
    .cv-step small {color:#a8b8aa;}
    .cv-step + .cv-step {border-left:1px solid #365041;padding-left:22px;}
    .cv-session {display:flex;align-items:center;justify-content:space-between;gap:15px;background:linear-gradient(100deg,#234833,#14271e);border:1px solid #446044;border-radius:10px;padding:14px 20px;margin-bottom:15px;font-family:Arial,sans-serif;}
    .cv-session strong {display:block;color:#f3eedc;font-size:21px;}
    .cv-session small {display:block;color:#b8c9b8;margin-top:5px;font-size:12px;}
    .cv-session span {font:11px monospace;letter-spacing:2px;color:#d6e7a3;}
    .section-banner {padding-bottom:11px;margin-bottom:8px;}
    .section-banner-title {font-size:17px;}
    [data-testid="stVideo"],.stVideo {width:100%!important;height:clamp(180px,32vh,280px)!important;max-height:280px!important;overflow:hidden;border:1px solid #4b6352;border-radius:9px;background:#050b08;display:flex;align-items:center;justify-content:center;}
    [data-testid="stVideo"] video {display:block!important;width:100%!important;height:100%!important;max-height:280px!important;object-fit:contain!important;border-radius:8px;}
    video[data-testid="stVideo"] {height:clamp(180px,32vh,280px)!important;object-fit:contain!important;}
    .stButton button[kind="primary"] p {color:#182719!important;font-weight:700;}
    video:fullscreen {max-height:none!important;height:100vh!important;object-fit:contain!important;}
    .cv-replay-label {display:flex;justify-content:space-between;font:10px monospace;letter-spacing:1.5px;color:#c4d2b8;margin:0 0 7px;}
    .cv-replay-label span:last-child {color:#d2b981;}
    [data-testid="stFileUploaderDropzone"] {padding:12px!important;min-height:85px;}
    .cv-empty {background:repeating-linear-gradient(90deg,#172b2080 0,#172b2080 45px,#1c332580 45px,#1c332580 90px);padding:23px;}
    @media(max-width:760px){
      .block-container{padding-top:1rem;}
      .cv-hero{grid-template-columns:1fr;min-height:0;}
      .cv-copy{padding:23px;}
      .cv-copy h1{font-size:43px;}
      .cv-art{min-height:190px;}
      .cv-art img{object-position:center 44%;}
      .cv-steps{padding:14px;gap:12px;}
      .cv-step{display:block;font-size:10px;}
      .cv-step b{display:block;font-size:16px;margin-bottom:7px;}
      .cv-step + .cv-step{padding-left:10px;}
      .cv-step small{font-size:10px;}
      .cv-session{padding:13px;}
      .cv-session strong{font-size:18px;}
      .cv-session span{display:none;}
      [data-testid="stVideo"],.stVideo,video[data-testid="stVideo"]{height:220px!important;}
    }
    </style>""")
    st.html("""
    <a class="cv-skip" href="#practice">Skip to batting lab</a>
    <nav class="cv-nav" aria-label="Main navigation">
      <div class="cv-brand"><span class="cv-mark" aria-hidden="true"></span>CricketVision<span style="color:#cbb47d;font-size:11px;letter-spacing:2px">AI</span></div>
      <div class="cv-nav-links"><a href="#practice">Batting lab</a><a href="#field-notes">Coach's notes</a><span class="cv-tag">FROM THE NETS. FOR THE GAME.</span></div>
    </nav>""")
    if compact:
        st.html('<div id="practice" tabindex="-1" class="cv-session cv-anchor"><div><strong>In the nets. In your element.</strong><small>Your clip is ready. Read the delivery, then review your stroke.</small></div><span>BATTING LAB / SESSION IN PROGRESS</span></div>')
        return
    markup = """
    <section class="cv-hero">
      <div class="cv-copy"><div class="cv-eyebrow">YOUR GAME. UNDER THE SPOTLIGHT.</div>
      <h1>Read the ball.<br><em>Own the crease.</em></h1>
      <p>Your personal batting analysis lab. Upload a stroke, discover the shot you played, explore your movement, and take a focused drill back to the nets.</p>
      <a class="cv-cta" href="#practice">Pad up. Let's analyse. &nbsp; ↗</a></div>
      <div class="cv-art">
      <svg viewBox="0 0 620 340" role="img" aria-label="Batter playing a drive under floodlights, with a visual swing path">
      <defs>
        <linearGradient id="sky" x2="0" y2="1"><stop stop-color="#0b1b18"/><stop offset="1" stop-color="#33523b"/></linearGradient>
        <linearGradient id="beam" x2=".4" y2="1"><stop stop-color="#f9edba" stop-opacity=".15"/><stop offset="1" stop-color="#e4f8c9" stop-opacity="0"/></linearGradient>
        <linearGradient id="kit" x2="1" y2="1"><stop stop-color="#fff7df"/><stop offset="1" stop-color="#aabdaa"/></linearGradient>
        <pattern id="seats" width="12" height="9" patternUnits="userSpaceOnUse"><rect x="2" y="2" width="5" height="2" fill="#a9bd90" opacity=".3"/></pattern>
      </defs>
      <rect width="620" height="340" fill="url(#sky)"/>
      <path d="M0 157Q310 225 620 146V230H0Z" fill="#11251b"/>
      <path d="M0 163Q310 235 620 153V224H0Z" fill="url(#seats)"/>
      <path d="M0 196Q310 267 620 192" fill="none" stroke="#acc19a" opacity=".3"/>
      <ellipse cx="310" cy="340" rx="480" ry="125" fill="#244833"/>
      <ellipse cx="310" cy="340" rx="430" ry="107" fill="none" stroke="#cbd4b1" opacity=".5"/>
      <path d="M0 300L620 234M0 340L620 270" stroke="#416645" stroke-width="30" opacity=".25"/>
      <path d="M280 246L388 246L465 340H209Z" fill="#aa9163" opacity=".65"/>
      <path d="M240 300H430M263 290V313M398 290V313" stroke="#e9e7c8" stroke-width="2" opacity=".8"/>
      <g stroke="#63766a" stroke-width="4"><path d="M80 40V187M541 28V177"/></g>
      <g fill="#d8e8bc"><rect x="52" y="32" width="59" height="19" rx="3"/><rect x="514" y="20" width="59" height="19" rx="3"/></g>
      <g fill="#fff9d8"><rect x="57" y="35" width="49" height="3"/><rect x="57" y="44" width="49" height="3"/><rect x="519" y="23" width="49" height="3"/><rect x="519" y="32" width="49" height="3"/></g>
      <path d="M52 51L1 300L450 300L111 51Z M514 39L210 300L620 300L573 39Z" fill="url(#beam)"/>
      <ellipse cx="335" cy="302" rx="87" ry="10" fill="#0a1a12" opacity=".5"/>
      <g fill="#f4e7bd" stroke="#d4c193" stroke-width="1"><rect x="266" y="242" width="4" height="62"/><rect x="274" y="242" width="4" height="62"/><rect x="282" y="242" width="4" height="62"/><path d="M264 242h24"/></g>
      <!-- A balanced front-foot drive: helmet, shirt, gloves, pads and willow. -->
      <path d="M326 207L353 216L381 279L399 293L387 301L365 290L337 247L315 240Z" fill="#9bb2a0"/>
      <path d="M308 205L336 220L312 254L301 294L283 297L289 249Z" fill="url(#kit)"/>
      <path d="M289 251L310 255L300 291L283 292Z" fill="#e9edda" stroke="#91a38d" stroke-width="2"/>
      <path d="M292 257L287 286M299 258L294 287M371 265L384 287" stroke="#b3bfa7" stroke-width="2"/>
      <path d="M282 291L302 290L306 301L274 303L273 298Z M385 291L399 290L414 299L411 305L384 302Z" fill="#f1eedb"/>
      <path d="M306 146Q328 135 347 158L352 195L334 227L301 212L305 179Z" fill="url(#kit)"/>
      <path d="M309 157L285 182L304 200L318 190L307 180L327 166" fill="none" stroke="#e4e9d5" stroke-width="17" stroke-linejoin="round"/>
      <path d="M341 157L362 183L337 201L323 191L344 179L333 167" fill="#d7e0cb"/>
      <path d="M304 195L323 201L337 195" fill="none" stroke="#f5f0d7" stroke-width="13" stroke-linecap="round"/>
      <path d="M325 196L363 117" stroke="#343f30" stroke-width="7"/>
      <path d="M354 133L376 79Q379 74 383 78L396 85Q398 88 395 93L370 143Z" fill="#d8b77c" stroke="#f3deb0" stroke-width="2"/>
      <path d="M371 111L384 86" stroke="#ab8051" stroke-width="3"/>
      <path d="M319 137L321 150L337 152L343 135Z" fill="#bd9270"/>
      <path d="M310 124Q310 101 330 102Q351 105 350 125L345 138L317 137Z" fill="#152f24" stroke="#6e8872" stroke-width="2"/>
      <path d="M331 120L354 123L350 140L333 141Z" fill="none" stroke="#b6c6b0" stroke-width="1.5"/>
      <path d="M332 128L352 130M332 135L350 136" stroke="#b6c6b0"/>
      <path d="M340 230Q441 240 499 167" fill="none" stroke="#ddf6a1" stroke-width="2" stroke-dasharray="6 6"/>
      <circle cx="499" cy="167" r="7" fill="#d45b43"/><path d="M496 161q8 5 6 11" fill="none" stroke="#f7d8b0"/>
      <g fill="#e2e8cb" font-family="Arial,sans-serif" font-size="9" letter-spacing="2"><text x="420" y="141">FIND YOUR TIMING</text><text x="31" y="321" opacity=".55">ILLUSTRATED PRACTICE VIEW</text></g>
      </svg><div class="cv-art-label">THE SWEET SPOT IS IN THE DETAILS.</div></div>
    </section>
    <div class="cv-steps">
      <div class="cv-step"><b>01 /</b><div>Know your stroke<small>AI shot recognition from your clip</small></div></div>
      <div class="cv-step"><b>02 /</b><div>Look beyond the boundary<small>Shot selection, balance & movement</small></div></div>
      <div class="cv-step"><b>03 /</b><div>Make the next ball count<small>Coaching checkpoints & net drills</small></div></div>
    </div>
    <div id="practice" tabindex="-1" class="cv-eyebrow cv-anchor">THE BATTING LAB / YOUR NEXT INNINGS STARTS HERE</div>
    """
    svg = re.search(r"<svg.*?</svg>", markup, re.DOTALL).group(0)
    svg = svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    encoded = base64.b64encode(svg.encode()).decode()
    markup = re.sub(r"<svg.*?</svg>", f'<img src="data:image/svg+xml;base64,{encoded}" alt="Batter driving under stadium floodlights, with an illustrated swing path">', markup, flags=re.DOTALL)
    st.html(markup)


def section(title, label):
    st.html(f'<div class="section-banner"><div class="section-banner-title">{title}</div><span class="section-banner-pill">{label}</span></div>')


def review_navigation(has_pose):
    pose_link = '<a href="#pose-frame">Pose frame</a>' if has_pose else ''
    st.html(f'''<nav class="cv-review-nav" aria-label="Analysis sections">
    <strong>Your innings review</strong><a href="#shot-review">Shot breakdown</a>
    <a href="#movement-review">Movement</a><a href="#practice-drills">Drills</a>
    {pose_link}<a href="#slow-replay">Slow replay</a><a href="#practice">Back to clip ↑</a></nav>''')


def review_anchor(name):
    st.html(f'<div id="{name}" class="cv-anchor" tabindex="-1"></div>')


def render_field_notes():
    st.html('<div id="field-notes" class="cricket-divider"></div>')
    with st.expander("Field notes · Get more from your next net session"):
        st.markdown("**Frame the whole batter.** Keep the head, bat and feet visible, with a steady camera and good light. Use a short clip containing one stroke.")
        st.markdown("**Read the ball first.** Set the actual delivery line and length. Shot-selection feedback uses these settings; the pitch diagram illustrates your inputs.")
        st.markdown("**Take one focus into the nets.** Review the suggested checkpoints, choose one drill, then record another stroke from the same camera angle.")
        st.caption("Model confidence describes the predicted shot. Technique scores and 2D pose estimates are practice aids, not calibrated performance measurements.")
    with st.expander("Keyboard guide · Work comfortably from your laptop"):
        st.markdown("**Tab / Shift+Tab** move between controls. **Enter / Space** activate buttons. Use the arrow keys in dropdowns and radio groups.")
        st.markdown("**Replay:** focus the replay image with Tab, then use **Space** to play or pause, **← / →** to step through frames, and **Home / End** to jump to the first or last frame.")
        st.caption("The first keyboard link skips to the batting lab. After analysis, section links jump to results, drills, the pose frame, or slow replay. Reduced-motion preferences are respected.")
    st.html('<div class="cv-footer"><span>CricketVision / Made for time in the middle.</span><span>Watch closely. Practise purposefully. Play your game.</span></div>')
