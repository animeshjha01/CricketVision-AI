"""On-demand visual replay. Does not modify training data or model features."""
import base64
import json

import cv2
import numpy as np

EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
         (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def make_payload(frames, selection, boxes, points, confidence, fps, impact_local, mirrored=False):
    """Project crop-relative points onto the exact sampled image; omit unseen joints."""
    if not frames or not np.isfinite(fps) or fps <= 0:
        raise ValueError('The clip has no usable replay frames.')
    items, seen = [], set()
    height, width = frames[0].shape[:2]
    impact_frame = int(selection[impact_local])
    for local, source in enumerate(selection):
        source = int(source)
        if source in seen:
            continue
        seen.add(source)
        frame = frames[source]
        x1, y1, x2, y2 = boxes[local]
        joints = []
        for joint, score in zip(points[local], confidence[local]):
            x = float(x1 + joint[0] * (x2 - x1))
            y = float(y1 + joint[1] * (y2 - y1))
            if mirrored:
                x = width - 1 - x
            joints.append([x / width, y / height] if score > .30 and np.isfinite([x, y]).all()
                          and 0 <= x < width and 0 <= y < height else None)
        if mirrored:
            frame = cv2.flip(frame, 1)
        scale = min(1.0, 720 / max(width, height))
        frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise ValueError('A replay frame could not be prepared.')
        items.append(dict(image=base64.b64encode(jpg).decode('ascii'), joints=joints,
                          time=(source - int(selection[0])) / fps, source=source))
    if not items or not any(any(j is not None for j in item['joints']) for item in items):
        raise ValueError('No confident batter pose was found for replay. Try a clearer clip.')
    return dict(frames=items, fps=float(fps), edges=EDGES,
                impact=min(range(len(items)), key=lambda i: abs(items[i]['source'] - impact_frame)),
                start=float(selection[0]) / fps)


def prepare_replay(video_path, handedness, on_progress=None):
    # Lazy imports reuse the application's loaded models. No training or dataset writes.
    from extract_data import (pose_model, YOLO_DEV, is_left_handed, _select_window, _crop_pose, CROP_PX)
    from striker_tracker import read_frames, locate_striker, smooth_boxes, crop_box_for

    mirrored = is_left_handed(handedness)
    if on_progress:
        on_progress("Reading and sampling the selected clip…")
    frames, fps = read_frames(video_path, mirror=mirrored)
    if len(frames) < 5:
        raise ValueError('The clip is too short to prepare a pose replay.')
    if on_progress:
        on_progress("Locating the batter and estimated shot moment…")
    tracked = locate_striker(pose_model, frames, YOLO_DEV)
    if tracked is None or np.all(np.isnan(tracked['boxes'][:, 0])):
        raise ValueError('The batter could not be tracked reliably for replay.')
    height, width = frames[0].shape[:2]
    selection, impact = _select_window(len(frames), int(tracked['impact_t']), tracked.get('seg'))
    tracked_boxes = smooth_boxes(tracked['boxes'])
    boxes = [crop_box_for(tracked_boxes[i], width, height) for i in selection]
    crops = []
    for i, (x1, y1, x2, y2) in zip(selection, boxes):
        crop = frames[i][y1:y2, x1:x2]
        if not crop.size:
            raise ValueError('The batter moved outside the usable replay frame.')
        crops.append(cv2.resize(crop, (CROP_PX, CROP_PX)))
    if on_progress:
        on_progress("Estimating pose points in the shot window…")
    points, confidence = _crop_pose(crops)
    if on_progress:
        on_progress("Building the frame-by-frame replay…")
    return make_payload(frames, selection, boxes, points, confidence, fps, impact, mirrored)


def replay_html(payload):
    data = json.dumps(payload, allow_nan=False).replace('<', '\\u003c')
    return """<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    *{box-sizing:border-box}body{margin:0;background:#101c18;color:#f2f0e5;font:13px Arial,sans-serif}
    .player{border:1px solid #48604e;border-radius:12px;padding:12px;max-width:780px;margin:auto}
    canvas{display:block;width:100%;height:270px;object-fit:contain;background:#050b08;border-radius:7px}
    .controls{display:flex;align-items:center;flex-wrap:wrap;gap:8px;margin-top:10px}
    button,select{border:1px solid #536a56;border-radius:5px;background:#20372a;color:#f2f0e5;padding:8px;cursor:pointer}
    button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #d6e7a3;outline-offset:2px}
    #play{background:#d6e7a3;color:#182719;min-width:55px}input[type=range]{width:100%;accent-color:#d6e7a3}
    .status{display:flex;justify-content:space-between;gap:8px;font-size:11px;color:#b4c6b7;margin:9px 0 4px}
    @media(max-width:500px){canvas{height:230px}.player{padding:9px}.controls{gap:6px}button,select{padding:8px 6px}}
    </style></head><body><div class="player">
    <canvas id="screen" tabindex="0" aria-label="Batting replay. Space plays or pauses. Left and right arrows step through frames. Home and End jump to the first and last frame." aria-keyshortcuts="Space ArrowLeft ArrowRight Home End"></canvas>
    <div class="status"><span id="status" role="status">Loading replay…</span><span id="time"></span></div>
    <input id="scrub" type="range" min="0" value="0" step="1" aria-label="Replay frame" disabled>
    <div class="controls">
      <button id="play" disabled>Play</button>
      <button id="prev" aria-label="Previous frame" disabled>← Frame</button>
      <button id="next" aria-label="Next frame" disabled>Frame →</button>
      <label>Speed <select id="speed"><option value="0.25">¼×</option><option value="0.5" selected>½×</option><option value="1">1×</option></select></label>
      <label><input id="overlay" type="checkbox" checked> Pose overlay</label>
      <button id="impact" disabled>Shot moment</button>
    </div><div style="font-size:11px;color:#b4c6b7;margin-top:8px">Focus the image: Space to play/pause · ← → to step · Home / End to jump</div></div>
    <script>
    const data=__PAYLOAD__, frames=data.frames, canvas=document.getElementById('screen'), ctx=canvas.getContext('2d');
    const $=id=>document.getElementById(id), images=[];
    let index=0, playing=false, elapsed=0, last=0, animation=0;
    $('scrub').max=frames.length-1;
    function draw(){
      const img=images[index];canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;
      ctx.drawImage(img,0,0);const joints=frames[index].joints;
      if($('overlay').checked){
        ctx.strokeStyle='#a5ff73';ctx.lineWidth=Math.max(2,canvas.width/250);
        for(const [a,b] of data.edges){if(joints[a]&&joints[b]){ctx.beginPath();ctx.moveTo(joints[a][0]*canvas.width,joints[a][1]*canvas.height);ctx.lineTo(joints[b][0]*canvas.width,joints[b][1]*canvas.height);ctx.stroke();}}
        ctx.fillStyle='#ffcf53';for(const point of joints){if(point){ctx.beginPath();ctx.arc(point[0]*canvas.width,point[1]*canvas.height,Math.max(3,canvas.width/180),0,2*Math.PI);ctx.fill();}}
      }
      $('scrub').value=index;$('status').textContent='Frame '+(index+1)+' / '+frames.length+(index===data.impact?' · Estimated shot moment':'');
      $('status').setAttribute('aria-live',playing?'off':'polite');
      $('time').textContent=(data.start+frames[index].time).toFixed(2)+'s in clip';
      $('prev').disabled=index===0;$('next').disabled=index===frames.length-1;
    }
    function pause(){playing=false;cancelAnimationFrame(animation);$('play').textContent='Play';}
    function seek(i){pause();index=Math.max(0,Math.min(frames.length-1,i));elapsed=frames[index].time;draw();}
    function tick(now){
      if(!playing)return;
      elapsed+=(now-last)/1000*Number($('speed').value);last=now;
      let next=index;while(next<frames.length-1&&frames[next+1].time<=elapsed)next++;
      if(next!==index){index=next;draw();}
      if(elapsed>=frames[frames.length-1].time+1/data.fps){pause();return;}
      animation=requestAnimationFrame(tick);
    }
    $('play').onclick=()=>{if(playing){pause();return;}if(index===frames.length-1)seek(0);playing=true;$('play').textContent='Pause';last=performance.now();animation=requestAnimationFrame(tick);};
    $('prev').onclick=()=>seek(index-1);$('next').onclick=()=>seek(index+1);
    $('impact').onclick=()=>seek(data.impact);$('scrub').oninput=e=>seek(Number(e.target.value));
    $('overlay').onchange=()=>{if(images.length===frames.length)draw();};
    canvas.addEventListener('keydown',event=>{
      if(images.length!==frames.length)return;
      if([' ','ArrowLeft','ArrowRight','Home','End'].includes(event.key))event.preventDefault();
      if(event.key===' ')$('play').click();
      if(event.key==='ArrowLeft')seek(index-1);
      if(event.key==='ArrowRight')seek(index+1);
      if(event.key==='Home')seek(0);
      if(event.key==='End')seek(frames.length-1);
    });
    document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
    Promise.all(frames.map(frame=>new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src='data:image/jpeg;base64,'+frame.image;})))
      .then(loaded=>{images.push(...loaded);$('play').disabled=false;$('impact').disabled=false;$('scrub').disabled=false;draw();})
      .catch(()=>{$('status').textContent='Replay could not load. Prepare it again.';});
    </script></body></html>""".replace('__PAYLOAD__', data)
