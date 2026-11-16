"""Manual real-clip replay check: python check_pose_replay.py [path-to-clip]."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright
from pose_replay import prepare_replay, replay_html

if __name__ == '__main__':
    clip = sys.argv[1] if len(sys.argv) > 1 else 'user_test_cover_drive_3.mp4'
    payload = prepare_replay(clip, 'Right-Handed (RHB)')
    print('Replay frames:', len(payload['frames']), 'FPS:', payload['fps'], flush=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={'width': 780, 'height': 430})
        page.set_content(replay_html(payload))
        page.wait_for_function('!document.getElementById("play").disabled')
        page.click('#impact')
        page.screenshot(path=str(Path(__file__).with_name('pose-replay-preview.png')))
        page.click('#play')
        page.wait_for_function('document.getElementById("play").textContent === "Play"')
        assert page.locator('#scrub').input_value() == str(len(payload['frames']) - 1)
        browser.close()
    print('Real-clip replay and playback passed.')
