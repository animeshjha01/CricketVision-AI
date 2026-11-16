"""Check responsive layout and the real upload preview against a running local app."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent

with sync_playwright() as p:
    browser = p.chromium.launch()
    for name, width, height in [('desktop', 1440, 900), ('mobile', 390, 844)]:
        page = browser.new_page(viewport={'width': width, 'height': height})
        page.goto('http://localhost:8502')
        page.locator('.cv-footer').wait_for(timeout=60000)
        assert page.locator('.cv-art img').evaluate('(e) => e.complete && e.naturalWidth > 0')
        page.screenshot(path=str(ROOT / f'frontend-{name}.png'))
        page.get_by_text('Upload your clip', exact=True).click()
        page.locator('input[type=file]').set_input_files(str(ROOT / 'user_test_cover_drive_3.mp4'))
        page.locator('.cv-session').wait_for(timeout=30000)
        video = page.locator('video')
        video.wait_for()
        page.wait_for_function('document.querySelector("video").readyState >= 2', timeout=30000)
        box = video.bounding_box()
        print(name, 'video bounds after upload:', box, flush=True)
        print(name, 'video element:', video.evaluate('(e) => e.outerHTML.slice(0, 350)'), flush=True)
        assert 180 <= box['height'] <= 281, box
        assert box['y'] >= 0 and box['y'] + box['height'] <= height, box
        assert video.evaluate('(e) => getComputedStyle(e).objectFit') == 'contain'
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        assert not page.locator('[data-testid=stException]').count()
        page.screenshot(path=str(ROOT / f'frontend-{name}-upload.png'))
        page.close()
    browser.close()
