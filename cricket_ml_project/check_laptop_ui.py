"""Real-analysis laptop smoke check against a locally running app on port 8502."""
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1366, 'height': 768})
    page.goto('http://localhost:8502')
    page.locator('.cv-footer').wait_for(timeout=60000)
    page.locator('.cv-skip').focus()
    assert page.locator('.cv-skip').bounding_box()['y'] >= 0
    page.keyboard.press('Enter')
    assert page.evaluate('location.hash') == '#practice'
    page.get_by_role('button', name='Analyse my batting').click()
    page.locator('.cv-review-nav').wait_for(timeout=180000)
    page.get_by_role('button', name='Prepare pose replay').wait_for()
    page.wait_for_function('Array.from(document.querySelectorAll(".st-key-pose_review img")).every(img => img.complete)')
    assert not page.locator('[data-testid=stException]').count()
    for width, height in [(1366, 768), (1280, 720)]:
        page.set_viewport_size({'width': width, 'height': height})
        page.locator('.cv-review-nav a[href="#movement-review"]').click()
        expect(page.locator('#movement-review')).to_be_in_viewport()
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        assert page.locator('.cv-metric-card').count() == 4
        page.screenshot(path=str(Path(__file__).with_name(f'laptop-results-{width}.png')))
    page.emulate_media(reduced_motion='reduce')
    assert page.locator('.cv-review-nav').evaluate('(e)=>getComputedStyle(e).animationName') == 'none'
    page.locator('.cv-review-nav a[href="#slow-replay"]').click()
    print('Replay link:', page.evaluate('location.hash'), 'anchor:', page.locator('#slow-replay').bounding_box(), 'button:', page.get_by_role('button', name='Prepare pose replay').bounding_box(), flush=True)
    page.screenshot(path=str(Path(__file__).with_name('laptop-replay-navigation.png')))
    expect(page.get_by_role('button', name='Prepare pose replay')).to_be_in_viewport()
    print('Real analysis, result links, cards, keyboard skip link, reduced motion and both laptop sizes passed.')
    browser.close()
