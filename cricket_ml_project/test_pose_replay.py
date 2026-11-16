"""Replay geometry and browser controls without models or dataset changes."""
import unittest
import numpy as np
from pose_replay import make_payload, replay_html


def fixture(mirrored=False):
    frames = [np.full((100, 200, 3), n * 60, dtype=np.uint8) for n in range(3)]
    points = np.full((3, 17, 2), .5)
    confidence = np.ones((3, 17))
    confidence[:, 0] = .1
    return make_payload(frames, [0, 1, 2], [(20, 10, 100, 90)] * 3,
                        points, confidence, 10, 1, mirrored)


class ReplayTests(unittest.TestCase):
    def test_projection_and_confidence(self):
        payload = fixture()
        self.assertEqual(payload['frames'][0]['joints'][1], [.3, .5])
        self.assertIsNone(payload['frames'][0]['joints'][0])
        self.assertEqual(payload['impact'], 1)
        self.assertEqual(payload['frames'][2]['time'], .2)

    def test_left_handed_projection(self):
        self.assertEqual(fixture(True)['frames'][0]['joints'][1], [139 / 200, .5])

    def test_duplicate_frames_keep_source_timing(self):
        frames = [np.zeros((10, 10, 3), np.uint8)] * 2
        payload = make_payload(frames, [0, 0, 1], [(0, 0, 10, 10)] * 3,
                               np.ones((3, 17, 2)) * .5, np.ones((3, 17)), 5, 2)
        self.assertEqual(len(payload['frames']), 2)
        self.assertEqual(payload['frames'][1]['time'], .2)
        self.assertEqual(payload['impact'], 1)

    def test_browser_controls(self):
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={'width': 390, 'height': 500})
            page.set_content(replay_html(fixture()))
            page.wait_for_function('!document.getElementById("play").disabled')
            page.focus('#screen')
            page.keyboard.press('End')
            self.assertIn('Frame 3 / 3', page.locator('#status').inner_text())
            page.keyboard.press('Home')
            page.keyboard.press('ArrowRight')
            self.assertIn('Frame 2 / 3', page.locator('#status').inner_text())
            page.keyboard.press('ArrowLeft')
            page.click('#next')
            self.assertIn('Frame 2 / 3', page.locator('#status').inner_text())
            page.click('#prev')
            page.click('#impact')
            self.assertIn('Estimated shot moment', page.locator('#status').inner_text())
            page.uncheck('#overlay')
            page.select_option('#speed', '0.25')
            page.focus('#screen')
            page.keyboard.press('Space')
            page.wait_for_function('document.getElementById("play").textContent === "Play"')
            self.assertIn('Frame 3 / 3', page.locator('#status').inner_text())
            self.assertFalse(page.evaluate('document.documentElement.scrollWidth > innerWidth'))
            self.assertLess(page.locator('.player').bounding_box()['height'], 430)
            browser.close()


if __name__ == '__main__':
    unittest.main()
