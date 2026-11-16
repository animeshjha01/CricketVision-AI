"""UI regression checks; inference is stubbed to keep these focused on interactions."""
import unittest
from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest


class PracticeWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=60)
        self.assertFalse(self.app.exception)

    def button(self, label):
        return next(b for b in self.app.button if b.label == label)

    def test_presets_and_sample_delivery(self):
        self.button('Short at the body').click().run()
        self.assertEqual(self.app.selectbox(key='sb_len').value, 'Short')
        self.assertEqual(self.app.selectbox(key='sb_line').value, 'Stumps')
        self.button("Use this sample's delivery").click().run()
        self.assertEqual(self.app.selectbox(key='sb_len').value, 'Full')
        self.assertEqual(self.app.selectbox(key='sb_line').value, 'Outside Off')
        self.assertFalse(self.app.exception)

    def test_upload_empty_state(self):
        self.app.radio[0].set_value('Upload your clip').run()
        self.assertTrue(self.button('Analyse my batting  →').disabled)
        self.assertFalse(self.app.exception)

    def test_rerun_refreshes_cached_frontend(self):
        # Reproduce a long-running server retaining the previous no-argument API.
        with patch('frontend.render_header', lambda: None):
            self.app.run()
            self.assertFalse(self.app.exception)
            self.assertTrue(any('Own the crease.' in item.proto.body
                                for item in self.app.get('html')))

    def test_failed_analysis_has_error_status(self):
        with patch('predict.analyze_batting_video', return_value={'error': 'No batter detected.'}):
            self.button('Analyse my batting  →').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.status[0].state, 'error')
        self.assertTrue(any('No batter detected.' in item.value for item in self.app.error))

    def test_review_and_stale_result(self):
        import numpy as np
        result = dict(best_core_key='cover_drive', pose_seq=np.ones((30, 34)),
                      shot_played='Cover Drive', shot_family='Drive', confidence=85.0,
                      confidence_str='85%', execution_accuracy=75.0,
                      execution_accuracy_str='75%', is_correct_selection=True,
                      ideal_shots=['Cover Drive'], vis_frame=None,
                      top3=[{'name': 'Cover Drive', 'pct': 85.0}])
        with patch('predict.analyze_batting_video', return_value=result):
            self.button('Analyse my batting  →').click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(self.app.session_state['last_result']['shot_played'], 'Cover Drive')
        self.assertTrue(any('03 / Your shot breakdown' in item.proto.body for item in self.app.get('html')))
        self.app.selectbox(key='sb_handedness').set_value('Left-Handed (LHB)').run()
        self.assertFalse(any('03 / Your shot breakdown' in item.proto.body for item in self.app.get('html')))
        self.assertFalse(self.app.exception)


if __name__ == '__main__':
    unittest.main()
