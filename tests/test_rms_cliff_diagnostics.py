"""Check saved-acquisition replay in the changing SLM coordinate frame."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from systematic_testing.rms_cliff_sweep.diagnose_failures import diagnose, replay_workspace
from systematic_testing.rms_cliff_sweep.run_sweep import SweepConfig, run_case


class DiagnosticReplay(unittest.TestCase):
    def test_all_three_replayed_losses_match_recorded_fits_after_slm_updates(self):
        config = SweepConfig(rms_values=(.3,), trials=1, size=8, pupil_size=12,
            sample_planes=9, reconstruction_planes=3, random_starts=1,
            max_iterations=1, backend='cpu')
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            run_case(config, root, 0)
            case = root/'case_0000'
            meta = json.loads((case/'run.json').read_text())
            for loop in (1, 2, 3):
                workspace, truth, fitted, _, _, inputs = replay_workspace(case, loop)
                self.assertEqual(len(inputs['images']), 36*loop)
                self.assertAlmostEqual(workspace.loss(fitted)/meta['loops'][loop-1]['loss'], 1., places=12)
                with np.load(case/f'acquisition_{loop}.npz') as acquired, np.load(case/'results.npz') as saved:
                    np.testing.assert_allclose(truth, saved['truth_strengths']+acquired['acquisition_slm_strengths'])
                row, profiles, trace, controls = diagnose(case, loop, profile=False)
                self.assertLess(row['saved_loss_relative_replay_error'], 1e-12)
                self.assertTrue(np.isfinite(row['fit_over_truth']))
                self.assertTrue(trace)
                self.assertEqual(profiles, [])
                self.assertEqual(controls, [])


if __name__ == '__main__':
    unittest.main()
