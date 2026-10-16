"""Shared-zero camera acquisitions survive export and replay without denoising."""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from systematic_testing.gpu_demo.run_demo import DemoConfig, run_demo
from systematic_testing.gpu_demo.replay import load_acquisition_preview, load_gpu_case


class CameraDemoTests(unittest.TestCase):
    def test_prepared_frames_match_full_run_and_cumulative_fit(self):
        config = DemoConfig(backend='cpu', size=8, pupil_size=12,
                            sample_planes=5, reconstruction_planes=3,
                            random_starts=0, max_iterations=1,
                            shared_zero=True, diversity_settings=(-.1, -.05, .05, .1),
                            read_sigma_e=3., background_e=3., checkerboard_e=3.)
        with tempfile.TemporaryDirectory() as directory:
            preview, full = [Path(directory)/name for name in ('preview', 'full')]
            run_demo(config, preview, prepare_only=True)
            meta = run_demo(config, full)
            self.assertEqual([x['fit_images'] for x in meta['loops']], [52, 104, 156])
            frames, _, _ = load_acquisition_preview(preview)
            with np.load(preview/'loop_1_input.npz') as expected, np.load(full/'loop_1_input.npz') as actual:
                np.testing.assert_array_equal(actual['images'], expected['images'])
                self.assertEqual(actual['images'].shape, (52, 8, 8))
                for i, frame in enumerate(frames):
                    np.testing.assert_array_equal(frame['image'], actual['images'][i])
                first = actual['images'].copy()
            with np.load(full/'loop_3_input.npz') as saved:
                np.testing.assert_array_equal(saved['images'][:52], first)
            frames, _, _ = load_gpu_case(full)
            with np.load(full/'correction_preview.npz') as saved:
                for frame in frames:
                    np.testing.assert_array_equal(frame['image'], saved['images'][int(np.argmin(abs(np.asarray(meta['focal_positions_mm'])))), frame['progress']])


if __name__ == '__main__':
    unittest.main()
