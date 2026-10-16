"""Check cumulative demo exports and offline replay on a small physical scene."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np

from optimization import reconstruction
from systematic_testing.gpu_demo.run_demo import (
    DemoConfig, MODES, coefficients, make_scene, run_demo)
from systematic_testing.gpu_demo.plot_demo import load_run, make_replay, make_plots, plot_rms
from systematic_testing.gpu_demo.replay import load_gpu_case
from utils import zernike

try:
    from optimization.reconstruction_cuda import _cupy
    _cupy()
    CUDA_AVAILABLE = True
except RuntimeError:
    if os.environ.get('REQUIRE_CUDA') == '1':
        raise
    CUDA_AVAILABLE = False


class DemoChecks:
    backend = 'cpu'

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)/'demo'
        cls.config = DemoConfig(backend=cls.backend, size=8, pupil_size=12,
            sample_planes=9, reconstruction_planes=3, random_starts=2, max_iterations=1,
            object_stride=2, save_objects=True)
        with contextlib.redirect_stdout(io.StringIO()):
            run_demo(cls.config, cls.output)
        cls.metadata, cls.scene, cls.history = load_run(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_cumulative_noise_and_known_slm_phases_are_preserved(self):
        _, _, _, _, _, _, schedule = make_scene(self.config)
        prior_images, known_phases = [], []
        for loop in range(1, 4):
            with np.load(self.output/f'loop_{loop}_input.npz', allow_pickle=False) as saved:
                images = saved['images']
                current = saved['acquisition_slm_strengths']
                self.assertEqual(len(images), 36*loop)
                if prior_images:
                    np.testing.assert_array_equal(images[:-36], np.concatenate(prior_images))
                prior_images.append(images[-36:])
                probes = np.array([coefficients(probe) for _ in range(4) for probe in schedule[loop-1]])
                known_phases.append(probes+current)
                np.testing.assert_allclose(saved['diversity_strengths']+current,
                    np.concatenate(known_phases), atol=2e-16, rtol=2e-15)
        self.assertEqual(self.metadata['status'], 'complete')
        self.assertEqual([loop['fit_images'] for loop in self.metadata['loops']], [36, 72, 108])

    def test_core_timestamps_rms_and_correction_boundaries(self):
        h = self.history
        self.assertAlmostEqual(h['applied_residual_rms'][0], .20, places=12)
        self.assertTrue(np.all(np.diff(h['core_elapsed_seconds']) >= 0))
        self.assertAlmostEqual(self.metadata['core_seconds'], sum(
            loop['core_seconds'] for loop in self.metadata['loops']), places=12)
        for loop in range(1, 4):
            selected = h['loop'] == loop
            self.assertTrue(selected.any())
            self.assertTrue(np.all(h['core_elapsed_seconds'][selected] >= h['loop_end_core_seconds'][loop-1]))
            self.assertTrue(np.all(h['core_elapsed_seconds'][selected] <= h['loop_end_core_seconds'][loop]))
            final = np.flatnonzero(selected)[-1]
            expected = h['applied_slm_strengths'][loop-1]-np.clip(h['strengths'][final], -.1, .1)
            np.testing.assert_allclose(h['applied_slm_strengths'][loop], expected, atol=1e-15)
            self.assertLessEqual(self.metadata['loops'][loop-1]['screening_seconds'],
                                 self.metadata['loops'][loop-1]['optimization_seconds'])
        np.testing.assert_allclose(h['static_estimate_strengths'],
                                   h['strengths']-h['acquisition_slm_strengths'])
        for i in range(len(h['strengths'])):
            error = zernike.zernike_RMS_difference(
                zernike.Aberration(MODES, h['static_estimate_strengths'][i]),
                zernike.Aberration(MODES, self.scene['truth_strengths']), float(self.scene['alpha']))
            self.assertAlmostEqual(h['rms_error'][i], error, places=12)

    def test_saved_final_object_matches_independent_cpu_refit(self):
        microscope, grid, _, _, _, _, _ = make_scene(self.config)
        with np.load(self.output/'loop_3_input.npz', allow_pickle=False) as saved:
            sample = reconstruction.estimate_sample_3D(microscope, grid,
                saved['images'], saved['focal_z'], saved['sample_z'],
                zernike.Aberration(MODES, self.history['strengths'][-1]), self.config.rho,
                diversities=[zernike.Aberration(MODES, s) for s in saved['diversity_strengths']],
                backend='cpu')
        objects = np.load(self.output/'object_history.npy', allow_pickle=False)
        self.assertEqual(objects.dtype, np.float32)
        np.testing.assert_allclose(objects[-1], sample.image_mask, rtol=3e-6,
                                   atol=3e-7*float(abs(sample.image_mask).max()))
        with np.load(self.output/'object_frames.npz', allow_pickle=False) as frames:
            for loop in range(1, 4):
                final = np.flatnonzero(self.history['loop'] == loop)[-1]
                self.assertIn(final, frames['history_indices'])

    def test_offline_plots_and_image_wavefront_replay_never_refit(self):
        from utils import psf
        with patch.object(reconstruction, 'estimate_sample_3D', side_effect=AssertionError('Unexpected local refit')), \
             patch.object(psf.Microscope, 'compute_image_3D', side_effect=AssertionError('Unexpected local simulation')):
            output = make_plots(self.output, use_tex=False)
            self.assertTrue((output/'rms_time.png').exists())
            self.assertTrue((output/'acquired_images.png').exists())
        with patch.object(reconstruction, 'estimate_sample_3D', side_effect=AssertionError('Unexpected local refit')), \
             patch.object(psf.Microscope, 'compute_image_3D') as forward:
            forward.side_effect = lambda *args, **kwargs: (self.scene['x'], self.scene['y'],
                np.asarray(kwargs['focal_z_levels']), np.ones((1, 8, 8)))
            viewer = make_replay(self.output, stride=20, use_tex=False, show_progress=False)
            self.assertEqual(forward.call_count, 4)
            self.assertIn('<iframe', viewer.data)
            document = (output/'ao_replay.html').read_text()
            self.assertIn('const frames=', document)
            self.assertIn('id="seek"', document)
            self.assertNotIn('voxels', document)

    def test_gpu_history_replay_preserves_all_steps_slm_frames_and_times(self):
        frames, points, _ = load_gpu_case(self.output)
        self.assertEqual(len(frames), len(self.history['iteration'])+4)
        self.assertEqual([p['images_taken'] for p in points], [0, 36, 72, 108])
        fit_frames = [f for f in frames if 'optimization_rms' in f]
        np.testing.assert_allclose([f['optimization_rms'] for f in fit_frames], self.history['rms_error'])
        np.testing.assert_array_equal([f['core_elapsed_seconds'] for f in fit_frames],
                                      self.history['core_elapsed_seconds'])
        for loop in range(1, 4):
            selected = [f for f in fit_frames if f['loop'] == loop]
            for frame in selected:
                np.testing.assert_array_equal(frame['image'], selected[0]['image'])
        final = frames[-1]
        self.assertAlmostEqual(final['rms'], self.metadata['final_residual_rms'], places=12)
        self.assertEqual(final['core_elapsed_seconds'], self.history['loop_end_core_seconds'][-1])

    def test_rms_figure_keeps_shared_sweep_style(self):
        import matplotlib.pyplot as plt
        fig, ax = plot_rms(self.metadata, self.scene, self.history)
        np.testing.assert_array_equal(fig.get_size_inches(), [6, 6])
        self.assertEqual(fig.dpi, 300)
        self.assertTrue(ax.xaxis.label.get_usetex())
        self.assertAlmostEqual(ax.xaxis.label.get_fontsize(), 10.8)  # bmh scales axis labels by 1.2.
        self.assertEqual(ax.get_ylabel(), 'RMS error [waves]')
        self.assertEqual(ax.get_title(), '')
        self.assertEqual(ax.get_legend().get_bbox_to_anchor()._bbox.y0, 1.16)
        plt.close(fig)

    def test_existing_run_is_not_overwritten(self):
        before = (self.output/'run.json').read_bytes()
        with self.assertRaises(FileExistsError):
            run_demo(self.config, self.output)
        self.assertEqual(before, (self.output/'run.json').read_bytes())


class PortableDemoTests(DemoChecks, unittest.TestCase):
    pass


@unittest.skipUnless(CUDA_AVAILABLE, 'No CUDA device')
class CUDADemoTests(DemoChecks, unittest.TestCase):
    backend = 'cuda'


if __name__ == '__main__':
    unittest.main()
