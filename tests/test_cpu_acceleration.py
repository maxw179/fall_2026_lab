"""Dense/ZoomFFT parity and deterministic CPU parallel operations."""
import unittest
from unittest.mock import patch
import numpy as np
from optimization._reconstruction_common import _derivative_geometry
from optimization.reconstruction import CPUReconstructionWorkspace
from utils.psf import Microscope, Arbitrary_Grid, _convolution_grid
from utils.zernike import Aberration


class CPUAccelerationTests(unittest.TestCase):
    def test_iteration_callback_is_observational_and_public(self):
        from optimization import reconstruction
        m = Microscope(1, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
        g = Arbitrary_Grid(.003, .003, 4, 5, 0, 0, 0)
        args = (m, g, np.random.default_rng(2).random((2, 4, 5)),
                [0, .001], [0], [[0, 4]], .01)
        expected, ref = reconstruction.optimize_aberration_3D(
            *args, backend='cpu', max_iterations=2, verbose=False, num_random_starts=0)
        events, stages = [], []

        def observe(event):
            events.append(event.copy())
            event['strengths'][:] = 100  # Callbacks cannot mutate optimizer coefficients.

        actual, info = reconstruction.optimize_aberration_3D(
            *args, backend='cpu', max_iterations=2, verbose=False, on_iteration=observe, on_progress=stages.append, num_random_starts=0)
        np.testing.assert_array_equal(actual.strengths, expected.strengths)
        self.assertEqual(info['loss'], ref['loss'])
        self.assertEqual(len(events), len(info['history']))
        self.assertEqual([e['iteration'] for e in events], list(range(len(events))))
        self.assertTrue(all(e['backend'] == 'cpu' and e['method'] == 'newton' for e in events))
        self.assertTrue(all(np.isfinite(e['scaled_gradient_inf']) for e in events))
        self.assertEqual(stages[0]['stage'], 'derivatives')
        self.assertTrue({'propagation', 'svd', 'second_derivatives', 'hessian_response'}
                        <= {e['stage'] for e in stages})

    def test_fields_match_dense_at_large_and_offset_grids(self):
        rng = np.random.default_rng(1)
        for size, shape, offset in [(16, (4, 5), .02), (64, (63, 64), 0),
                                    (128, (64, 65), -.003), (32, (2, 5), 0)]:
            m = Microscope(3, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, size)
            g = Arbitrary_Grid(.003, .004, *shape, offset, -.02, 0)
            dense = _derivative_geometry(m, g, [[0, 4]], 'vector', propagation='dense')[-1]
            zoom = _derivative_geometry(m, g, [[0, 4]], 'vector', propagation='zoom')[-1]
            pupil = rng.normal(size=(2, 3, size, size))+1j*rng.normal(size=(2, 3, size, size))
            target = dense(pupil)
            np.testing.assert_allclose(zoom(pupil), target, rtol=1e-10,
                                       atol=np.max(abs(target))*2e-12)
            self.assertEqual(zoom(pupil).dtype, np.complex128)

    def test_derivatives_sample_and_optimizer(self):
        for mode, order, shape, padding, k, p in [
                ('scalar', 1, (6, 5), 0, 2, 3),
                ('vector', 3, (5, 6), (1, 2), 3, 2)]:
            m = Microscope(order, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 32)
            g = Arbitrary_Grid(.003, .004, *shape, .01, -.02, 0)
            a = Aberration([[0, 4], [-2, 2]], [-.07, .03])
            rho = float(m.compute_PSF(_convolution_grid(g), a, mode)[2].sum()**2*.03)
            args = (m, g, np.random.default_rng(3).random((k, *shape)),
                    np.linspace(0, .001, k), np.linspace(0, .002, p), a.modes, rho)
            options = dict(mode=mode, padding=padding, frequency_batch_size=7)
            baseline = CPUReconstructionWorkspace(*args, propagation='dense', **options)
            accelerated = CPUReconstructionWorkspace(*args, propagation='zoom',
                                                      fft_workers=2, cpu_workers=2, **options)
            expected = baseline.derivatives(a.strengths)
            for actual, target in zip(accelerated.derivatives(a.strengths), expected):
                np.testing.assert_allclose(np.asarray(actual)/expected[0],
                                           np.asarray(target)/expected[0], rtol=2e-9, atol=2e-10)
            ref = baseline.estimate_sample(a.strengths)
            result = accelerated.estimate_sample(a.strengths)
            np.testing.assert_allclose(result.image_mask, ref.image_mask,
                                       rtol=2e-9, atol=np.max(abs(ref.image_mask))*2e-10)
            # Finite differences independently check the exact Hessian.
            delta = np.array([1e-5, 0])
            numerical = (accelerated.derivatives(a.strengths+delta)[1]
                         - accelerated.derivatives(a.strengths-delta)[1])/(2e-5)
            np.testing.assert_allclose(numerical/expected[0], expected[2][:, 0]/expected[0],
                                       rtol=2e-5, atol=2e-6)
            ref, reference_info = baseline.optimize(max_iterations=3, num_random_starts=0)
            actual, info = accelerated.optimize(max_iterations=3, num_random_starts=0)
            np.testing.assert_allclose(actual.strengths, ref.strengths, rtol=2e-8, atol=2e-9)
            self.assertEqual(info['success'], reference_info['success'])
            self.assertIsNotNone(info['hessian'])
            points = [[0, 0], [-.07, .03], [0, 0], [.1, -.03]]
            expected_losses = baseline.losses(points)
            accelerated.clear_cache()
            actual_losses = accelerated.losses(points, workers=2)
            np.testing.assert_allclose(actual_losses, expected_losses, rtol=2e-9)
            self.assertIsNone(accelerated._last_fit)
            starts = [[0, 0], [-.07, .03]]
            expected_runs = baseline.optimize_starts(starts, max_iterations=2, num_random_starts=0)
            actual_runs = accelerated.optimize_starts(starts, workers=2, max_iterations=2, num_random_starts=0)
            for (ref, expected_info), (actual, info) in zip(expected_runs, actual_runs):
                np.testing.assert_allclose(actual.strengths, ref.strengths, rtol=2e-8, atol=2e-9)
                self.assertEqual(info['success'], expected_info['success'])
                self.assertEqual(info['method'], 'newton')

    def test_validation(self):
        m = Microscope(1, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
        g = Arbitrary_Grid(.003, .003, 4, 5, 0, 0, 0)
        args = (m, g, np.zeros((1, 4, 5)), [0], [0], [[0, 4]], .01)
        for options in [dict(fft_workers=0), dict(cpu_workers=0), dict(propagation='bad')]:
            with self.assertRaises(ValueError):
                CPUReconstructionWorkspace(*args, **options)


if __name__ == '__main__':
    unittest.main()
