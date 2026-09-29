"""Portable numerical tests plus the same checks on CUDA when available."""
import os
import unittest
from unittest.mock import patch
import numpy as np
from optimization import reconstruction as cpu
from optimization import reconstruction_cuda as gpu
from utils.psf import Microscope, Arbitrary_Grid, _convolution_grid
from utils.zernike import Aberration


def cuda_backend():
    try:
        return gpu._cupy()
    except (RuntimeError, ImportError):
        if os.environ.get('REQUIRE_CUDA') == '1':
            raise
        return None


CP = cuda_backend()


class NumericalChecks:
    def test_parity(self):
        for mode, order, shape, padding, k, p in [
            ('scalar', 1, (4, 5), 0, 2, 3),
            ('vector', 3, (5, 4), (1, 2), 3, 2),
            ('vector', 1, (4, 4), None, 2, 2),
        ]:
            with self.subTest(mode=mode, shape=shape):
                m = Microscope(order, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
                grid = Arbitrary_Grid(.003, .004, *shape, .02, -.03, 0)
                a = Aberration([[0, 4], [-2, 2]], [-.07, .03])
                focal, planes = np.linspace(0, .001, k), np.linspace(0, .002, p)
                diversities = [Aberration([[0, 4]], [x]) for x in np.linspace(-.1, .1, k)]
                data = np.random.default_rng(3).random((k, *shape))
                rho = float(m.compute_PSF(_convolution_grid(grid), a, mode)[2].sum()**2*.03)
                args = (m, grid, data, focal, planes)
                expected = cpu.evaluate_loss_derivatives_3D(*args, a, rho, diversities, mode, 7, padding)
                for psf_batch, freq_batch in [(1, 7), (4, 1000)]:
                    w = gpu.CUDAReconstruction(*args, a.modes, rho, diversities, mode,
                                               freq_batch, padding, psf_batch_size=psf_batch,
                                               _xp=self.xp)
                    actual = w.derivatives(a.strengths)
                    for x, y in zip(actual, expected):
                        np.testing.assert_allclose(np.asarray(x)/expected[0], np.asarray(y)/expected[0],
                                                   rtol=2e-8, atol=2e-9)
                    sample, info = w.sample(a.strengths, True)
                    ref, ref_info = cpu.estimate_sample_3D(*args, a, rho, diversities, mode, 7, padding, True)
                    np.testing.assert_allclose(sample.image_mask, ref.image_mask, rtol=2e-8, atol=1e-35)
                    for key in ('loss', 'relative_residual', 'measured_relative_residual'):
                        np.testing.assert_allclose(info[key], ref_info[key], rtol=2e-9)
                    delta = np.array([1e-5, 0.])
                    numerical = (w.loss(a.strengths+delta)-w.loss(a.strengths-delta))/(2e-5)
                    np.testing.assert_allclose(numerical/expected[0], actual[1][0]/expected[0], rtol=2e-5, atol=1e-7)
                    plus = w.derivatives(a.strengths+delta)[1]
                    minus = w.derivatives(a.strengths-delta)[1]
                    np.testing.assert_allclose((plus-minus)/(2e-5*expected[0]), actual[2][:, 0]/expected[0], rtol=2e-5, atol=1e-6)

    def test_device_output_zero_data_and_public_api(self):
        m = Microscope(1, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
        g = Arbitrary_Grid(.003, .003, 4, 5, 0, 0, 0)
        a = Aberration([[0, 4]], [0.])
        data = self.xp.zeros((1, 4, 5))
        with patch.object(gpu, '_cupy', return_value=self.xp):
            volume, info = gpu.estimate_sample_3D(
                m, g, data, [0], [0], a, .01, return_info=True, return_device=True)
            self.assertIsInstance(volume, self.xp.ndarray)
            self.assertEqual(volume.shape, (1, 4, 5))
            self.assertEqual(info['loss'], 0.)
            self.assertEqual(info['relative_residual'], 0.)
            self.assertEqual(info['measured_relative_residual'], 0.)
            self.assertEqual(gpu.evaluate_loss_3D(m, g, data, [0], [0], a, .01), 0.)
            loss, gradient, hessian = gpu.evaluate_loss_derivatives_3D(
                m, g, data, [0], [0], a, .01)
            self.assertEqual(loss, 0.)
            np.testing.assert_array_equal(gradient, [0.])
            np.testing.assert_array_equal(hessian, [[0.]])
            estimated, result = gpu.optimize_aberration_3D(
                m, g, data, [0], [0], a.modes, .01, verbose=False)
            self.assertTrue(result['success'])
            np.testing.assert_array_equal(estimated.strengths, [0.])

    def test_optimizer_matches_cpu(self):
        m = Microscope(1, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
        g = Arbitrary_Grid(.003, .003, 5, 5, 0, 0, 0)
        data = np.random.default_rng(5).random((2, 5, 5))
        modes = [[0, 4]]
        div = [Aberration(modes, [v]) for v in [-.1, .1]]
        args = (m, g, data, [0, .001], [0], modes, .01, div)
        expected, info = cpu.optimize_aberration_3D(*args, mode='scalar', max_iterations=3, verbose=False)
        w = gpu.CUDAReconstruction(*args, mode='scalar', _xp=self.xp)
        actual, other = w.optimize(max_iterations=3, verbose=False)
        np.testing.assert_allclose(actual.strengths, expected.strengths, rtol=1e-7, atol=1e-9)
        self.assertEqual(other['success'], info['success'])
        np.testing.assert_allclose(other['loss'], info['loss'], rtol=1e-9)


class PortableTests(NumericalChecks, unittest.TestCase):
    xp = np

    def test_invalid_inputs(self):
        m = Microscope(1, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
        g = Arbitrary_Grid(.003, .003, 4, 5, 0, 0, 0)
        kwargs = dict(microscope=m, grid=g, images=np.zeros((1, 4, 5)),
                      focal_z_levels=[0], sample_z_levels=[0], modes=[[0, 4]],
                      rho=.01, _xp=np)
        for override in [dict(rho=0), dict(padding=-1), dict(mode='bad'),
                         dict(frequency_batch_size=0), dict(psf_batch_size=0),
                         dict(sample_z_levels=[0, 0]), dict(modes=[[1, 2]]),
                         dict(images=np.zeros((2, 4, 5))), dict(diversities=[])]:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    gpu.CUDAReconstruction(**(kwargs | override))

    def test_no_implicit_cpu_fallback(self):
        with patch('optimization.reconstruction_cuda._cupy', side_effect=RuntimeError('no GPU')):
            with self.assertRaisesRegex(RuntimeError, 'no GPU'):
                gpu.CUDAReconstruction(None, None, None, [0], [0], [[0, 0]], 1)


@unittest.skipIf(CP is None, 'CUDA/CuPy unavailable')
class CUDATests(NumericalChecks, unittest.TestCase):
    xp = CP
