"""Automatic selection and gradient parity without requiring CUDA locally."""
import unittest
from unittest.mock import patch
import numpy as np
from optimization import reconstruction as api
from optimization.reconstruction_cuda import CUDAReconstruction
from optimization._reconstruction_cpu import CPUReconstructionWorkspace
from utils.psf import Microscope, Arbitrary_Grid
from utils.zernike import Aberration


class BackendTests(unittest.TestCase):
    def setup_case(self):
        m = Microscope(1, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
        g = Arbitrary_Grid(.003, .003, 5, 4, 0, 0, 0)
        data = np.random.default_rng(9).random((2, 5, 4))
        modes = [[0, 4], [-2, 2]]
        div = [Aberration([[0, 4]], [v]) for v in [-.1, .1]]
        return (m, g, data, [0, .001], [0, .001, .002], modes, .01, div)

    def test_detection_and_forcing(self):
        with patch('optimization.reconstruction_cuda._cupy', side_effect=RuntimeError('unavailable')):
            self.assertEqual(api.get_backend(), 'cpu')
            with self.assertRaises(RuntimeError):
                api.get_backend('cuda')
        with patch('optimization.reconstruction_cuda._cupy', return_value=np) as probe:
            self.assertEqual(api.get_backend('cpu'), 'cpu')
            probe.assert_not_called()
            self.assertEqual(api.get_backend(), 'cuda')
        with self.assertRaises(ValueError):
            api.get_backend('invalid')

    def test_automatic_dispatch_and_gpu_gradient(self):
        args = self.setup_case()
        expected = CPUReconstructionWorkspace(*args, reuse_svd=False)
        strengths = np.array([-.07, .03])
        with patch('optimization.reconstruction_cuda._cupy', return_value=np):
            workspace = api.ReconstructionWorkspace(*args, reuse_svd=False)
            self.assertIsInstance(workspace, CUDAReconstruction)
            for actual, target in zip(workspace.loss_gradient(strengths), expected.loss_gradient(strengths)):
                np.testing.assert_allclose(actual, target, rtol=2e-9)
            # Accepted fits and gradient arrays remain cached on the backend.
            with patch.object(workspace, '_propagate', side_effect=AssertionError('cache miss')):
                workspace.loss_gradient(strengths)
            a, info = workspace.optimize(method='gradient', max_iterations=3)
            ref, other = expected.optimize(method='gradient', max_iterations=3)
            np.testing.assert_allclose(a.strengths, ref.strengths, rtol=2e-9)
            self.assertIsNone(info['hessian'])
            self.assertEqual(info['backend'], 'cuda')
            np.testing.assert_allclose(info['loss'], other['loss'], rtol=2e-9)

    def test_public_default_and_explicit_cpu(self):
        m, g, data, focal, planes, modes, rho, div = self.setup_case()
        a = Aberration(modes, [0, 0])
        args = (m, g, data, focal, planes, a, rho, div)
        with patch('optimization.reconstruction_cuda._cupy', return_value=np):
            sample, info = api.estimate_sample_3D(*args, return_info=True)
            ref, other = api.estimate_sample_3D(*args, return_info=True, backend='cpu')
            self.assertEqual(info['backend'], 'cuda')
            self.assertEqual(other['backend'], 'cpu')
            np.testing.assert_allclose(sample.image_mask, ref.image_mask)


if __name__ == '__main__':
    unittest.main()
