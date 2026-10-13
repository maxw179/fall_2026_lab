"""Validate QR/SVD reduction, including null-space response and optical derivatives."""
import os
import unittest
import numpy as np

from optimization._batched_svd import qr_reduced_svd
from optimization._reconstruction_cpu import CPUReconstructionWorkspace
from optimization.reconstruction_cuda import CUDAReconstruction, _cupy
from utils.psf import Microscope, Arbitrary_Grid, _convolution_grid
from utils.zernike import Aberration

try:
    CP = _cupy()
except RuntimeError:
    if os.environ.get('REQUIRE_CUDA') == '1':
        raise
    CP = None


class QRChecks:
    def host(self, value):
        return np.asarray(value) if self.xp is np else self.xp.asnumpy(value)

    def test_factorization_shapes_rank_deficiency_and_scale(self):
        rng = np.random.default_rng(7)
        for rows, columns in [(24, 40), (40, 24), (3, 33), (33, 3)]:
            for complex_data in [False, True]:
                a = rng.normal(size=(3, rows, columns))
                if complex_data:
                    a = a + 1j * rng.normal(size=a.shape)
                a[0] = 0
                a[1, :, -1] = a[1, :, 0]  # Rank deficient in the tall case.
                for scale in [1e-200, 1., 1e26, 1e200]:
                    for full in [False, True]:
                        with self.subTest(shape=(rows, columns), complex=complex_data,
                                          scale=scale, full=full):
                            u, singular, vh = map(self.host, qr_reduced_svd(
                                self.xp.asarray(a * scale), full, self.xp))
                            rank = min(rows, columns)
                            self.assertEqual(u.shape, (3, rows, rows if full else rank))
                            self.assertEqual(vh.shape, (3, columns if full else rank, columns))
                            reconstructed = (u[:, :, :rank] * (singular / scale)[:, None, :]) @ vh[:, :rank]
                            np.testing.assert_allclose(reconstructed, a, rtol=2e-11, atol=2e-12)
                            np.testing.assert_allclose(u.conj().swapaxes(-1, -2) @ u,
                                np.broadcast_to(np.eye(u.shape[-1]), (3, u.shape[-1], u.shape[-1])),
                                rtol=2e-11, atol=2e-12)
                            np.testing.assert_allclose(vh @ vh.conj().swapaxes(-1, -2),
                                np.broadcast_to(np.eye(vh.shape[-2]), (3, vh.shape[-2], vh.shape[-2])),
                                rtol=2e-11, atol=2e-12)
                            np.testing.assert_allclose(singular / scale, np.linalg.svd(a, compute_uv=False),
                                                       rtol=2e-11, atol=2e-12)

    def test_small_rho_full_null_space_response(self):
        rng = np.random.default_rng(8)
        left = np.linalg.qr(rng.normal(size=(24, 24)) + 1j*rng.normal(size=(24, 24)))[0]
        right = np.linalg.qr(rng.normal(size=(40, 40)) + 1j*rng.normal(size=(40, 40)))[0]
        spectrum = np.r_[np.ones(3), np.geomspace(.1, 1e-12, 18), np.zeros(3)]
        a = ((left * spectrum) @ right[:, :24].conj().T)[None]
        c = rng.normal(size=(1, 40, 3)) + 1j*rng.normal(size=(1, 40, 3))
        u, singular, vh = map(self.host, qr_reduced_svd(self.xp.asarray(a), True, self.xp))
        ru, rs, rvh = np.linalg.svd(a, full_matrices=True)
        data = rng.normal(size=(1, 24, 1)) + 1j*rng.normal(size=(1, 24, 1))
        for rho in [.03, 1e-12]:
            def response(s, v):
                denom = np.full((1, 40), rho)
                denom[:, :24] += s**2
                return v.conj().swapaxes(-1, -2) @ ((v @ c) / denom[:, :, None])
            target = response(rs, rvh)
            np.testing.assert_allclose(response(singular, vh), target,
                                       rtol=2e-7, atol=2e-9*np.max(abs(target)))
            def fit(uu, ss, vv):
                projected = uu.conj().swapaxes(-1, -2) @ data
                return vv[:, :24].conj().swapaxes(-1, -2) @ (
                    (ss / (ss**2 + rho))[:, :, None] * projected)
            target = fit(ru, rs, rvh)
            np.testing.assert_allclose(fit(u, singular, vh), target,
                                       rtol=2e-7, atol=2e-9*np.max(abs(target)))

    def test_guard_mixed_batch_and_optimizer(self):
        m = Microscope(1, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
        grid = Arbitrary_Grid(.003, .004, 4, 5, 0, 0, 0)
        args = (m, grid, np.random.default_rng(11).random((2, 4, 5)),
                [0, .001], np.linspace(0, .002, 33), [[0, 4]], .01)
        fast = CUDAReconstruction(*args, svd_method='qr', _xp=self.xp)
        matrices = np.random.default_rng(12).normal(size=(3, 2, 33)).astype(complex)
        matrices[0] *= 1e4
        for full in (False, True):
            u, s, vh = map(self.host, fast._svd(self.xp.asarray(matrices), full))
            np.testing.assert_allclose((u[:, :, :2]*s[:, None]) @ vh[:, :2], matrices,
                                       rtol=2e-10, atol=1e-10)
        self.assertEqual(fast.svd_counts, dict(qr=4, direct=2))
        # Moderate optical scale so QR is exercised throughout Newton, not just the guard.
        a = Aberration([[0, 4]], [-.07])
        rho = float(m.compute_PSF(_convolution_grid(grid), a, 'scalar')[2].sum()**2 * .03)
        fast = CUDAReconstruction(*args[:-1], rho, svd_method='qr', mode='scalar', _xp=self.xp)
        ref = CPUReconstructionWorkspace(*args[:-1], rho, mode='scalar')
        target, info = ref.optimize(max_iterations=3, num_random_starts=2,
                                   show_initialization_progress=False)
        actual, other = fast.optimize(max_iterations=3, num_random_starts=2,
                                     show_initialization_progress=False)
        np.testing.assert_allclose(actual.strengths, target.strengths, rtol=2e-7, atol=2e-8)
        np.testing.assert_allclose(other['loss'], info['loss'], rtol=2e-8)
        self.assertGreater(fast.svd_counts['qr'], 0)
        np.testing.assert_allclose(other['initialization']['candidate_losses'],
                                   info['initialization']['candidate_losses'], rtol=2e-8)

    def test_optical_parity_cached_fit_and_finite_difference(self):
        m = Microscope(3, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
        grid = Arbitrary_Grid(.003, .004, 4, 5, 0, 0, 0)
        a = Aberration([[0, 4], [-2, 2]], [-.07, .03])
        base_rho = float(m.compute_PSF(_convolution_grid(grid), a, 'vector')[2].sum()**2 * .03)
        for k, p, reuse, rho_factor in [(24, 40, True, 1), (2, 33, False, 1),
                                         (33, 2, True, 1), (2, 33, True, 1e-8)]:
            args = (m, grid, np.random.default_rng(9).random((k, 4, 5)),
                    np.linspace(0, .001, k), np.linspace(0, .002, p),
                    a.modes, base_rho * rho_factor)
            ref = CPUReconstructionWorkspace(*args, frequency_batch_size=7)
            fast = CUDAReconstruction(*args, frequency_batch_size=7, reuse_svd=reuse,
                                      svd_method='qr', _xp=self.xp)
            expected = ref.derivatives(a.strengths)
            fast.loss(a.strengths)  # Retained full right basis must survive the cache hit.
            actual = fast.derivatives(a.strengths)
            for value, target in zip(actual, expected):
                np.testing.assert_allclose(np.asarray(value)/expected[0], np.asarray(target)/expected[0],
                                           rtol=2e-8, atol=2e-9)
            sample = fast.estimate_sample(a.strengths).image_mask
            target = ref.estimate_sample(a.strengths).image_mask
            np.testing.assert_allclose(sample, target, rtol=2e-8, atol=2e-9*np.max(abs(target)))
            if rho_factor == 1:
                self.assertGreater(fast.svd_counts['qr'], 0)
            else:
                self.assertGreater(fast.svd_counts['direct'], 0)
            if rho_factor == 1:
                delta = np.array([1e-5, 0])
                numerical = (fast.derivatives(a.strengths+delta)[1]
                             - fast.derivatives(a.strengths-delta)[1])/(2e-5)
                np.testing.assert_allclose(numerical/expected[0], actual[2][:, 0]/expected[0],
                                           rtol=3e-5, atol=2e-6)


class PortableQRTests(QRChecks, unittest.TestCase):
    xp = np


@unittest.skipIf(CP is None, 'CUDA/CuPy unavailable')
class CUDAQRTests(QRChecks, unittest.TestCase):
    xp = CP


if __name__ == '__main__':
    unittest.main()
