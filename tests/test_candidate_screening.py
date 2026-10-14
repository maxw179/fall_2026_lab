"""Candidate scores, numerical fallback, seed selection and screening state."""
import os
import unittest
from unittest.mock import patch
import numpy as np

from optimization._screening import linear_screening_loss, spectral_screening_loss
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


class ScreeningChecks:
    def host(self, value):
        return np.asarray(value) if self.xp is np else self.xp.asnumpy(value)

    def case(self, k=2, p=3, mode='vector', order=3, padding=0, rho_factor=1):
        m = Microscope(order, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
        g = Arbitrary_Grid(.003, .004, 4, 5, 0, 0, 0)
        a = Aberration([[0, 4], [-2, 2]], [-.07, .03])
        rho = float(m.compute_PSF(_convolution_grid(g), a, mode)[2].sum()**2 * .03 * rho_factor)
        args = (m, g, np.random.default_rng(3).random((k, 4, 5)),
                np.linspace(0, .001, k), np.linspace(0, .002, p), a.modes, rho)
        options = dict(mode=mode, padding=padding, frequency_batch_size=7, psf_batch_size=3)
        return args, options

    def test_linear_loss_matches_independent_svd_fit(self):
        rng = np.random.default_rng(1)
        for k, p in [(24, 40), (40, 24), (2, 33), (33, 2), (3, 3)]:
            for scale in [1., 1e26]:
                s = (rng.normal(size=(3, k, p)) + 1j*rng.normal(size=(3, k, p))) * scale
                s[0] = 0
                s[1, -1] = s[1, 0]  # Includes a rank-deficient measurement operator.
                d = rng.normal(size=(3, k)) + 1j*rng.normal(size=(3, k))
                rho = .03*scale**2
                u, singular, vh = np.linalg.svd(s, full_matrices=False)
                projected = (u.conj().swapaxes(-1, -2) @ d[..., None])[..., 0]
                fitted = (vh.conj().swapaxes(-1, -2) @ (
                    singular/(singular**2+rho)*projected)[..., None])[..., 0]
                residual = d - (s @ fitted[..., None])[..., 0]
                target = np.sum(abs(residual)**2, axis=1) + rho*np.sum(abs(fitted)**2, axis=1)
                actual = self.host(linear_screening_loss(
                    self.xp.asarray(s), self.xp.asarray(d), rho, self.xp))
                np.testing.assert_allclose(actual, target, rtol=2e-9, atol=2e-12)

    def test_spectral_loss_closed_form_with_left_null_space(self):
        for k, p in [(2, 33), (33, 2)]:
            s = np.zeros((1, k, p), dtype=complex)
            s[0, 0, 0], s[0, 1, 1] = 1., 1.7
            d = np.ones((1, k), dtype=complex) * (1.+.3j)
            for rho in [1e-12, 1e-18]:
                target = abs(d[0, 0])**2 * (rho/(1+rho) + rho/(1.7**2+rho) + (k-2))
                def svd(matrices, full_matrices):
                    return self.xp.linalg.svd(matrices, full_matrices=full_matrices)
                actual = self.host(spectral_screening_loss(
                    self.xp.asarray(s), self.xp.asarray(d), rho, svd, self.xp))[0]
                np.testing.assert_allclose(actual, target, rtol=2e-9, atol=target*2e-12)

    def test_linear_guard_boundary_with_ill_conditioned_operator(self):
        rng = np.random.default_rng(13)
        u = np.linalg.qr(rng.normal(size=(24, 24))+1j*rng.normal(size=(24, 24)))[0]
        v = np.linalg.qr(rng.normal(size=(40, 40))+1j*rng.normal(size=(40, 40)))[0]
        singular = np.geomspace(1., 1e-12, 24)
        rho = np.sum(singular**2)/1e6
        s = ((u*singular) @ v[:, :24].conj().T)[None]
        d = (u @ (rng.normal(size=24)+1j*rng.normal(size=24)))[None]
        expected = np.sum(rho/(singular**2+rho)*abs(u.conj().T @ d[0])**2)
        actual = self.host(linear_screening_loss(
            self.xp.asarray(s), self.xp.asarray(d), rho, self.xp))[0]
        np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-9*expected)

    def test_near_tied_candidates_rechecked_before_selection(self):
        args, options = self.case()
        w = CUDAReconstruction(*args, **options, screening_batch_size=3, _xp=self.xp)
        candidates = np.array([[0., 0.], [.01, 0.], [.02, 0.]])
        approximate = np.array([1.-1e-9, 1.+1e-9, 1.])
        canonical = np.array([1., 1.-2e-9, 1.+2e-9])
        def fast(matrices, data, counts):
            bins = len(matrices)//3
            return self.xp.repeat(self.xp.asarray(approximate), bins)/float(self.host(w.weights).sum())
        with patch.object(w, '_screen_frequency_losses', side_effect=fast), \
             patch.object(w, 'loss', side_effect=lambda point: canonical[int(round(point[0]*100))]) as checked:
            scores = w.screen_losses(candidates)
        np.testing.assert_array_equal(scores, canonical)
        self.assertEqual(int(np.argmin(scores)), 1)
        self.assertEqual(checked.call_count, 3)

    def test_optical_scores_batches_fallback_and_progress(self):
        candidates = np.array([[0., 0.], [-.07, .03], [0., 0.], [.12, -.2], [.005, .009]])
        for k, p, mode, order, padding, rho_factor in [
            (2, 3, 'scalar', 1, 0, 1), (24, 40, 'vector', 3, (1, 2), 1),
            (33, 2, 'vector', 1, None, 1), (2, 33, 'vector', 3, 0, 1e-8)]:
            args, options = self.case(k, p, mode, order, padding, rho_factor)
            reference = CPUReconstructionWorkspace(*args, **options)
            target = reference.screen_losses(candidates)
            for method, batch in [('linear', 1), ('linear', 3), ('svd', 3)]:
                with self.subTest(k=k, p=p, rho_factor=rho_factor, method=method, batch=batch):
                    w = CUDAReconstruction(*args, **options, svd_method='qr',
                        screening_method=method, screening_batch_size=batch, _xp=self.xp)
                    completed = []
                    with patch.object(w, 'derivatives', side_effect=AssertionError('Screening requested derivatives')):
                        scores = w.screen_losses(candidates, on_completed=completed.append)
                    np.testing.assert_allclose(scores, target, rtol=2e-8, atol=2e-9*np.max(target))
                    self.assertEqual(int(np.argmin(scores)), int(np.argmin(target)))
                    self.assertEqual(completed[-1], len(candidates))
                    self.assertEqual(completed, sorted(set(completed)))
                    self.assertTrue(w.reuse_svd)
                    self.assertIsNone(w._last_fit)
                    summary = w._last_screening_info
                    self.assertEqual(summary['linear_frequencies'] + summary['svd_fallback_frequencies'],
                                     len(candidates)*len(w.weights))
                    if method == 'linear' and rho_factor == 1:
                        self.assertGreater(summary['linear_frequencies'], 0)
                    if rho_factor < 1:
                        self.assertGreater(summary['svd_fallback_frequencies'], 0)

    def test_seeded_optimizer_pool_and_result_match_serial(self):
        args, options = self.case(p=33, padding=(1, 2))
        reference = CPUReconstructionWorkspace(*args, **options)
        fast = CUDAReconstruction(*args, **options, svd_method='qr',
                                  screening_batch_size=3, _xp=self.xp)
        kwargs = dict(initial_strengths=[-.02, .03], max_iterations=2,
                      num_random_starts=7, random_start_seed=7, show_initialization_progress=False)
        target, info = reference.optimize(**kwargs)
        actual, other = fast.optimize(**kwargs)
        left, right = info['initialization'], other['initialization']
        np.testing.assert_array_equal(left['candidate_strengths'], right['candidate_strengths'])
        np.testing.assert_allclose(left['candidate_losses'], right['candidate_losses'], rtol=2e-8)
        self.assertEqual(left['selected_index'], right['selected_index'])
        np.testing.assert_allclose(actual.strengths, target.strengths, rtol=2e-7, atol=2e-8)
        np.testing.assert_allclose(other['loss'], info['loss'], rtol=2e-8)

    def test_failure_restores_workspace_and_zero_data_ties(self):
        args, options = self.case()
        w = CUDAReconstruction(*args, **options, svd_method='qr', _xp=self.xp)
        old = object()
        w._progress_callback = old
        with patch.object(w, '_screen_frequency_losses', side_effect=RuntimeError('screening failed')):
            with self.assertRaisesRegex(RuntimeError, 'screening failed'):
                w.optimize(num_random_starts=3, show_initialization_progress=False)
        self.assertIs(w._progress_callback, old)
        self.assertTrue(w.reuse_svd)
        self.assertIsNone(w._last_fit)
        w._progress_callback = None
        w.observed.fill(0)
        _, info = w.optimize(num_random_starts=3, show_initialization_progress=False)
        self.assertEqual(info['initialization']['selected_index'], 0)
        self.assertEqual(info['initialization']['refined_candidates'], 4)
        np.testing.assert_array_equal(info['initialization']['candidate_losses'], np.zeros(4))


class PortableScreeningTests(ScreeningChecks, unittest.TestCase):
    xp = np

    def test_invalid_options(self):
        args, options = self.case()
        for extra in [dict(screening_method='bad'), dict(screening_batch_size=0),
                      dict(screening_batch_size=1.5), dict(screening_batch_size=True)]:
            with self.assertRaises(ValueError):
                CUDAReconstruction(*args, **options, _xp=np, **extra)


@unittest.skipIf(CP is None, 'CUDA/CuPy unavailable')
class CUDAScreeningTests(ScreeningChecks, unittest.TestCase):
    xp = CP


if __name__ == '__main__':
    unittest.main()
