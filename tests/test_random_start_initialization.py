"""Loss-only initialization, RMS distribution and reproducibility checks."""
import unittest
from unittest.mock import patch
import numpy as np
from optimization._random_starts import random_aberration_starts
from optimization._reconstruction_cpu import CPUReconstructionWorkspace
from utils.zernike import Aberration, zernike_RMS


class RandomStartInitializationTests(unittest.TestCase):
    modes = np.array([[0, 4], [-2, 2], [2, 2]])
    alpha = .8

    def workspace(self):
        w = CPUReconstructionWorkspace.__new__(CPUReconstructionWorkspace)
        w.modes = self.modes.copy()
        w._alpha = self.alpha
        w.reuse_svd = True
        return w

    def test_distribution_and_prefixes(self):
        small = random_aberration_starts(self.modes, self.alpha, 10, seed=4)
        large = random_aberration_starts(self.modes, self.alpha, 100, seed=4)
        np.testing.assert_array_equal(small, large[:10])
        rms = np.array([zernike_RMS(Aberration(self.modes, x), self.alpha) for x in large])
        self.assertTrue(np.all((rms >= .005) & (rms <= .5)))
        # Independent draws of radius and direction, using the documented draw order.
        rng = np.random.default_rng(4)
        expected = []
        for _ in range(100):
            rng.normal(size=3)
            expected.append(np.exp(rng.uniform(np.log(.005), np.log(.5))))
        np.testing.assert_allclose(rms, expected, rtol=1e-12)

    def test_default_screens_100_by_loss_before_any_derivatives(self):
        w = self.workspace()
        target = random_aberration_starts(self.modes, self.alpha, 100, seed=0)[9]
        screened = []
        def loss(s):
            self.assertFalse(w.reuse_svd)
            screened.append(s.copy())
            return 1+np.sum((s-target)**2)
        def derivatives(s):
            self.assertEqual(len(screened), 101)
            self.assertTrue(w.reuse_svd)
            return 1+np.sum((s-target)**2), 2*(s-target), 2*np.eye(3)
        events=[]
        with patch.object(w, 'loss', side_effect=loss), patch.object(w, 'derivatives', side_effect=derivatives):
            estimate, info = w.optimize(on_progress=events.append)
        np.testing.assert_array_equal(estimate.strengths, target)
        self.assertEqual(info['initialization']['selected_index'], 10)
        self.assertEqual(info['initialization']['num_random_starts'], 100)
        self.assertTrue(info['success'])
        self.assertEqual(events[0]['stage'], 'initialization')
        self.assertEqual([e for e in events if e['stage']=='initialization'][-1]['completed'],101)

    def test_explicit_start_and_zero_are_candidates(self):
        w=self.workspace(); supplied=np.array([.04,.02,-.01])
        with patch.object(w,'loss',side_effect=lambda s:1+np.sum((s-supplied)**2)), \
             patch.object(w,'derivatives',side_effect=lambda s:(1.,2*(s-supplied),2*np.eye(3))):
            estimate,info=w.optimize(supplied,num_random_starts=5)
        np.testing.assert_array_equal(estimate.strengths,supplied)
        points=info['initialization']['candidate_strengths']
        self.assertEqual(points.shape,(7,3))
        np.testing.assert_array_equal(points[1],np.zeros(3))

    def test_zero_disables_screening(self):
        w=self.workspace();supplied=np.array([.01,.02,.03])
        with patch.object(w,'loss',side_effect=AssertionError('Screening must be disabled')), \
             patch.object(w,'derivatives',return_value=(1.,np.zeros(3),np.eye(3))):
            estimate,info=w.optimize(supplied,num_random_starts=0)
        np.testing.assert_array_equal(estimate.strengths,supplied)
        self.assertIsNone(info['initialization'])

    def test_gradient_method_does_not_compute_hessian(self):
        w=self.workspace()
        with patch.object(w,'loss',side_effect=lambda s:1+np.sum(s**2)), \
             patch.object(w,'loss_gradient',side_effect=lambda s:(1+np.sum(s**2),2*s)), \
             patch.object(w,'derivatives',side_effect=AssertionError('Hessian called')):
            estimate,info=w.optimize(method='gradient',num_random_starts=3)
        np.testing.assert_array_equal(estimate.strengths,np.zeros(3))
        self.assertIsNone(info['hessian'])

    def test_failure_restores_svd_and_callback(self):
        w=self.workspace();old=object();w._progress_callback=old
        with patch.object(w,'loss',side_effect=RuntimeError('failed scoring')):
            with self.assertRaises(RuntimeError):w.optimize(num_random_starts=1)
        self.assertTrue(w.reuse_svd)
        self.assertIs(w._progress_callback,old)

    def test_screening_progress_updates_and_closes_on_success_or_failure(self):
        for fail in [False, True]:
            w = self.workspace()
            with patch('tqdm.auto.tqdm') as progress, \
                 patch.object(w, 'loss', side_effect=RuntimeError('failed') if fail else lambda s: 1+np.sum(s**2)), \
                 patch.object(w, 'derivatives', return_value=(1., np.zeros(3), np.eye(3))):
                if fail:
                    with self.assertRaises(RuntimeError):
                        w.optimize(num_random_starts=3)
                else:
                    w.optimize(num_random_starts=3)
                self.assertEqual(progress.call_args.kwargs['total'], 4)
                self.assertEqual(progress.return_value.update.call_count, 0 if fail else 4)
                progress.return_value.close.assert_called_once()

    def test_invalid_options(self):
        for count in [-1,1.5,True]:
            with self.assertRaises(ValueError):self.workspace().optimize(num_random_starts=count)
        for bounds in [(0,.5),(.5,.1),(.1,np.inf)]:
            with self.assertRaises(ValueError):self.workspace().optimize(random_start_rms_range=bounds)


if __name__=='__main__':unittest.main()
