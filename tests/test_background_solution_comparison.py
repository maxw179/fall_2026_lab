"""Independent constrained-photometry checks for fixed-wavefront diagnostics."""
import unittest
import numpy as np
from experimental_10_15.compare_background_solutions import constrained_gains


class ConstrainedGainTests(unittest.TestCase):
    def test_recovers_noiseless_gains_when_true_scale_satisfies_constraint(self):
        gains=np.array([.4,.8,1.2,1.6]);gains/=np.sqrt(np.mean(gains*gains))
        power=np.array([4.,12.,2.,8.])
        actual=constrained_gains(power*gains,power)
        np.testing.assert_allclose(actual,gains,rtol=1e-11)

    def test_beats_other_feasible_gain_vectors_and_preserves_global_scale(self):
        rng=np.random.default_rng(71)
        a=rng.uniform(.5,20,8);b=rng.uniform(1,15,8)
        gains=constrained_gains(a,b)
        np.testing.assert_allclose(np.mean(gains*gains),1,rtol=1e-10)
        self.assertTrue(np.all(gains>0))
        feasible=rng.uniform(.1,3,(1000,8));feasible/=np.sqrt(np.mean(feasible**2,axis=1))[:,None]
        objective=lambda g:np.sum(b*g*g-2*a*g,axis=-1)
        self.assertLessEqual(objective(gains),np.min(objective(feasible))+1e-10)
        self.assertLessEqual(objective(gains),objective(np.ones(8))+1e-10)


if __name__=='__main__':unittest.main()
