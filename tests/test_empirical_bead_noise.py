"""Post-binning noise preserves means and has the intended mean/variance law."""
import unittest
import numpy as np
from experimental_10_15.empirical_noise import add_noise,_variance_fit


class EmpiricalNoiseTests(unittest.TestCase):
    def test_poisson_gaussian_mean_variance_signed_floor_and_seed(self):
        clean=np.broadcast_to(np.array([-7.,13.,193.])[:,None,None],(3,512,512)).copy()
        sigma=np.array([3.,4.,5.]);gain=2.3
        actual=add_noise(clean,-7.,gain,sigma,seed=9)
        np.testing.assert_array_equal(actual,add_noise(clean,-7.,gain,sigma,seed=9))
        np.testing.assert_allclose(actual.mean(axis=(1,2)),clean[:,0,0],atol=.15)
        np.testing.assert_allclose(actual.var(axis=(1,2)),gain*np.array([0.,20.,200.])+sigma**2,rtol=.015)
        self.assertTrue(np.any(actual[0]<0))

    def test_variance_fit_recovers_effective_gain_without_extra_averaging(self):
        rng=np.random.default_rng(2);signal=np.repeat([0.,5.,25.,55.,110.,220.,440.,900.,1700.],6000)
        residual=rng.normal(size=len(signal))*np.sqrt(1.8*signal+16.)
        gain,floor,_=_variance_fit([dict(signal=signal,residual=residual)])
        self.assertAlmostEqual(gain,1.8,delta=.15)
        self.assertAlmostEqual(floor,16.,delta=1.)


if __name__=='__main__':unittest.main()
