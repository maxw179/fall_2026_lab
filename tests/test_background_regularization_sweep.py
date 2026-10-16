"""Exact SVD regularization path versus canonical solves and ridge monotonicity."""
import contextlib
import io
import unittest
import numpy as np
from experimental_10_15.image_model_diagnostics import workspace
from experimental_10_15.sweep_background_regularization import regularization_path


class RegularizationPathTests(unittest.TestCase):
    def test_path_matches_independent_refits_and_obeys_ridge_monotonicity(self):
        rng=np.random.default_rng(73)
        case=dict(images=rng.normal(size=(4,8,8)),modes=np.array([[2,2],[0,4]]),
            params=dict(N_order=3,lambd=.0013,n=1.333,num_apt=1.05,f=7.2,mag=4,w_0=3.5,L_bfp=15.12,grid_bfp=8),
            diversity_strengths=np.array([[0,0],[.1,0],[-.1,0],[0,.1]]),
            focal_z=np.array([-.001,0,0,.001]),sample_z=np.array([-.001,0,.001]),
            solver=dict(rho=1e46),field_of_view_um=6.5)
        original=case['images'].copy();ws=workspace(case)
        with contextlib.redirect_stdout(io.StringIO()):
            rows,checks=regularization_path(ws,np.array([.02,-.03]),[1e42,1e46,1e50])
        self.assertEqual(len(checks),3)
        self.assertLess(max(r['relative_error'] for r in checks),1e-11)
        self.assertTrue(np.all(np.diff([r['loss'] for r in rows])>=0))
        self.assertTrue(np.all(np.diff([r['data_loss'] for r in rows])>=0))
        self.assertTrue(np.all(np.diff([r['object_norm_squared'] for r in rows])<=0))
        for r in rows:self.assertAlmostEqual(r['loss'],r['data_loss']+r['regularization_loss'])
        np.testing.assert_array_equal(case['images'],original)
        self.assertEqual(ws.rho,1e46)


if __name__=='__main__':unittest.main()
