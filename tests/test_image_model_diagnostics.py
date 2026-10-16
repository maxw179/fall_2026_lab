"""Check scientific bookkeeping in the read-only image-model diagnostics."""
import unittest
import numpy as np
from scipy import fft
from experimental_10_15.image_model_diagnostics import robust_affine,phase_transform,workspace,fit_diagnostics
from utils import zernike


class ImageModelDiagnosticsTests(unittest.TestCase):
    def test_phase_rotation_and_reflection_include_both_azimuthal_components(self):
        modes=np.array([[-2,2],[2,2],[-1,3],[1,3],[0,4]])
        coefficients=np.array([.03,-.07,.11,.04,-.08]);alpha=.9
        theta,phi=np.meshgrid(np.linspace(0,alpha,21),np.linspace(-np.pi,np.pi,35))
        for reflected in [False,True]:
            for angle in [0,.17,np.pi/2,np.pi]:
                transformed=phase_transform(coefficients,modes,angle,reflected)
                actual=zernike.Aberration(modes,transformed).construct_map(alpha)(theta,phi)
                expected=zernike.Aberration(modes,coefficients).construct_map(alpha)(theta,angle+(-phi if reflected else phi))
                np.testing.assert_allclose(actual,expected,atol=4e-15)

    def test_robust_photometric_fit_recovers_gain_offset_despite_outliers(self):
        x=np.linspace(-20,100,1000);y=1.4*x-7
        y[::100]+=1000
        gain,offset=robust_affine(x,y)
        self.assertAlmostEqual(gain,1.4,delta=.005)
        self.assertAlmostEqual(offset,-7,delta=.08)

    def test_spatial_residual_matches_fourier_prediction_and_preserves_input(self):
        rng=np.random.default_rng(71)
        images=rng.normal(size=(4,8,8));original=images.copy()
        strengths=np.array([.02,-.03])
        case=dict(images=images,modes=np.array([[2,2],[0,4]]),
                  params=dict(N_order=3,lambd=.0013,n=1.333,num_apt=1.05,f=7.2,mag=4,w_0=3.5,L_bfp=15.12,grid_bfp=8),
                  diversity_strengths=np.array([[0,0],[.1,0],[-.1,0],[0,.1]]),
                  focal_z=np.array([-.001,0,0,.001]),sample_z=np.array([-.001,0,.001]),
                  solver=dict(rho=1e46),field_of_view_um=6.5)
        ws=workspace(case)
        metrics,residual,sample=fit_diagnostics(ws,strengths,images)
        fit=ws._get_fit(strengths)[0]
        predicted=fft.irfft2(np.einsum('kpf,pf->kf',fit.otfs,fit.estimated).reshape(4,ws.shape[0],ws.shape[1]//2+1),s=ws.shape)[ws.crop]
        np.testing.assert_allclose(images-residual,predicted,atol=3e-15)
        np.testing.assert_array_equal(images,original)
        self.assertGreater(metrics['loss'],0)


if __name__=='__main__':unittest.main()
