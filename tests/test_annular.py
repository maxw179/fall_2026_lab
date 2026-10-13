import unittest
import numpy as np
from utils import psf, zernike
from utils.annular import AnnularMicroscope
from optimization._reconstruction_common import _derivative_geometry
from optimization.reconstruction import ReconstructionWorkspace


class AnnularTests(unittest.TestCase):
    def setUp(self):
        self.base = psf.Microscope(3, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 48)
        self.grid = psf.Centered_Square_Grid(.0065, 9, .001)
        self.phase = zernike.Aberration([[2, 2]], [.07])

    def test_full_pupil_matches_existing_forward(self):
        model = AnnularMicroscope(self.base, .5, 1.)
        for mode in ['scalar', 'vector']:
            expected = self.base.compute_PSF(self.grid, self.phase, mode)[2]
            actual = model.compute_PSF(self.grid, self.phase, mode)[2]
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=expected.max()*1e-12)

    def test_shared_geometry_and_power_scaling(self):
        model = AnnularMicroscope(self.base)
        a, mask, theta, phi, sz, _, propagate = _derivative_geometry(model, self.grid, [[2, 2]], 'vector')
        phase = np.zeros_like(theta, dtype=complex)
        phase[mask] = np.exp(1j*(self.phase.construct_map(model.alpha)(theta[mask], phi[mask])
                               + model.k*self.grid.z_level*sz[mask]))
        expected = np.sum(np.abs(propagate(a*phase))**2, axis=0)**model.N_order
        actual = model.compute_PSF(self.grid, self.phase)[2]
        np.testing.assert_allclose(actual, expected, rtol=1e-12)
        matched = AnnularMicroscope(self.base, match_power=True).compute_PSF(self.grid, self.phase)[2]
        np.testing.assert_allclose(matched, actual/model.transmitted_power_fraction**3, rtol=1e-12)

    def test_annular_reduced_loss_derivatives(self):
        model = AnnularMicroscope(self.base)
        zs = np.array([-.001, 0., .001])
        sample = psf.bead_img_3D(self.grid, zs, [0.], [0.], [0.], [.001])
        diversity = [zernike.Aberration([[2, 2]], [s]) for s in [-.2, 0., .2]]
        images = model.compute_image_3D(sample, self.phase, [0.]*3, diversities=diversity)[3]
        w = ReconstructionWorkspace(model, self.grid, images, [0.]*3, zs, [[2, 2]], 1e32,
                                    diversities=diversity, backend='cpu', padding=0)
        a = np.array([.04]); h = 1e-5
        loss, gradient, hessian = w.derivatives(a)
        self.assertGreater(abs(gradient[0]), 0)
        fd = (w.loss(a+h)-w.loss(a-h))/(2*h)
        fd2 = (w.derivatives(a+h)[1]-w.derivatives(a-h)[1])/(2*h)
        np.testing.assert_allclose(gradient, [fd], rtol=5e-4)
        np.testing.assert_allclose(hessian[:, 0], fd2, rtol=5e-4)


if __name__ == '__main__':
    unittest.main()
