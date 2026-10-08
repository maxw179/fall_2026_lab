import unittest
from unittest.mock import patch
import numpy as np
from optimization.reconstruction import (CPUReconstructionWorkspace,
                                         estimate_sample_3D, evaluate_loss_3D)
from utils.psf import (Microscope, Arbitrary_Grid, Image_Mask, Image_Mask_3D,
                       _convolution_grid, _grid_at_z)
from utils.zernike import Aberration


def fixed_kernel_otfs(kernel):
    """Supply a known convolution kernel without invoking optical propagation."""
    def build(workspace, strengths, first=False, include_base=True):
        if first:
            raise AssertionError("Synthetic kernel fixture has no phase derivatives.")
        otf = workspace._transform(kernel)
        shape = (len(workspace.focal_z), len(workspace.sample_z), len(workspace.weights))
        return np.broadcast_to(otf, shape).copy(), None, None
    return build


def synthetic_workspace():
    """Use the real Newton control flow with an independent analytic loss."""
    workspace = CPUReconstructionWorkspace.__new__(CPUReconstructionWorkspace)
    workspace.modes = np.array([[0, 4]])
    return workspace


class ReconstructionTests(unittest.TestCase):
    def test_padded_solve_matches_dense_periodic_ridge(self):
        m = Microscope(1,.0013,1.333,1.05,7.2,4,3.5,15.12,16)
        g = Arbitrary_Grid(.004,.006,4,5,.02,-.03,0)
        a = Aberration([[0,0]],[0])
        kernel = np.zeros((5,5))
        kernel[2,2], kernel[3,2], kernel[2,1] = 1, .2, .1
        data = np.arange(20.).reshape(1,4,5)
        padding = (2,3)
        shape = (8,11)
        columns = []
        for index in range(np.prod(shape)):
            impulse = np.zeros(shape); impulse.flat[index] = 1
            columns.append((impulse + .2*np.roll(impulse,1,axis=0)
                             + .1*np.roll(impulse,-1,axis=1)).ravel())
        A = np.column_stack(columns)
        padded = np.pad(data[0],((2,2),(3,3))).ravel()
        rho = .01
        fitted = np.linalg.solve(A.T@A+rho*np.eye(A.shape[1]),A.T@padded)
        expected = fitted.reshape(shape)[2:6,3:8]
        expected_loss = np.prod(shape) * (np.sum((A@fitted-padded)**2) + rho*np.sum(fitted**2))
        with patch.object(CPUReconstructionWorkspace, '_build_otfs', fixed_kernel_otfs(kernel)):
            sample, info = estimate_sample_3D(m,g,data,[0],[0],a,rho,padding=padding,return_info=True)
            loss = evaluate_loss_3D(m,g,data,[0],[0],a,rho,padding=padding)
        self.assertAlmostEqual(loss, expected_loss, places=9)
        self.assertEqual(loss, info['loss'])
        np.testing.assert_allclose(sample.image_mask[0],expected,atol=1e-12)
        np.testing.assert_array_equal(sample.get_xy()[0],g.get_xy()[0])
        self.assertEqual(info['padded_shape'],shape)

    def test_padding_alignment_and_validation(self):
        m = Microscope(1,.0013,1.333,1.05,7.2,4,3.5,15.12,16)
        a = Aberration([[0,0]],[0])
        for nx,ny in [(4,5),(5,4)]:
            g = Arbitrary_Grid(.004,.006,nx,ny,.02,-.03,0)
            kernel = np.zeros((5,5));kernel[2,2] = 1
            data = np.zeros((1,nx,ny));data[0,-1,0] = 1
            with patch.object(CPUReconstructionWorkspace, '_build_otfs', fixed_kernel_otfs(kernel)):
                for padding in [None,0,1,(2,3)]:
                    sample = estimate_sample_3D(m,g,data,[0],[0],a,.01,padding=padding)
                    np.testing.assert_allclose(sample.image_mask,data/1.01,atol=1e-14)
            for padding in [-1,1.5,(1,-2),(1,)]:
                with self.assertRaises(ValueError):
                    estimate_sample_3D(m,g,data,[0],[0],a,.01,padding=padding)

    def test_normal_equations(self):
        rng = np.random.default_rng(3)
        m = Microscope(1, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 24)
        a = Aberration([[0, 2]], [.1])
        for shape in [(6, 7), (7, 6)]:
            grid = Arbitrary_Grid(.006, .008, *shape, .01, -.02, 99)
            sample_z = [-.001, .0007, .002]
            focus = [0, .001]
            diversities = [Aberration([[1, 3]], [.07]), Aberration([[0, 2]], [-.2])]
            for mode in ['scalar', 'vector']:
                S = np.empty((*shape, 2, 3), dtype=complex)
                for k, z in enumerate(focus):
                    for i, zi in enumerate(sample_z):
                        nx, ny = [n + (n % 2 == 0) for n in shape]
                        kg = Arbitrary_Grid(.006*(nx-1)/(shape[0]-1), .008*(ny-1)/(shape[1]-1), nx, ny, 0, 0, zi-z)
                        kernel = m.compute_PSF(kg, a+diversities[k], mode)[2]
                        # Independent explicit wrapping of each kernel displacement.
                        wrapped = np.zeros(shape)
                        for u in range(nx):
                            for v in range(ny):
                                wrapped[(u-nx//2)%shape[0], (v-ny//2)%shape[1]] += kernel[u,v]
                        S[:,:,k,i] = np.fft.fft2(wrapped)
                true = rng.random((3,*shape))
                F = np.fft.fft2(true).transpose(1,2,0)
                D = (S @ F[...,None])[...,0]
                data = np.fft.ifft2(D.transpose(2,0,1)).real
                rho = float(np.abs(S).max()**2 * .001)
                sample = estimate_sample_3D(m, grid, data, focus, sample_z, a, rho,
                                            diversities, mode, frequency_batch_size=5, padding=0)
                adj = S.conj().swapaxes(-1,-2)
                expected_F = np.linalg.solve(adj@S+rho*np.eye(3), (adj@D[...,None]))[...,0]
                expected = np.fft.ifft2(expected_F.transpose(2,0,1)).real
                np.testing.assert_allclose(sample.image_mask, expected, rtol=1e-9, atol=1e-10)
                np.testing.assert_array_equal(sample.z_levels, sample_z)
                np.testing.assert_array_equal(sample.get_xy()[0], grid.get_xy()[0])

    def test_fourier_loss_matches_spatial_energy(self):
        from optimization.reconstruction import _fourier_loss
        rng = np.random.default_rng(42)
        for shape in [(4, 5), (5, 4), (3, 1)]:
            residual = rng.normal(size=(2, *shape))
            sample = rng.normal(size=(3, *shape))
            actual = _fourier_loss(
                np.fft.rfft2(residual).reshape(2, -1),
                np.fft.rfft2(sample).reshape(3, -1), .13, shape)
            expected = np.prod(shape) * (np.sum(residual**2) + .13*np.sum(sample**2))
            np.testing.assert_allclose(actual, expected, rtol=1e-14)

    def test_kernel_transform_wraps_and_reuses_buffer(self):
        from optimization.reconstruction import _KernelTransform
        rng = np.random.default_rng(43)
        for shape in [(4, 5), (9, 8)]:
            transform = _KernelTransform(shape)
            for kernel_shape in [(5, 5), (5, 5), (11, 13), (3, 3)]:
                kernel = rng.normal(size=kernel_shape)
                periodic = np.zeros(shape)
                for (x, y), value in np.ndenumerate(kernel):
                    periodic[(x-kernel_shape[0]//2) % shape[0],
                             (y-kernel_shape[1]//2) % shape[1]] += value
                np.testing.assert_allclose(
                    transform(kernel), np.fft.rfft2(periodic).ravel(), atol=1e-13)

    def test_invalid_inputs(self):
        m = Microscope(1,.0013,1.333,1.05,7.2,4,3.5,15.12,16)
        g = Arbitrary_Grid(.01,.01,3,3,0,0,0)
        a = Aberration([[0,0]],[0])
        for rho in [0, -1, np.nan]:
            with self.assertRaises(ValueError):
                estimate_sample_3D(m,g,np.zeros((1,3,3)),[0],[0],a,rho)
        with self.assertRaises(ValueError):
            estimate_sample_3D(m,g,np.zeros((2,3,3)),[0],[0],a,1)


class LossDerivativeTests(unittest.TestCase):
    def test_analytic_derivatives_against_refitted_loss(self):
        from optimization.reconstruction import evaluate_loss_derivatives_3D

        for mode, order, shape, padding, planes, focuses in [
            ('scalar', 1, (4, 5), 0, [0, .001, .002], [0, .001]),
            ('vector', 3, (5, 4), (1, 2), [0, .001], [0, 0, .001]),
        ]:
            with self.subTest(mode=mode):
                m = Microscope(order, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
                g = Arbitrary_Grid(.003, .004, *shape, 0, 0, 0)
                modes = [[0, 4], [-2, 2]]
                a = Aberration(modes, [-.07, .03])
                diversities = [Aberration([[0, 4]], [s]) for s in np.linspace(-.1, .1, len(focuses))]
                volume = Image_Mask_3D([
                    Image_Mask(_grid_at_z(g, z), np.random.default_rng(i).random(shape))
                    for i, z in enumerate(planes)])
                data = m.compute_image_3D(volume, Aberration(modes, [-.12, .08]),
                                          focuses, mode, diversities)[3]
                rho = float(m.compute_PSF(_convolution_grid(g), a, mode)[2].sum()**2 * .03)

                def evaluate(strengths, derivatives=False, batch=7):
                    fn = evaluate_loss_derivatives_3D if derivatives else evaluate_loss_3D
                    return fn(m, g, data, focuses, planes, Aberration(modes, strengths),
                              rho, diversities, mode, batch, padding)

                loss, gradient, hessian = evaluate(a.strengths, True)
                self.assertAlmostEqual(loss / evaluate(a.strengths), 1, places=12)
                step = 1e-5
                numerical_g = np.zeros(2)
                numerical_h = np.zeros((2, 2))
                for j in range(2):
                    delta = np.eye(2)[j] * step
                    plus = evaluate(a.strengths+delta, True)
                    minus = evaluate(a.strengths-delta, True)
                    numerical_g[j] = (plus[0]-minus[0])/(2*step)
                    numerical_h[:, j] = (plus[1]-minus[1])/(2*step)
                np.testing.assert_allclose(gradient/loss, numerical_g/loss, rtol=2e-5, atol=1e-7)
                np.testing.assert_allclose(hessian/loss, numerical_h/loss, rtol=2e-5, atol=1e-6)
                other = evaluate(a.strengths, True, 1000)
                np.testing.assert_allclose(other[1]/loss, gradient/loss, atol=1e-10)
                np.testing.assert_allclose(other[2]/loss, hessian/loss, atol=1e-9)


class AberrationOptimizationTests(unittest.TestCase):
    def test_negative_curvature_and_loss_scaling(self):
        from optimization.reconstruction import optimize_aberration_3D
        # The zero start is a stationary maximum of this double-well loss.
        for scale in [1e-50, 1e50]:
            def derivatives(*, aberration, **kwargs):
                x = aberration.strengths[0]
                return (scale*(1+(x*x-.04)**2),
                        scale*np.array([4*x*(x*x-.04)]),
                        scale*np.array([[12*x*x-.16]]))

            def loss(**kwargs):
                return derivatives(**kwargs)[0]

            workspace = synthetic_workspace()
            with patch('optimization.reconstruction._workspace', return_value=workspace), \
                 patch.object(workspace, 'derivatives', side_effect=lambda s:
                     derivatives(aberration=Aberration([[0, 4]], s))), \
                 patch.object(workspace, 'loss', side_effect=lambda s:
                     loss(aberration=Aberration([[0, 4]], s))):
                result, info = optimize_aberration_3D(
                    None, None, None, [0], [0], [[0, 4]], 1,
                    gradient_tolerance=1e-9, verbose=False)
            self.assertTrue(info['success'], info['message'])
            self.assertAlmostEqual(abs(result.strengths[0]), .2, places=6)
            self.assertTrue(np.all(np.diff([h['loss'] for h in info['history']]) < 0))

    def test_diverse_optical_images(self):
        from optimization.reconstruction import optimize_aberration_3D
        m = Microscope(1, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
        grid = Arbitrary_Grid(.003, .003, 5, 5, 0, 0, 0)
        plane = np.zeros((5, 5)); plane[2, 2] = 1
        volume = Image_Mask_3D([Image_Mask(grid, plane)])
        truth = Aberration([[0, 4]], [-.08])
        diversities = [Aberration([[0, 4]], [s]) for s in [0, -.15, .15]]
        data = m.compute_image_3D(volume, truth, [0]*3, 'scalar', diversities)[3]
        rho = float(m.compute_PSF(grid, truth, 'scalar')[2].sum()**2 * 1e-6)
        estimated, info = optimize_aberration_3D(
            m, grid, data, [0]*3, [0], [[0, 4]], rho, diversities,
            mode='scalar', padding=0, verbose=False)
        self.assertTrue(info['success'], info['message'])
        self.assertAlmostEqual(estimated.strengths[0], -.08, places=4)
        self.assertLess(info['loss'], info['history'][0]['loss'])
        np.testing.assert_allclose(info['loss'], evaluate_loss_3D(
            m, grid, data, [0]*3, [0], estimated, rho, diversities, 'scalar', padding=0))

    def test_iteration_limit_reports_nonconvergence(self):
        from optimization.reconstruction import optimize_aberration_3D
        def derivatives(*, aberration, **kwargs):
            x = aberration.strengths[0]
            return 1+(x-1)**2, np.array([2*(x-1)]), np.array([[2.]])
        workspace = synthetic_workspace()
        with patch('optimization.reconstruction._workspace', return_value=workspace), \
             patch.object(workspace, 'derivatives', side_effect=lambda s:
                 derivatives(aberration=Aberration([[0, 4]], s))), \
             patch.object(workspace, 'loss', side_effect=lambda s:
                 derivatives(aberration=Aberration([[0, 4]], s))[0]):
            result, info = optimize_aberration_3D(None, None, None, [0], [0], [[0, 4]],
                                                1, max_iterations=1, verbose=False)
        self.assertFalse(info['success'])
        self.assertEqual(info['iterations'], 1)
        self.assertAlmostEqual(result.strengths[0], .1)
