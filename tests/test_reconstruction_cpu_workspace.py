"""Check the CPU workspace and public APIs against the preserved reference."""
import unittest
from itertools import product
from unittest.mock import patch

import numpy as np

from optimization import reconstruction as production
from optimization.performance import reconstruction_reference as reference
from optimization.reconstruction import CPUReconstructionWorkspace
from utils.psf import (Arbitrary_Grid, Image_Mask, Image_Mask_3D, Microscope,
                       _convolution_grid, _grid_at_z)
from utils.zernike import Aberration


class CPUWorkspaceTests(unittest.TestCase):
    def setup_case(self, optical_mode, order, shape, padding, sample_z, focus):
        microscope = Microscope(order, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
        grid = Arbitrary_Grid(.003, .004, *shape, .01, -.02, 0)
        modes = [[0, 4], [-2, 2]]
        truth = Aberration(modes, [-.12, .08])
        # Include a known mode outside the correction basis.
        diversities = [Aberration([[1, 3]], [v])
                       for v in np.linspace(-.1, .1, len(focus))]
        rng = np.random.default_rng(9)
        volume = Image_Mask_3D([Image_Mask(_grid_at_z(grid, z), rng.random(shape))
                               for z in sample_z])
        images = microscope.compute_image_3D(volume, truth, focus, optical_mode, diversities)[3]
        power = microscope.compute_PSF(_convolution_grid(grid), truth, optical_mode)[2].sum()**2
        return modes, dict(microscope=microscope, grid=grid, images=images,
                           focal_z_levels=focus, sample_z_levels=sample_z,
                           rho=float(power*.03), diversities=diversities,
                           mode=optical_mode, padding=padding, frequency_batch_size=7)

    def assert_derivatives_match(self, expected, actual):
        scale = max(abs(expected[0]), np.finfo(float).tiny)
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(np.asarray(b)/scale, np.asarray(a)/scale,
                                       rtol=2e-8, atol=2e-9)

    def test_all_switches_match_reference_and_accepted_fits(self):
        cases = [
            # K < P exercises the right null space; zero padding aliases edges.
            ("scalar", 1, (4, 5), 0, [0, .001, .002], [0, .001]),
            # K > P, asymmetric padding, and vector multiphoton derivatives.
            ("vector", 3, (5, 4), (1, 2), [0, .001], [0, 0, .001]),
        ]
        for case in cases:
            modes, common = self.setup_case(*case)
            probes = [np.zeros(2), np.array([-.07, .03])]
            expected = [reference.evaluate_loss_derivatives_3D(
                aberration=Aberration(modes, s), **common) for s in probes]
            for strengths, target in zip(probes, expected):
                self.assert_derivatives_match(target, production.evaluate_loss_derivatives_3D(
                    aberration=Aberration(modes, strengths), **common))
                np.testing.assert_allclose(production.evaluate_loss_3D(
                    aberration=Aberration(modes, strengths), **common)/target[0], 1,
                    rtol=2e-8, atol=2e-9)
            for reuse_svd, spatial, dedup in product([False, True], repeat=3):
                with self.subTest(case=case[0], reuse_svd=reuse_svd,
                                  spatial=spatial, dedup=dedup):
                    workspace = CPUReconstructionWorkspace(modes=modes,
                        psf_batch_size=2, derivative_batch_size=2,
                        reuse_svd=reuse_svd, spatial_hessian=spatial,
                        deduplicate_psfs=dedup, **common)
                    for strengths, target in zip(probes, expected):
                        workspace.clear_cache()
                        self.assert_derivatives_match(target, workspace.derivatives(strengths))
                        workspace.clear_cache()
                        np.testing.assert_allclose(workspace.loss(strengths)/target[0], 1,
                                                   rtol=2e-8, atol=2e-9)
                        self.assert_derivatives_match(target, workspace.derivatives(strengths))

    def test_gradient_only_matches_reference_without_hessian_work(self):
        for case in [
            ('scalar', 1, (4, 5), 0, [0, .001, .002], [0, .001]),
            ('vector', 3, (5, 4), (1, 2), [0, .001], [0, 0, .001]),
        ]:
            modes, common = self.setup_case(*case)
            strengths = np.array([-.07, .03])
            target = reference.evaluate_loss_derivatives_3D(
                aberration=Aberration(modes, strengths), **common)
            workspace = CPUReconstructionWorkspace(modes=modes, reuse_svd=False, **common)
            with patch.object(workspace, 'derivatives', side_effect=AssertionError('Hessian requested')), \
                 patch.object(workspace, '_second_psf_hessian',
                              side_effect=AssertionError('Second derivative requested')):
                for after_loss in [False, True]:
                    workspace.clear_cache()
                    if after_loss:
                        workspace.loss(strengths)
                    actual = workspace.loss_gradient(strengths)
                    self.assert_derivatives_match(target[:2], actual)
                    self.assertEqual(workspace._last_fit.factors, [])
                    with patch.object(workspace, '_propagate',
                                      side_effect=AssertionError('Cached gradient recomputed')):
                        self.assert_derivatives_match(target[:2], workspace.loss_gradient(strengths))

    def test_gradient_optimizer_quadratic_and_public_method_switch(self):
        workspace = CPUReconstructionWorkspace.__new__(CPUReconstructionWorkspace)
        workspace.modes = np.array([[0, 4], [-2, 2]])
        target = np.array([-.08, .03])
        curvature = np.array([2.0, 3.0])

        def loss_gradient(strengths):
            delta = strengths - target
            return 1 + .5*np.sum(curvature*delta**2), curvature*delta

        with patch('optimization.reconstruction._workspace', return_value=workspace) as factory, \
             patch.object(workspace, 'loss_gradient', side_effect=loss_gradient), \
             patch.object(workspace, 'loss', side_effect=lambda s: loss_gradient(s)[0]), \
             patch.object(workspace, 'derivatives', side_effect=AssertionError('Hessian requested')), \
             patch('numpy.linalg.eigh', side_effect=AssertionError('Curvature checked')):
            result, info = production.optimize_aberration_3D(
                None, None, None, [0], [0], workspace.modes, 1,
                method='gradient', max_iterations=100, verbose=False)
        self.assertTrue(info['success'], info['message'])
        self.assertIsNone(info['hessian'])
        self.assertEqual(info['method'], 'gradient')
        self.assertFalse(factory.call_args.kwargs['reuse_svd'])
        np.testing.assert_allclose(result.strengths, target, atol=2e-6)
        history = info['history']
        self.assertTrue(np.all(np.diff([h['loss'] for h in history]) < 0))
        for previous, current in zip(history, history[1:]):
            self.assertLessEqual(np.max(abs(current['strengths'] - previous['strengths'])), .1 + 1e-12)

    def test_invalid_optimization_method(self):
        with self.assertRaisesRegex(ValueError, 'method'):
            production.optimize_aberration_3D(
                None, None, None, [0], [0], [[0, 4]], 1, method='unknown')

    def test_public_estimator_diagnostics_and_private_tuple(self):
        for optical_mode in ["scalar", "vector"]:
            with self.subTest(mode=optical_mode):
                modes, common = self.setup_case(optical_mode, 3, (4, 5), (1, 2),
                                                [0, .001, .002], [0, .001])
                aberration = Aberration(modes, [-.07, .03])
                expected_sample, expected_info = reference.estimate_sample_3D(
                    aberration=aberration, return_info=True, **common)
                actual_sample, actual_info = production.estimate_sample_3D(
                    aberration=aberration, return_info=True, **common)
                np.testing.assert_allclose(actual_sample.image_mask, expected_sample.image_mask,
                                           rtol=2e-8, atol=2e-9)
                np.testing.assert_array_equal(actual_sample.z_levels, expected_sample.z_levels)
                for actual, expected in zip(actual_sample.get_xy(), expected_sample.get_xy()):
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual_info["padding"], expected_info["padding"])
                self.assertEqual(actual_info["padded_shape"], expected_info["padded_shape"])
                for key in ["loss", "relative_residual", "measured_relative_residual"]:
                    np.testing.assert_allclose(actual_info[key], expected_info[key],
                                               rtol=2e-8, atol=2e-9)
                args = (common["microscope"], common["grid"], common["images"],
                        common["focal_z_levels"], common["sample_z_levels"],
                        aberration, common["rho"], common["diversities"],
                        common["mode"], common["frequency_batch_size"], common["padding"])
                expected = reference._solve_reconstruction(*args)
                actual = production._solve_reconstruction(*args)
                for index in [0, 1, 2, 5, 6]:
                    scale = max(float(np.max(abs(expected[index]))), np.finfo(float).tiny)
                    np.testing.assert_allclose(actual[index]/scale, expected[index]/scale,
                                               rtol=2e-8, atol=2e-9)
                for index in [3, 4, 7]:
                    self.assertEqual(actual[index], expected[index])

    def test_repeated_psfs_and_field_recomputation(self):
        modes, common = self.setup_case("vector", 2, (4, 5), 0,
                                        [0, .001, .002], [0, 0, .001, .001])
        # Identical diversity arrays at repeated focal positions.
        common["diversities"] = [Aberration([[0, 4]], [.05])]*4
        strengths = np.array([-.07, .03])
        target = reference.evaluate_loss_derivatives_3D(
            aberration=Aberration(modes, strengths), **common)
        for retain_fields in [False, True]:
            for batch in [1, 4]:
                with self.subTest(retain_fields=retain_fields, batch=batch):
                    workspace = CPUReconstructionWorkspace(modes=modes,
                        psf_batch_size=batch, derivative_batch_size=1,
                        retain_fields=retain_fields, **common)
                    self.assertLess(len(workspace.groups), 4*3)
                    self.assert_derivatives_match(target, workspace.derivatives(strengths))

    def test_gradient_and_hessian_against_finite_differences(self):
        modes, common = self.setup_case("vector", 3, (4, 5), (1, 2),
                                        [0, .001, .002], [0, .001])
        workspace = CPUReconstructionWorkspace(modes=modes, **common)
        strengths, step = np.array([-.07, .03]), 1e-5
        loss, gradient, hessian = workspace.derivatives(strengths)
        numerical_gradient, numerical_hessian = np.zeros(2), np.zeros((2, 2))
        for j in range(2):
            delta = np.eye(2)[j]*step
            plus = workspace.derivatives(strengths + delta)
            minus = workspace.derivatives(strengths - delta)
            numerical_gradient[j] = (plus[0] - minus[0])/(2*step)
            numerical_hessian[:, j] = (plus[1] - minus[1])/(2*step)
        np.testing.assert_allclose(gradient/loss, numerical_gradient/loss,
                                   rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(hessian/loss, numerical_hessian/loss,
                                   rtol=2e-5, atol=1e-6)

    def test_last_fit_cache_and_snapshot(self):
        modes, common = self.setup_case("scalar", 1, (4, 5), 0,
                                        [0, .001], [0, .001])
        workspace = CPUReconstructionWorkspace(modes=modes, **common)
        strengths = np.array([-.07, .03])
        with patch.object(workspace, "_propagate", wraps=workspace._propagate) as propagate:
            target = workspace.derivatives(strengths)
            count = propagate.call_count
            self.assertEqual(workspace.loss(strengths), target[0])
            self.assert_derivatives_match(target, workspace.derivatives(strengths))
            self.assertEqual(propagate.call_count, count)
        # External mutable setup/data changes must not invalidate a snapshot.
        common["images"].fill(0)
        common["microscope"].N_order = 3
        common["diversities"][0].strengths[:] = 2
        workspace.clear_cache()
        self.assert_derivatives_match(target, workspace.derivatives(strengths))

    def test_small_rho_null_space_matches_reference(self):
        modes, common = self.setup_case("scalar", 1, (4, 5), 0,
                                        [0, .001, .002], [0, .001])
        common["rho"] *= 1e-5
        strengths = np.array([-.07, .03])
        target = reference.evaluate_loss_derivatives_3D(
            aberration=Aberration(modes, strengths), **common)
        workspace = CPUReconstructionWorkspace(modes=modes, **common)
        self.assert_derivatives_match(target, workspace.derivatives(strengths))

    def test_newton_recovery_matches_reference(self):
        microscope = Microscope(1, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 16)
        grid = Arbitrary_Grid(.003, .003, 5, 5, 0, 0, 0)
        plane = np.zeros((5, 5))
        plane[2, 2] = 1
        sample = Image_Mask_3D([Image_Mask(grid, plane)])
        truth = Aberration([[0, 4]], [-.08])
        diversities = [Aberration([[0, 4]], [s]) for s in [0, -.15, .15]]
        images = microscope.compute_image_3D(sample, truth, [0]*3, "scalar", diversities)[3]
        rho = float(microscope.compute_PSF(grid, truth, "scalar")[2].sum()**2 * 1e-6)
        common = dict(microscope=microscope, grid=grid, images=images,
                      focal_z_levels=[0]*3, sample_z_levels=[0], rho=rho,
                      diversities=diversities, mode="scalar", padding=0)
        expected, expected_info = reference.optimize_aberration_3D(
            modes=truth.modes, verbose=False, **common)
        workspace = CPUReconstructionWorkspace(modes=truth.modes, **common)
        actual, actual_info = workspace.optimize()
        with patch("optimization.reconstruction._workspace", wraps=production._workspace) as factory:
            public, public_info = production.optimize_aberration_3D(
                modes=truth.modes, verbose=False, **common)
        self.assertEqual(factory.call_count, 1)
        self.assertTrue(public_info["success"], public_info["message"])
        np.testing.assert_allclose(public.strengths, expected.strengths, atol=1e-6)
        self.assertTrue(expected_info["success"], expected_info["message"])
        self.assertTrue(actual_info["success"], actual_info["message"])
        np.testing.assert_allclose(actual.strengths, expected.strengths, atol=1e-6)
        self.assertAlmostEqual(actual.strengths[0], -.08, places=4)
        np.testing.assert_allclose(actual_info["loss"]/expected_info["loss"], 1, rtol=1e-7)
        self.assertTrue(np.all(np.diff([h["loss"] for h in actual_info["history"]]) < 0))

    def test_newton_escapes_stationary_negative_curvature(self):
        modes, common = self.setup_case("scalar", 1, (4, 5), 0, [0], [0])
        workspace = CPUReconstructionWorkspace(modes=modes[:1], **common)

        def derivatives(strengths):
            x = strengths[0]
            return (1 + (x*x - .04)**2, np.array([4*x*(x*x - .04)]),
                    np.array([[12*x*x - .16]]))

        with patch.object(workspace, "derivatives", side_effect=derivatives), \
             patch.object(workspace, "loss", side_effect=lambda s: derivatives(s)[0]):
            result, info = workspace.optimize(gradient_tolerance=1e-9)
        self.assertTrue(info["success"], info["message"])
        self.assertAlmostEqual(abs(result.strengths[0]), .2, places=6)


if __name__ == "__main__":
    unittest.main()
