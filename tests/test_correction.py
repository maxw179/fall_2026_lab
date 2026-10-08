"""Feedback-loop checks using analytic measurements, without optical propagation."""

import unittest
from unittest.mock import patch

import numpy as np

from systematic_testing import correction
from utils import zernike


class AnalyticSimulator:
    """A known concave brightness surface for checking modal correction signs."""

    def __init__(self, truth):
        self.truth = np.asarray(truth, dtype=float)
        self.modes = [[-2, 2], [2, 2]][:len(truth)]
        self.slm_strengths = np.zeros(len(truth))
        self.images_taken = 0
        self.photons_detected = 0
        self.microscope = object()
        self.optical_mode = "vector"
        self.commands_at_acquisition = []

    def set_slm(self, strengths):
        self.slm_strengths = np.asarray(strengths).copy()

    def residual_rms(self):
        return float(np.linalg.norm(self.truth + self.slm_strengths))

    def acquire(self, focal_z_levels, diversities=None):
        self.commands_at_acquisition.append(self.slm_strengths.copy())
        frames = []
        for i, _ in enumerate(focal_z_levels):
            coefficients = self.truth + self.slm_strengths
            if diversities is not None:
                coefficients = coefficients.copy()
                for mode, strength in zip(diversities[i].modes, diversities[i].strengths):
                    coefficients[self.modes.index(list(mode))] += strength
            frames.append([[10 - np.sum(coefficients**2)]])
        self.images_taken += len(frames)
        return np.asarray(frames)


class CorrectionTests(unittest.TestCase):
    def test_booth_shared_baseline_budget_and_correction_sign(self):
        simulator = AnalyticSimulator([0.04, -0.03])
        history = correction.booth_feedback(
            simulator, focal_positions=[0, 1], bias=0.1, image_budget=19,
        )
        # Two modes and two planes cost 2 * (2*2+1) = 10 frames per sweep.
        self.assertEqual([p["images_taken"] for p in history], [0, 10])
        self.assertEqual(len(simulator.commands_at_acquisition), 5)
        for command in simulator.commands_at_acquisition:
            np.testing.assert_array_equal(command, [0, 0])
        np.testing.assert_allclose(simulator.slm_strengths, [-0.04, 0.03], atol=1e-12)
        self.assertLess(history[-1]["residual_rms"], 1e-12)
        self.assertTrue(history[-1]["used_parabola"].all())

    def test_booth_update_limit_and_gain(self):
        simulator = AnalyticSimulator([0.4])
        correction.booth_feedback(
            simulator, focal_positions=[0], bias=0.1, image_budget=3,
            gain=0.5, max_update=0.1,
        )
        np.testing.assert_allclose(simulator.slm_strengths, [-0.05])

    def test_stack_reacquires_residual_and_does_not_count_solver_iterations(self):
        simulator = AnalyticSimulator([0.04, -0.03])
        results = [
            (zernike.Aberration(simulator.modes, [0.04, -0.03]), {"success": False}),
            (zernike.EmptyAberration(simulator.modes), {"success": True}),
        ]
        with patch.object(correction.reconstruction, "optimize_aberration_3D", side_effect=results) as fit:
            history = correction.stack_feedback(
                simulator, grid=object(), sample_z_levels=[0],
                focal_positions=[0, 1], diversity_mode=[-2, 2],
                diversity_strengths=[-0.1, 0, 0.1], rho=1e46, image_budget=13,
            )
        self.assertEqual([p["images_taken"] for p in history], [0, 6, 12])
        self.assertEqual(fit.call_count, 2)
        np.testing.assert_array_equal(simulator.commands_at_acquisition[0], [0, 0])
        np.testing.assert_allclose(simulator.commands_at_acquisition[1], [-0.04, 0.03])
        for call in fit.call_args_list:
            np.testing.assert_array_equal(call.kwargs["initial_strengths"], [0, 0])
            self.assertEqual(len(call.kwargs["diversities"]), 6)
        self.assertFalse(history[1]["optimizer_info"]["success"])

    def test_acquisition_keeps_exposure_fixed_and_returns_intensity_units(self):
        class Optics:
            def compute_image_3D(self, **kwargs):
                signal = 10 + 100 * kwargs["aberration"].strengths[0]
                return None, None, None, np.full((len(kwargs["focal_z_levels"]), 2, 2), signal)

        class ExpectedCounts:
            def __init__(self):
                self.means = []

            def poisson(self, means):
                self.means.append(means.copy())
                return means.astype(int)

        rng = ExpectedCounts()
        simulator = correction.SimulatedSLM(
            microscope=Optics(), sample=object(),
            truth=zernike.Aberration([[-2, 2]], [0.1]),
            correction_modes=[[-2, 2]], exposure=2, rng=rng,
        )
        before = simulator.acquire([0, 1])
        simulator.set_slm([-0.1])
        after = simulator.acquire([0])
        np.testing.assert_allclose(before, 20)
        np.testing.assert_allclose(after, 10)
        np.testing.assert_allclose(rng.means[0], 40)
        np.testing.assert_allclose(rng.means[1], 20)
        self.assertEqual(simulator.images_taken, 3)
        self.assertEqual(simulator.photons_detected, 400)

    def test_curve_holds_previous_committed_state(self):
        history = [
            dict(images_taken=0, residual_rms=0.15),
            dict(images_taken=6, residual_rms=0.05),
            dict(images_taken=12, residual_rms=0.01),
        ]
        np.testing.assert_allclose(
            correction.rms_at_image_counts(history, [0, 5, 6, 11, 12, 15]),
            [0.15, 0.15, 0.05, 0.05, 0.01, 0.01],
        )


if __name__ == "__main__":
    unittest.main()
