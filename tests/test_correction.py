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
    def test_cumulative_frames_have_current_residual_phase_and_original_data(self):
        simulator = AnalyticSimulator([.07, -.03])
        schedule = [[zernike.Aberration(simulator.modes, [v, (loop+1)*v])
                     for v in [-.1, 0, .1]] for loop in range(3)]
        originals, fits, stages = [], [], []
        acquire = simulator.acquire
        def capture_acquisition(focal, diversities):
            images = acquire(focal, diversities)
            originals.append(images.copy())
            return images
        simulator.acquire = capture_acquisition
        estimates = [[.2, -.04], [-.02, .15], [.03, -.01]]
        def fit(**options):
            fits.append(len(options['images']))
            np.testing.assert_array_equal(options['images'], np.concatenate(originals))
            np.testing.assert_array_equal(options['focal_z_levels'], np.tile([0]*3+[1]*3, len(fits)))
            predicted = []
            current_residual = zernike.Aberration(simulator.modes,
                                                   simulator.truth+simulator.slm_strengths)
            for phase in options['diversities']:
                total = current_residual + phase
                predicted.append([[10-np.sum(total.strengths**2)]])
            np.testing.assert_allclose(predicted, options['images'], rtol=1e-13)
            # Latest frames have no SLM offset; historical frames do.
            for phase, probe in zip(options['diversities'][-6:], schedule[len(fits)-1]*2):
                np.testing.assert_allclose(phase.strengths, probe.strengths, atol=1e-15)
            return zernike.Aberration(simulator.modes, estimates[len(fits)-1]), dict(success=True)
        with patch.object(correction.reconstruction, 'optimize_aberration_3D', side_effect=fit):
            history = correction.stack_feedback(
                simulator, None, [0], [0,1], [0,4], [-.1,0,.1], 1, 18,
                gain=.35, max_update=.1, diversity_schedule=schedule,
                reuse_acquisitions=True, on_progress=stages.append)
        self.assertEqual(fits, [6,12,18])
        self.assertEqual(simulator.images_taken, 18)
        self.assertEqual([h['fit_images'] for h in history[1:]], fits)
        self.assertEqual([e['fit_images'] for e in stages if e['stage']=='optimizing'], fits)
        for record, command in zip(history[1:], simulator.commands_at_acquisition):
            np.testing.assert_array_equal(record['acquisition_slm_strengths'], command)

    def test_multimode_schedule_changes_each_loop_and_matches_fit(self):
        simulator = AnalyticSimulator([.04, -.02])
        schedule = [[zernike.Aberration(simulator.modes, [v, 2*v])
                     for v in [-.1/(loop+1), 0, .1/(loop+1)]] for loop in range(3)]
        acquired, fitted = [], []
        def acquire(focal, diversities):
            acquired.append(np.array([a.strengths for a in diversities]))
            simulator.images_taken += len(focal)
            return np.ones((len(focal), 1, 1))
        def fit(**options):
            fitted.append(np.array([a.strengths for a in options['diversities']]))
            return zernike.EmptyAberration(simulator.modes), dict(success=True)
        simulator.acquire = acquire
        with patch.object(correction.reconstruction, 'optimize_aberration_3D', side_effect=fit):
            history = correction.stack_feedback(
                simulator, None, [0], [0, 1], [0, 4], [-.1, 0, .1], 1, 18,
                diversity_schedule=schedule, diversity_decay=.5)
        self.assertEqual(len(history), 4)
        for loop in range(3):
            expected = np.tile([a.strengths for a in schedule[loop]], (2, 1))
            np.testing.assert_array_equal(acquired[loop], expected)
            np.testing.assert_array_equal(fitted[loop], expected)
            np.testing.assert_array_equal(history[loop+1]['diversity_aberrations'][0].strengths,
                                          schedule[loop][0].strengths)

    def test_short_diversity_schedule_rejected_before_acquisition(self):
        simulator = AnalyticSimulator([.04])
        with self.assertRaisesRegex(ValueError, 'shorter'):
            correction.stack_feedback(simulator, None, [0], [0], [0, 4], [-.1, 0, .1],
                                      1, 2, diversity_schedule=[[zernike.EmptyAberration()]])
        self.assertEqual(simulator.images_taken, 0)

    def test_joint_diversities_match_fit_and_image_budget(self):
        simulator = AnalyticSimulator([.04])
        schedule = [[0, 4], [-2, 2], [2, 2]]
        acquired, fitted = [], []

        def acquire(focal, diversities):
            acquired.append([(list(a.modes[0]), float(a.strengths[0])) for a in diversities])
            np.testing.assert_array_equal(focal, [0]*9 + [1]*9)
            simulator.images_taken += len(focal)
            return np.ones((len(focal), 1, 1))

        def fit(**options):
            fitted.append([(list(a.modes[0]), float(a.strengths[0])) for a in options['diversities']])
            return zernike.EmptyAberration(simulator.modes), dict(success=True)

        simulator.acquire = acquire
        with patch.object(correction.reconstruction, 'optimize_aberration_3D', side_effect=fit):
            history = correction.stack_feedback(
                simulator, None, [0], [0, 1], [0, 4], [-.1, 0, .1], 1, 37,
                diversity_modes=schedule, diversity_strategy='joint', diversity_decay=.5)
        self.assertEqual(acquired, fitted)
        self.assertEqual(len(history), 3)
        self.assertEqual(simulator.images_taken, 36)
        for cycle, records in enumerate(acquired):
            expected = [(m, s*.5**cycle) for _ in range(2)
                        for m in schedule for s in [-.1, 0, .1]]
            self.assertEqual(records, expected)
            self.assertEqual(history[cycle+1]['diversity_modes'], schedule)

    def test_diversity_mode_cycle_matches_acquisition_and_fit(self):
        simulator = AnalyticSimulator([.04])
        schedule = [[0, 4], [-2, 2], [2, 2]]
        acquired, fitted = [], []

        def acquire(focal, diversities):
            acquired.append([(np.asarray(a.modes).tolist(), np.asarray(a.strengths).tolist()) for a in diversities])
            simulator.images_taken += len(focal)
            return np.ones((len(focal), 1, 1))

        simulator.acquire = acquire

        def fit(**options):
            fitted.append([(np.asarray(a.modes).tolist(), np.asarray(a.strengths).tolist()) for a in options['diversities']])
            return zernike.EmptyAberration(simulator.modes), dict(success=True)

        with patch.object(correction.reconstruction, 'optimize_aberration_3D', side_effect=fit):
            history = correction.stack_feedback(
                simulator, None, [0], [0, 1], [0, 4], [-.1, 0, .1], 1, 24,
                diversity_modes=schedule, diversity_decay=.5, min_diversity_strength=.02)
        expected_modes = schedule + [schedule[0]]
        self.assertEqual([h['diversity_mode'] for h in history[1:]], expected_modes)
        self.assertEqual(acquired, fitted)
        for cycle, records in enumerate(acquired):
            self.assertEqual(len(records), 6)
            self.assertTrue(all(mode == [expected_modes[cycle]] for mode, strength in records))
        np.testing.assert_allclose([h['diversity_strengths'][2] for h in history[1:]],
                                   [.1, .05, .025, .02])
        self.assertEqual(simulator.images_taken, 24)

    def test_decaying_diversities_match_acquisition_and_fit(self):
        simulator = AnalyticSimulator([.04])
        acquired, fitted = [], []
        original = simulator.acquire

        def acquire(focal, diversities):
            acquired.append([float(a.strengths[0]) for a in diversities])
            return original(focal, diversities)

        simulator.acquire = acquire

        def fit(**options):
            fitted.append([float(a.strengths[0]) for a in options['diversities']])
            return zernike.EmptyAberration(simulator.modes), dict(success=True)

        with patch.object(correction.reconstruction, 'optimize_aberration_3D', side_effect=fit):
            history = correction.stack_feedback(
                simulator, None, [0], [0], [-2, 2], [-.1, 0, .1], 1, 12,
                diversity_decay=.5, min_diversity_strength=.03)
        expected = [[-.1, 0, .1], [-.05, 0, .05], [-.03, 0, .03], [-.03, 0, .03]]
        np.testing.assert_allclose(acquired, expected)
        np.testing.assert_array_equal(acquired, fitted)
        np.testing.assert_allclose([h['diversity_strengths'] for h in history[1:]], expected)
        self.assertEqual(simulator.images_taken, 12)

    def test_iteration_rms_error_compares_to_current_residual(self):
        from types import SimpleNamespace
        simulator = AnalyticSimulator([.04, -.03])
        simulator.microscope = SimpleNamespace(alpha=.8)
        simulator.set_slm([-.02, .01])
        simulator._residual = lambda: zernike.Aberration(
            simulator.modes, simulator.truth + simulator.slm_strengths)
        events = []

        def fit(**options):
            estimate = simulator._residual()
            options['on_iteration'](dict(iteration=0, loss=1.,
                                        strengths=estimate.strengths.copy()))
            return estimate, dict(success=True)

        with patch.object(correction.reconstruction, 'optimize_aberration_3D', side_effect=fit):
            correction.stack_feedback(
                simulator, None, [0], [0], [-2, 2], [-.1, 0, .1], 1, 3,
                optimizer_kwargs=dict(on_iteration=events.append))
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]['aberration_rms_error'], 0., places=14)
        self.assertEqual(events[0]['loss'], 1.)

    def test_progress_reports_acquisition_fit_and_booth_probes(self):
        simulator = AnalyticSimulator([0.04, -0.03])
        events, updates = [], []
        fit_result = (zernike.Aberration(simulator.modes, [0.04, -0.03]), {"success": True})
        with patch.object(correction.reconstruction, "optimize_aberration_3D", return_value=fit_result):
            correction.stack_feedback(
                simulator, grid=None, sample_z_levels=[0], focal_positions=[0],
                diversity_mode=[-2, 2], diversity_strengths=[-.1, 0, .1],
                rho=1, image_budget=3, on_progress=events.append, on_update=updates.append)
        self.assertEqual([e['stage'] for e in events], ['acquiring', 'optimizing'])
        self.assertEqual([e['images_taken'] for e in events], [0, 3])
        self.assertEqual([e['update'] for e in events], [1, 1])
        self.assertEqual(len(updates), 1)
        events.clear()
        simulator = AnalyticSimulator([0.04, -0.03])
        correction.booth_feedback(simulator, [0], .1, 5, on_progress=events.append)
        self.assertEqual([e['stage'] for e in events], ['acquiring baseline', 'probing', 'probing'])
        self.assertEqual([e['mode_index'] for e in events[1:]], [1, 2])
        self.assertEqual(simulator.images_taken, 5)

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
