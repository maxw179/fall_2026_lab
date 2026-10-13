"""Statistical checks for the camera-noise experiment's physical unit model."""
import unittest
import numpy as np
from systematic_testing.camera_noise import simulate_camera, camera_metrics


class CameraNoiseTests(unittest.TestCase):
    def test_electron_mean_and_variance(self):
        signal = np.full((2000, 8, 8), 20.)
        r = simulate_camera(signal, background_e=5, dark_e=1, read_sigma_e=3,
                            pedestal_adu=100, gain_adu_per_e=2)
        data = r['pedestal_only_e']
        self.assertAlmostEqual(float(data.mean()), 26, delta=.08)
        self.assertAlmostEqual(float(data.var()), 35, delta=.5)
        np.testing.assert_allclose(r['raw_adu'], 100+2*data)

    def test_camera_effects_preserve_signal_photon_draws(self):
        signal = np.full((3, 8, 8), 20.)
        base = simulate_camera(signal)
        noisy = simulate_camera(signal, read_sigma_e=3, background_e=5,
                                offset_pattern_e=2, pattern='rows')
        np.testing.assert_array_equal(base['signal_counts'], noisy['signal_counts'])
        np.testing.assert_array_equal(noisy['raw_adu'], simulate_camera(
            signal, read_sigma_e=3, background_e=5,
            offset_pattern_e=2, pattern='rows')['raw_adu'])

    def test_offset_pattern_fixed_across_frames_and_removed_by_calibration(self):
        signal = np.zeros((3, 8, 8))
        r = simulate_camera(signal, pedestal_adu=64, pattern='checkerboard', offset_pattern_e=2)
        np.testing.assert_array_equal(r['pedestal_only_e'][0], r['pedestal_only_e'][2])
        np.testing.assert_array_equal(r['known_maps_e'], np.zeros_like(signal))
        self.assertEqual(set(np.unique(r['raw_adu'])), {62., 66.})

    def test_sensitivity_changes_photon_mean_and_calibration(self):
        signal = np.full((2000, 8, 8), 20.)
        r = simulate_camera(signal, pattern='rows', sensitivity_fraction=.2)
        means = r['signal_counts'].mean(axis=0)
        np.testing.assert_allclose(means[:, ::2], 24, atol=.5)
        np.testing.assert_allclose(means[:, 1::2], 16, atol=.5)
        np.testing.assert_allclose(r['known_maps_e'].mean(axis=0), 20, atol=.5)

    def test_metrics_distinguish_bias_from_temporal_noise(self):
        signal = np.full((2, 8, 8), 20.)
        r = simulate_camera(signal, background_e=5, read_sigma_e=3)
        temporal, total = camera_metrics(signal, r, 'pedestal_only', np.ones((8,8), bool))
        np.testing.assert_allclose(temporal, 20/np.sqrt(34))
        self.assertAlmostEqual(total, 20/np.sqrt(34+25))
        _, corrected = camera_metrics(signal, r, 'known_maps', np.ones((8,8), bool))
        self.assertAlmostEqual(corrected, 20/np.sqrt(34))


if __name__ == '__main__':
    unittest.main()
