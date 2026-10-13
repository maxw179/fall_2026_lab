import copy
import unittest
import numpy as np
from systematic_testing.ao_animation.viewer import RecordingSLM, replay_history, _rms_trace
from utils import zernike


class TinyMicroscope:
    alpha = .5

    def compute_image_3D(self, image, aberration, focal_z_levels, diversities=None, mode=None):
        return None, None, None, np.ones((len(focal_z_levels), 5, 5))


class TestAOReplay(unittest.TestCase):
    def test_replay_preserves_state_and_uses_real_measurements(self):
        sim = RecordingSLM(TinyMicroscope(), None, zernike.Aberration([[0, 2]], [.1]),
                           [[0, 2]], 100, np.random.default_rng(2))
        initial = dict(images_taken=0, slm_strengths=np.array([0.]), residual_rms=.1)
        measured = sim.acquire([0., .001])
        sim.set_slm([-.08])
        final = dict(images_taken=2, slm_strengths=np.array([-.08]), residual_rms=.02,
                     estimated_strengths=np.array([.08]))
        rng_state = copy.deepcopy(sim.rng.bit_generator.state)
        frames, points = replay_history([initial, final], sim, include_acquisitions=True)
        self.assertEqual(len(frames), 5)
        np.testing.assert_array_equal(frames[1]['acquisition'], measured[0])
        self.assertEqual(frames[1]['rms'], .1)
        self.assertEqual(frames[3]['rms'], .1)
        self.assertEqual(frames[-1]['rms'], .02)
        self.assertEqual(sim.images_taken, 2)
        np.testing.assert_array_equal(sim.slm_strengths, [-.08])
        self.assertEqual(sim.rng.bit_generator.state, rng_state)
        self.assertIsNotNone(frames[3]['estimate'])
        self.assertEqual(len(points), 2)

    def test_all_optimizer_iterates_and_post_update_reference(self):
        sim = RecordingSLM(TinyMicroscope(), None, zernike.Aberration([[0, 2]], [.1]),
                           [[0, 2]], 100, np.random.default_rng(2))
        initial = dict(images_taken=0, slm_strengths=np.array([0.]), residual_rms=.1)
        final = dict(images_taken=2, slm_strengths=np.array([-.08]), residual_rms=.02,
                     optimizer_info=dict(history=[
                         dict(iteration=0, strengths=np.array([.03])),
                         dict(iteration=1, strengths=np.array([.08]))]))
        frames, _ = replay_history([initial, final], sim)
        self.assertEqual(len(frames), 4)
        self.assertIn('iteration 0', frames[1]['label'])
        self.assertIn('iteration 1', frames[2]['label'])
        self.assertEqual(frames[1]['rms'], .1)
        expected = zernike.zernike_RMS(zernike.Aberration([[0, 2]], [.07]), sim.microscope.alpha)
        self.assertAlmostEqual(frames[1]['optimization_rms'], expected)
        np.testing.assert_allclose(frames[-1]['estimate'][np.isfinite(frames[-1]['estimate'])], 0, atol=1e-15)

    def test_rms_hold_during_fit_and_jump_at_application(self):
        points = [dict(images_taken=0, residual_rms=.1),
                  dict(images_taken=36, residual_rms=.04)]
        self.assertEqual(_rms_trace(dict(progress=0, count=36), points), ([0., 36.], [.1, .1]))
        self.assertEqual(_rms_trace(dict(progress=1, count=36), points), ([0., 36.], [.1, .04]))

    def test_update_limit_and_invalid_history(self):
        sim = RecordingSLM(TinyMicroscope(), None, zernike.Aberration([[0, 2]], [.1]),
                           [[0, 2]], 100, np.random.default_rng(2))
        point = dict(images_taken=0, slm_strengths=np.array([0.]), residual_rms=.1)
        frames, points = replay_history([point, point], sim, max_updates=0)
        self.assertEqual(len(frames), 1)
        with self.assertRaises(ValueError):
            replay_history([], sim)
        with self.assertRaises(ValueError):
            replay_history([point], sim, max_updates=-1)


if __name__ == '__main__':
    unittest.main()
