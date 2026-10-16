"""Signed, basis-aware calibration of an independently solved zero control."""
import unittest
import numpy as np
from experimental_10_15.baseline_comparison import baseline_vector, error_histories


class BaselineTests(unittest.TestCase):
    def fixture(self):
        shared = dict(microscope_json=np.array('{"NA":1.05}'), shape=(64,64), field_of_view_um=6.5)
        target = dict(shared, modes=np.array([[2,2],[0,4]]), alpha=.9,
                      imposed_for_validation=np.array([.1,-.2]), cumulative_correction=np.array([.02,-.03]),
                      history=[dict(strengths=[.08,-.17], iteration=0)])
        # Control basis is reversed and has its own already-applied correction.
        control = dict(shared, modes=np.array([[0,4],[2,2]]), imposed_for_validation=np.zeros(2),
                       estimate=np.array([.06,-.04]))
        return target, control

    def test_signed_control_removes_bias_in_reordered_basis_after_known_correction(self):
        target, control = self.fixture()
        adjusted, errors, vectors = error_histories(target, {'system_AO': control})
        np.testing.assert_allclose(vectors['Minus System AO control'], [-.04,.06])
        np.testing.assert_allclose(adjusted['Minus System AO control'], [[.1,-.2]])
        self.assertLess(errors['Minus System AO control'][-1], 1e-15)
        self.assertGreater(errors['Original'][-1], .01)

    def test_nonzero_truth_and_mismatched_optics_or_basis_are_rejected(self):
        target, control = self.fixture()
        for change in [dict(imposed_for_validation=np.ones(2)), dict(shape=(32,32)),
                       dict(modes=np.array([[0,4],[1,3]])), dict(microscope_json=np.array('{"NA":0.8}'))]:
            with self.assertRaises(ValueError):
                baseline_vector(target, dict(control, **change))


if __name__ == '__main__':
    unittest.main()
