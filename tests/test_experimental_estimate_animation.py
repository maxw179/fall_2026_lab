"""Iteration selection and exact object-grid bookkeeping for saved animations."""
import unittest
import numpy as np
from experimental_10_15.animate_estimates import selected_history,sample_objects
from experimental_10_15.bead_pd import acquisition_grid


class EstimateAnimationTests(unittest.TestCase):
    def test_stride_always_includes_final_saved_iterate(self):
        history=[dict(iteration=i) for i in range(6)]
        self.assertEqual([r['iteration'] for r in selected_history(history,3)],[0,3,5])
        self.assertEqual(selected_history(history,1),history)
        with self.assertRaises(ValueError):selected_history(history,0)

    def test_sample_values_and_axial_order_are_preserved(self):
        grid=acquisition_grid(6.5,(4,4));z=np.array([.001,-.001,0.])
        values=np.arange(2*3*4*4).reshape(2,3,4,4)-40
        samples=sample_objects(values,grid,z)
        for i,sample in enumerate(samples):
            np.testing.assert_array_equal(sample.image_mask,values[i])
            np.testing.assert_array_equal(sample.z_levels,z)
            np.testing.assert_array_equal(sample.get_xy()[0],grid.get_xy()[0])


if __name__=='__main__':unittest.main()
