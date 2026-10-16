"""Independent public-model checks for the cached finite-sphere renderer."""
import unittest
import numpy as np
from experimental_10_15.bead_pd import acquisition_grid
from experimental_10_15.reference_bead_simulation import SphereModel,orient_images,ORIENTATIONS
from utils import psf,zernike


class SphereModelTests(unittest.TestCase):
    def test_all_eight_orientations_are_distinct_lossless_frame_preserving_transforms(self):
        images=np.arange(2*3*3).reshape(2,3,3)
        transformed=[orient_images(images,key) for key in ORIENTATIONS]
        self.assertEqual(len({array.tobytes() for array in transformed}),8)
        for array in transformed:
            np.testing.assert_array_equal(np.sort(array,axis=None),np.sort(images,axis=None))
            np.testing.assert_array_equal(np.sort(array[0],axis=None),np.arange(9))
        np.testing.assert_array_equal(orient_images(orient_images(images,'rotate_90'),'rotate_270'),images)
        for key in ['identity','transpose','flip_x','flip_y','rotate_180','anti_transpose']:
            np.testing.assert_array_equal(orient_images(orient_images(images,key),key),images)

    def test_cached_sphere_matches_public_vector_forward_model(self):
        microscope=psf.Microscope(3,.0013,1.333,1.05,7.2,4,3.5,15.12,12)
        modes=np.array([[2,2],[-1,3],[0,4]])
        coefficients=np.array([.07,.02,-.03]);focus=np.array([-.2,.3])
        for size in [15,16]:
            grid=acquisition_grid(6.5,(size,size))
            renderer=SphereModel(microscope,grid,modes,size=size)
            sample=psf.bead_img_3D(grid,np.linspace(-.0005,.0005,21),[0],[0],[0],[.0005])
            expected=microscope.compute_image_3D(sample,zernike.Aberration(modes,coefficients),focus/1000,mode='vector')[3]
            actual=renderer.templates(coefficients,focus)
            np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=expected.max()*1e-12)
            # Repeated axial offsets must preserve input ordering and phase.
            repeated=renderer.templates(coefficients,[.3,-.2,.3])
            np.testing.assert_allclose(repeated,actual[[1,0,1]],rtol=2e-10,atol=expected.max()*1e-12)


if __name__=='__main__':unittest.main()
