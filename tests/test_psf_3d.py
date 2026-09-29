import unittest
import numpy as np
from scipy.signal import fftconvolve

from utils.psf import Arbitrary_Grid, Image_Mask, Image_Mask_3D, Microscope
from utils.zernike import Aberration


class VolumeTests(unittest.TestCase):
    def setUp(self):
        self.m = Microscope(3, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, 24)
        self.a = Aberration([[0, 2]], [.15])

    def plane(self, z, value=1):
        grid = Arbitrary_Grid(.006, .008, 6, 7, .01, -.02, z)
        mask = np.zeros((6, 7))
        mask[2, 3] = value
        return Image_Mask(grid, mask)

    def test_container_validation(self):
        for planes in ([], [self.plane(0), self.plane(0)], [self.plane(np.nan)]):
            with self.assertRaises(ValueError):
                Image_Mask_3D(planes)
        other = self.plane(.001)
        other.x_offset += 1
        with self.assertRaises(ValueError):
            Image_Mask_3D([self.plane(0), other])
        volume = Image_Mask_3D([self.plane(.002), self.plane(-.001)])
        self.assertEqual(volume.image_mask.shape, (2, 6, 7))
        np.testing.assert_array_equal(volume.z_levels, [.002, -.001])

    def test_psf_stack_matches_slices(self):
        for mode in ('scalar', 'vector'):
            grid = self.plane(99)
            x, y, z, stack = self.m.compute_PSF_3D(grid, self.a, [-.001, .002], mode)
            for i, level in enumerate(z):
                expected = self.m.compute_PSF(self.plane(level), self.a, mode)[2]
                np.testing.assert_allclose(stack[i], expected)
            np.testing.assert_array_equal(x, grid.get_xy()[0])
            self.assertEqual(stack.shape, (2, len(x), len(y)))

    def test_discrete_forward_sum(self):
        volume = Image_Mask_3D([self.plane(-.001, 2), self.plane(.0025, 3)])
        for mode in ('scalar', 'vector'):
            x, y, z, stack = self.m.compute_image_3D(volume, self.a, [0, .001], mode)
            for k, focus in enumerate(z):
                expected = np.zeros((6, 7))
                for plane in volume.planes:
                    # Independent kernel construction: 7 x 7, original spacing,
                    # zero lateral offset and the formalism's signed defocus.
                    grid = Arbitrary_Grid(.006 * 6 / 5, .008, 7, 7,
                                          0, 0, plane.z_level - focus)
                    kernel = self.m.compute_PSF(grid, self.a, mode)[2]
                    expected += fftconvolve(plane.image_mask, kernel, mode='same')
                np.testing.assert_allclose(stack[k], expected, rtol=1e-10, atol=abs(expected).max()*1e-12)
            self.assertEqual(stack.shape, (2, 6, 7))
            np.testing.assert_array_equal(x, volume.get_xy()[0])

    def test_single_plane_and_axial_translation(self):
        plane = self.plane(.001)
        for mode in ('scalar', 'vector'):
            actual = self.m.compute_image_3D(Image_Mask_3D([plane]), self.a, [0], mode)[3][0]
            expected = self.m.compute_image(plane, self.a, mode, norm=False)[2]
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=abs(expected).max()*1e-12)
            shifted = self.m.compute_image_3D(
                Image_Mask_3D([self.plane(.011)]), self.a, [.01], mode)[3][0]
            np.testing.assert_allclose(actual, shifted, atol=abs(actual).max()*1e-12)
        z = self.m.compute_image_3D(Image_Mask_3D([plane]), self.a)[2]
        np.testing.assert_array_equal(z, [.001])
        for levels in ([], [np.nan], [[0]]):
            with self.assertRaises(ValueError):
                self.m.compute_image_3D(Image_Mask_3D([plane]), self.a, levels)

    def test_diversities_match_separate_acquisitions(self):
        volume = Image_Mask_3D([self.plane(-.001), self.plane(.001)])
        focuses = [0, 0, .001]
        diversities = [Aberration([[0, 4]], [s]) for s in [0, -.15, .15]]
        for mode in ('scalar', 'vector'):
            stack = self.m.compute_image_3D(
                volume, self.a, focuses, mode, diversities=diversities)[3]
            for k, (focus, diversity) in enumerate(zip(focuses, diversities)):
                expected = self.m.compute_image_3D(
                    volume, self.a + diversity, [focus], mode)[3][0]
                np.testing.assert_allclose(stack[k], expected)
        for invalid in ([], diversities[:1], [None] * 3):
            with self.assertRaises(ValueError):
                self.m.compute_image_3D(volume, self.a, focuses, diversities=invalid)


if __name__ == '__main__':
    unittest.main()
