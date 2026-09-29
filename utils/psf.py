from __future__ import annotations
from typing import Literal, Sequence
from utils.rw import *
from utils.zernike import *
import numpy as np
from scipy.signal import fftconvolve

class Arbitrary_Grid():
    """
    Initializes the object.
    Params:
        L_ffp_x (float): x extent [mm]
        L_ffp_y (float): y extent [mm]
        grid_ffp_x (int): x sample count
        grid_ffp_y (int): y sample count
        x_offset (float): x center [mm]
        y_offset (float): y center [mm]
        z_level (float): axial position [mm]
    Returns:
        None
    """
    def __init__(self,
                 L_ffp_x: float,
                 L_ffp_y: float,
                 grid_ffp_x: int, 
                 grid_ffp_y: int, 
                 x_offset: float, 
                 y_offset: float,
                 z_level: float):
        self.L_ffp_x = L_ffp_x
        self.L_ffp_y = L_ffp_y
        self.grid_ffp_x = grid_ffp_x
        self.grid_ffp_y = grid_ffp_y
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_level = z_level
        self.get_xy()  # Validate focal-plane sample counts at construction.

    """
    Gets the lateral sample coordinates.
    Params:
        None
    Returns:
        x, y: lateral coordinate arrays [mm]
    """
    def get_xy(self):
        x = get_ffp_axis(self.L_ffp_x, self.grid_ffp_x, self.x_offset)
        y = get_ffp_axis(self.L_ffp_y, self.grid_ffp_y, self.y_offset)
        return x,y
    
    """
    Creates an empty lateral sample array.
    Params:
        None
    Returns:
        grid: zero array with shape (x, y)
    """
    def get_grid(self):
        x, y = self.get_xy()
        return np.zeros((len(x), len(y)))

class Square_Grid(Arbitrary_Grid):
    """
    Initializes the object.
    Params:
        L_ffp (float): lateral extent [mm]
        grid_ffp (int): lateral sample count
        x_offset (float): x center [mm]
        y_offset (float): y center [mm]
        z_level (float): axial position [mm]
    Returns:
        None
    """
    def __init__(self, 
                 L_ffp: float, 
                 grid_ffp: int, 
                 x_offset: float, 
                 y_offset: float,
                 z_level: float):
        super().__init__(L_ffp, L_ffp, grid_ffp, grid_ffp, x_offset, y_offset, z_level)

class Centered_Square_Grid(Square_Grid):
    """
    Initializes the object.
    Params:
        L_ffp (float): lateral extent [mm]
        grid_ffp (int): lateral sample count
        z_level (float): axial position [mm]
    Returns:
        None
    """
    def __init__(self, L_ffp: float, grid_ffp: int, z_level: float):
        super().__init__(L_ffp, grid_ffp, 0.0, 0.0, z_level)

class Image_Mask(Arbitrary_Grid):   
    """
    Initializes the object.
    Params:
        grid (Arbitrary_Grid): lateral sampling grid
        image_mask (np.ndarray): two-dimensional sample values
    Returns:
        None
    """
    def __init__(self,
                 grid: Arbitrary_Grid,
                 image_mask: np.ndarray):
        super().__init__(grid.L_ffp_x, 
                         grid.L_ffp_y, 
                         grid.grid_ffp_x, 
                         grid.grid_ffp_y,
                         grid.x_offset, 
                         grid.y_offset, 
                         grid.z_level)
        self.image_mask = image_mask 
        if np.shape(self.image_mask) != np.shape(self.get_grid()):
            raise RuntimeError("Shape of the image mask differs from the L_ffp, grid_ffp parameters.")

class Bead_Image(Image_Mask):
    """
    Initializes the object.
    Params:
        grid (Arbitrary_Grid): lateral sampling grid
        xs (Sequence[float] | np.ndarray): bead x coordinates [mm]
        ys (Sequence[float] | np.ndarray): bead y coordinates [mm]
        bead_sizes (Sequence[float] | np.ndarray): bead radii [mm]
    Returns:
        None
    """
    def __init__(self, 
                 grid: Arbitrary_Grid,
                 xs: Sequence[float] | np.ndarray, 
                 ys: Sequence[float] | np.ndarray, 
                 bead_sizes: Sequence[float] | np.ndarray):
        _, _, image_mask = bead_img(grid, xs, ys, bead_sizes)
        super().__init__(grid, image_mask)


class Image_Mask_3D:
    """
    Stores 2D sample planes on a shared lateral grid in input order.
    Params:
        planes: Image_Mask objects at distinct z levels [mm], with values
            including any desired axial integration weights
    Returns:
        An Image_Mask_3D object
    """
    """
    Validates and stores the sample planes.
    Params:
        planes (Sequence[Image_Mask]): nonempty sequence of Image_Mask objects sharing a lateral grid
    Returns:
        None
    """
    def __init__(self, planes: Sequence[Image_Mask]):
        self.planes = tuple(planes)
        if not self.planes or not all(isinstance(p, Image_Mask) for p in self.planes):
            raise ValueError("Provide at least one Image_Mask plane.")
        attributes = ("L_ffp_x", "L_ffp_y", "grid_ffp_x", "grid_ffp_y",
                      "x_offset", "y_offset")
        first = self.planes[0]
        if any(any(getattr(p, name) != getattr(first, name) for name in attributes)
               for p in self.planes):
            raise ValueError("All sample planes must share the same lateral grid.")
        z = self.z_levels
        if not np.all(np.isfinite(z)) or len(np.unique(z)) != len(z):
            raise ValueError("Sample z levels must be finite and distinct.")

    """
    Gets the axial positions of the sample planes in input order.
    Params:
        None
    Returns:
        z_levels: one-dimensional array of sample positions [mm]
    """
    @property
    def z_levels(self):
        return np.array([p.z_level for p in self.planes], dtype=float)

    """
    Stacks the sample masks in input order.
    Params:
        None
    Returns:
        image_mask: sample array with shape (planes, x, y)
    """
    @property
    def image_mask(self):
        return np.stack([p.image_mask for p in self.planes])

    """
    Gets the shared lateral coordinates of the sample planes.
    Params:
        None
    Returns:
        x: x coordinates [mm]
        y: y coordinates [mm]
    """
    def get_xy(self):
        return self.planes[0].get_xy()

    """
    Gets coordinate grids indexed as (plane, x pixel, y pixel).
    The indexing matches the image stack returned by compute_image_3D.
    Params:
        focal_z_levels (Sequence[float] | np.ndarray | None): focal positions
            [mm]; None uses the sample positions in their stored order
    Returns:
        X (np.ndarray): x coordinates with shape (planes, x, y) [mm]
        Y (np.ndarray): y coordinates with shape (planes, x, y) [mm]
        Z (np.ndarray): z coordinates with shape (planes, x, y) [mm]
    """
    def get_xyz(self, focal_z_levels: Sequence[float] | np.ndarray | None = None):
        x, y = self.get_xy()
        z = _z_array(self.z_levels if focal_z_levels is None else focal_z_levels)
        Z, X, Y = np.meshgrid(z, x, y, indexing="ij")
        return X, Y, Z


"""
Copies a lateral grid at a specified axial position.
Params:
    grid (Arbitrary_Grid): grid supplying lateral coordinates and sample counts
    z (float): axial position [mm]
Returns:
    Arbitrary_Grid: grid at the specified position
"""
def _grid_at_z(grid: Arbitrary_Grid, z: float):
    return Arbitrary_Grid(grid.L_ffp_x, grid.L_ffp_y,
                          grid.grid_ffp_x, grid.grid_ffp_y,
                          grid.x_offset, grid.y_offset, z)


"""
Creates a centered, odd-sized kernel grid at the image pixel spacing.
Params:
    grid (Arbitrary_Grid): image grid supplying pixel spacing and axial position
Returns:
    Arbitrary_Grid: convolution grid with zero lateral offsets
"""
def _convolution_grid(grid: Arbitrary_Grid):
    # Odd kernel sizes include zero at the original image pixel spacing.
    nx = grid.grid_ffp_x + (grid.grid_ffp_x % 2 == 0)
    ny = grid.grid_ffp_y + (grid.grid_ffp_y % 2 == 0)
    return Arbitrary_Grid(
        grid.L_ffp_x * (nx - 1) / (grid.grid_ffp_x - 1),
        grid.L_ffp_y * (ny - 1) / (grid.grid_ffp_y - 1),
        nx, ny, 0.0, 0.0, grid.z_level,
    )


"""
Converts axial positions to a nonempty, finite one-dimensional array.
Params:
    z_levels (Sequence[float] | np.ndarray): sequence of axial positions [mm]
Returns:
    z: floating-point array of axial positions [mm]
"""
def _z_array(z_levels: Sequence[float] | np.ndarray):
    z = np.asarray(z_levels, dtype=float)
    if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)):
        raise ValueError("Provide a nonempty, finite 1D sequence of z levels.")
    return z
    
class Microscope():
    """
    Initializes the object.
    Params:
        N_order (int): multiphoton excitation order
        lambd (float): vacuum wavelength [mm]
        n (float): refractive index
        num_apt (float): numerical aperture
        f (float): focal length [mm]
        mag (float): beam magnification
        w_0 (float): beam waist [mm]
        L_bfp (float): back focal plane extent [mm]
        grid_bfp (int): back focal plane sample count
    Returns:
        None
    """
    def __init__(self, 
                 N_order: int, 
                 lambd: float, 
                 n: float, 
                 num_apt: float, 
                 f: float, 
                 mag: float, 
                 w_0: float,
                 L_bfp: float,
                 grid_bfp: int,
                 ):
        if not 0 < num_apt < n:
            raise ValueError("Numerical aperture must satisfy 0 < NA < n.")
        self.N_order = N_order 
        self.lambd =lambd
        self.n = n
        self.num_apt = num_apt 
        self.f = f 
        self.mag = mag
        self.w_0 = w_0
        #the length of the back focal plane
        self.L_bfp = L_bfp
        self.grid_bfp = grid_bfp
        self.k = (2*n*np.pi)/lambd 
        self.alpha = np.arcsin(num_apt / n)

    """
    Computes the pupil aberration phase map.
    Params:
        aberration (Aberration): pupil aberration
    Returns:
        phase: pupil phase array [rad]
    """
    def compute_phase_map(self, aberration: Aberration):
        return get_phase_map(
            alpha = self.alpha,
            f = self.f, 
            n = self.n, 
            L_bfp = self.L_bfp, 
            aberration = aberration, 
            grid_bfp = self.grid_bfp
        )

    """
    Computes the complex pupil field.
    Params:
        aberration (Aberration): pupil aberration
        gaussian (bool): whether to apply Gaussian illumination
    Returns:
        pupil: complex pupil array
    """
    def compute_pupil_function(self,
                               aberration: Aberration,
                               gaussian: bool = True):
        return get_pupil_function(
            alpha=self.alpha,
            mag=self.mag,
            w_0=self.w_0,
            f=self.f,
            n=self.n,
            L_bfp=self.L_bfp,
            aberration=aberration,
            grid_bfp=self.grid_bfp,
            gaussian = gaussian
        )
    
    """
    Computes the complex scalar focal field.
    Params:
        grid (Arbitrary_Grid): lateral sampling grid
        aberration (Aberration): pupil aberration
    Returns:
        h: complex scalar field array
    """
    def compute_scalar_h(self, 
                         grid: Arbitrary_Grid,
                         aberration: Aberration):
        return get_scalar_h(
            L_ffp_x=grid.L_ffp_x,
            L_ffp_y=grid.L_ffp_y,
            grid_ffp_x=grid.grid_ffp_x,
            grid_ffp_y=grid.grid_ffp_y,
            x_offset=grid.x_offset,
            y_offset=grid.y_offset,
            alpha=self.alpha,
            k=self.k,
            f=self.f,
            n=self.n,
            mag=self.mag,
            w_0=self.w_0,
            L_bfp=self.L_bfp,
            aberration=aberration,
            grid_bfp=self.grid_bfp,
            z = grid.z_level
        )

    """
    Computes the scalar multiphoton PSF.
    Params:
        grid (Arbitrary_Grid): lateral sampling grid
        aberration (Aberration): pupil aberration
    Returns:
        x, y, PSF: coordinates [mm] and intensity array
    """
    def compute_scalar_psf(self,
                           grid: Arbitrary_Grid,
                           aberration: Aberration):
        h = self.compute_scalar_h(grid, aberration)
        x, y = grid.get_xy()
        return x, y, np.abs(h)**(2 * self.N_order)

    """
    Computes the scalar or vector multiphoton PSF.
    Params:
        grid (Arbitrary_Grid): lateral sampling grid
        aberration (Aberration): pupil aberration
        mode (Literal["vector", "scalar"]): optical model
    Returns:
        x, y, PSF: coordinates [mm] and intensity array
    """
    def compute_PSF(self,
                    grid: Arbitrary_Grid, 
                    aberration: Aberration,
                    mode: Literal["vector", "scalar"] = "vector"):
        
        if mode == "vector":
            _, _, I = rw_fast(
                L_ffp_x=grid.L_ffp_x,
                L_ffp_y=grid.L_ffp_y,
                grid_ffp_x=grid.grid_ffp_x,
                grid_ffp_y=grid.grid_ffp_y,
                x_offset=grid.x_offset,
                y_offset=grid.y_offset,
                alpha=self.alpha,
                k=self.k,
                f=self.f,
                n=self.n,
                mag=self.mag, 
                w_0=self.w_0,
                L_bfp=self.L_bfp, 
                aberration=aberration,
                grid_bfp=self.grid_bfp,
                N_order=self.N_order,
                z=grid.z_level
            )
            x, y = grid.get_xy()
            return x, y, I
        elif mode == "scalar":
            return self.compute_scalar_psf(grid, aberration)

        else:
            raise RuntimeError("Invalid mode to compute PSF")
    
    """
    Convolves a sample with its PSF.
    Params:
        image (Image_Mask): sample to convolve
        aberration (Aberration): pupil aberration
        mode (Literal["vector", "scalar"]): optical model
        norm (bool): whether to normalize the PSF sum
    Returns:
        x, y, image: coordinates [mm] and convolved image
    """
    def compute_image(self, 
                      image: Image_Mask, 
                      aberration: Aberration,
                      mode: Literal["vector", "scalar"] = "vector",
                      norm: bool = True):

        # Convolution kernels use displacements from zero, not object positions.
        # Odd kernel sizes include zero while preserving the image pixel spacing.
        kernel_grid = _convolution_grid(image)
        _, _, PSF = self.compute_PSF(kernel_grid, aberration, mode)
        x, y = image.get_xy()
        return x, y, psf_convolve(image.image_mask, PSF, norm)

    """
    Computes unnormalized PSF slices at specified defocus distances.
    Params:
        grid (Arbitrary_Grid): grid supplying lateral coordinates and sample counts
        aberration (Aberration): aberration used to construct the pupil phase
        z_levels (Sequence[float] | np.ndarray): defocus distances replacing grid.z_level [mm]
        mode (Literal["vector", "scalar"]): optical model, "vector" or "scalar"
    Returns:
        x: x coordinates [mm]
        y: y coordinates [mm]
        z: defocus distances in input order [mm]
        PSF: raw intensity array with shape (planes, x, y)
    """
    def compute_PSF_3D(self, grid: Arbitrary_Grid, aberration: Aberration, z_levels: Sequence[float] | np.ndarray, mode: Literal["vector", "scalar"]="vector"):
        z = _z_array(z_levels)
        slices = [self.compute_PSF(_grid_at_z(grid, level), aberration, mode)[2]
                  for level in z]
        x, y = grid.get_xy()
        return x, y, z, np.stack(slices)

    """
    Computes an image stack by summing 2D sample-plane convolutions.
    At each focus z_k, sum f_i convolved with PSF(z_i - z_k).
    Arbitrary plane spacing is supported; no dz factor is applied.
    PSFs retain raw intensities to preserve relative axial signal.
    Lateral convolution uses zero padding and crops to the sample grid.
    Params:
        image (Image_Mask_3D): Image_Mask_3D containing the sample planes
        aberration (Aberration): unknown aberration shared by the acquisition
        focal_z_levels (Sequence[float] | np.ndarray | None): focal positions [mm]; None uses the sample positions
        mode (Literal["vector", "scalar"]): optical model, "vector" or "scalar"
        diversities (Sequence[Aberration] | None): known additional aberration per
            image; None uses zero diversity. Repeat focal positions to acquire
            multiple diversities at the same focus.
    Returns:
        x: x coordinates [mm]
        y: y coordinates [mm]
        focal_z: focal positions in requested order [mm]
        images: intensity array with shape (focal planes, x, y)
    """
    def compute_image_3D(self, image: Image_Mask_3D, aberration: Aberration,
                         focal_z_levels: Sequence[float] | np.ndarray | None=None, mode: Literal["vector", "scalar"]="vector",
                         diversities: Sequence[Aberration] | None = None):
        focal_z = _z_array(image.z_levels if focal_z_levels is None else focal_z_levels)
        if diversities is None:
            diversities = [EmptyAberration()] * len(focal_z)
        if len(diversities) != len(focal_z) or not all(isinstance(a, Aberration) for a in diversities):
            raise ValueError('Provide one diversity Aberration per acquired image.')
        kernel_grid = _convolution_grid(image.planes[0])
        images = []
        for focus, diversity in zip(focal_z, diversities):
            total_aberration = aberration + diversity
            result = np.zeros_like(image.planes[0].image_mask, dtype=float)
            for plane in image.planes:
                grid = _grid_at_z(kernel_grid, plane.z_level - focus)
                _, _, PSF = self.compute_PSF(grid, total_aberration, mode)
                result += psf_convolve(plane.image_mask, PSF, norm=False)
            images.append(result)
        images = np.stack(images)
        x, y = image.get_xy()
        return x, y, focal_z, images

"""
Creates a binary image of circular beads.
Params:
    grid (Arbitrary_Grid): lateral sampling grid
    xs (Sequence[float] | np.ndarray): bead x coordinates [mm]
    ys (Sequence[float] | np.ndarray): bead y coordinates [mm]
    bead_sizes (Sequence[float] | np.ndarray): bead radii [mm]
Returns:
    x, y, image: coordinates [mm] and binary bead image
"""
def bead_img(grid: Arbitrary_Grid, 
             xs: Sequence[float] | np.ndarray, 
             ys: Sequence[float] | np.ndarray, 
             bead_sizes: Sequence[float] | np.ndarray):
    if not (len(xs) == len(ys) == len(bead_sizes)):
        raise RuntimeError("xs, ys, and bead_sizes must have the same length.")

    num_pixels_x = grid.grid_ffp_x

    num_pixels_y = grid.grid_ffp_y

    x, y = grid.get_xy()
    img = np.zeros((num_pixels_x, num_pixels_y))
    for k in range(len(xs)):
        x_center = xs[k]
        y_center = ys[k]
        bead_size = bead_sizes[k]
        for i in range(num_pixels_x):
            for j in range(num_pixels_y):
                x_img = x[i]
                y_img = y[j]

                distance = np.sqrt((x_center - x_img)**2 + (y_center - y_img)**2)
                if distance <= bead_size:
                    img[i, j] = 1
    return x, y, img

"""
Creates a binary image of randomly positioned beads.
Params:
    grid (Arbitrary_Grid): lateral sampling grid
    num_beads (int): number of beads
    bead_size (float): common bead radius [mm]
Returns:
    x, y, image: coordinates [mm] and binary bead image
"""
def random_bead_img(grid: Arbitrary_Grid, 
                    num_beads: int,
                    bead_size: float):
    rng = np.random.default_rng(15)
    x_max = grid.x_offset + grid.L_ffp_x/2
    x_min = grid.x_offset - grid.L_ffp_x/2
    y_max = grid.y_offset + grid.L_ffp_y/2
    y_min = grid.y_offset - grid.L_ffp_y/2

    xs = rng.uniform(x_min, x_max, num_beads)
    ys = rng.uniform(y_min, y_max, num_beads)
    bead_sizes = np.zeros(num_beads) + bead_size 
    return bead_img(grid,xs, ys,bead_sizes)


"""
Creates a stack of binary spherical bead cross-sections at specified planes.
Overlapping beads have value one. Beads are clipped at the grid boundaries.
Params:
    grid (Arbitrary_Grid): grid supplying lateral coordinates; grid.z_level is ignored
    z_levels (Sequence[float] | np.ndarray): distinct sample-plane positions in input order [mm]
    xs (Sequence[float] | np.ndarray): bead center x coordinates [mm]
    ys (Sequence[float] | np.ndarray): bead center y coordinates [mm]
    zs (Sequence[float] | np.ndarray): bead center z coordinates [mm]
    bead_sizes (Sequence[float] | np.ndarray): nonnegative bead radii [mm], one per center
Returns:
    Image_Mask_3D: sample containing masks with shape (planes, x, y)
"""
def bead_img_3D(grid: Arbitrary_Grid, z_levels: Sequence[float] | np.ndarray, xs: Sequence[float] | np.ndarray, ys: Sequence[float] | np.ndarray, zs: Sequence[float] | np.ndarray, bead_sizes: Sequence[float] | np.ndarray):
    z = _z_array(z_levels)
    if len(np.unique(z)) != len(z):
        raise ValueError("Sample z levels must be distinct.")
    coordinates = [np.asarray(values, dtype=float) for values in (xs, ys, zs, bead_sizes)]
    if any(values.ndim != 1 or not np.all(np.isfinite(values)) for values in coordinates):
        raise ValueError("Bead coordinates and radii must be finite 1D sequences.")
    xs, ys, zs, radii = coordinates
    if not len(xs) == len(ys) == len(zs) == len(radii):
        raise ValueError("xs, ys, zs, and bead_sizes must have the same length.")
    if np.any(radii < 0):
        raise ValueError("Bead radii must be nonnegative.")
    x, y = grid.get_xy()
    planes = []
    for level in z:
        mask = np.zeros((len(x), len(y)))
        for xc, yc, zc, radius in zip(xs, ys, zs, radii):
            cross_section_squared = radius**2 - (level - zc)**2
            if cross_section_squared >= 0:
                inside = (x[:, None] - xc)**2 + (y[None, :] - yc)**2 <= cross_section_squared
                mask[inside] = 1
        planes.append(Image_Mask(_grid_at_z(grid, level), mask))
    return Image_Mask_3D(planes)


"""
Creates spherical beads with uniformly random centers within a stack's bounds.
Centers may lie between planes; spheres are clipped at the sample boundaries.
Params:
    grid (Arbitrary_Grid): grid supplying lateral coordinates and offsets
    z_levels (Sequence[float] | np.ndarray): distinct sample-plane positions [mm]
    num_beads (int): nonnegative integer number of beads
    bead_size (float): common nonnegative bead radius [mm]
    rng (np.random.Generator | None): optional NumPy random generator; None uses seed 15
Returns:
    Image_Mask_3D: randomly dispersed bead sample in input plane order
"""
def random_bead_img_3D(grid: Arbitrary_Grid, z_levels: Sequence[float] | np.ndarray, num_beads: int, bead_size: float, rng: np.random.Generator | None=None):
    z = _z_array(z_levels)
    if not isinstance(num_beads, (int, np.integer)) or num_beads < 0:
        raise ValueError("num_beads must be a nonnegative integer.")
    if not np.isfinite(bead_size) or bead_size < 0:
        raise ValueError("Bead radius must be finite and nonnegative.")
    if rng is None:
        rng = np.random.default_rng(15)
    x, y = grid.get_xy()
    xs = rng.uniform(np.min(x), np.max(x), num_beads)
    ys = rng.uniform(np.min(y), np.max(y), num_beads)
    zs = rng.uniform(np.min(z), np.max(z), num_beads)
    return bead_img_3D(grid, z, xs, ys, zs, np.full(num_beads, bead_size))


"""
Convolves an image with a centered PSF.
Params:
    image (np.ndarray): sample to convolve
    psf (np.ndarray): point-spread function
    norm (bool): whether to normalize the PSF sum
Returns:
    image: convolved array with the input image shape
"""
def psf_convolve(image: np.ndarray, psf: np.ndarray, norm: bool = True):
    # Even kernels are centered between samples. Interpolate onto integer
    # displacements, using zero outside the kernel and preserving its sum.
    for axis, size in enumerate(psf.shape):
        if size % 2 == 0:
            before = [(0, 0)] * psf.ndim
            after = [(0, 0)] * psf.ndim
            before[axis] = (1, 0)
            after[axis] = (0, 1)
            psf = 0.5 * (np.pad(psf, before) + np.pad(psf, after))
    if norm:
        psf_sum = np.sum(psf)
        if not np.isfinite(psf_sum) or psf_sum <= 0:
            raise ValueError("Cannot normalize a PSF with a nonpositive or nonfinite sum.")

    else:
        psf_sum = 1
    
    return fftconvolve(image, psf / psf_sum, mode="same")

"""
Adds Gaussian noise at the specified signal-to-noise ratio.
Params:
    image (np.ndarray): sample to convolve
    SNR (float): signal-to-noise ratio
    rng (np.random.Generator): random number generator
    percentile (float): signal percentile used to set noise strength
Returns:
    image: sample array with additive noise
"""
def add_gaussian_noise(image: np.ndarray, SNR: float, rng: np.random.Generator, percentile: float = 90):
    sigma_noise = np.percentile(image, percentile) / SNR
    return image + rng.normal(0, sigma_noise, image.shape)
