"""Regularized 3D reconstruction and exact-Hessian CPU aberration optimization.

The public functions keep their existing arguments and return values. A persistent
NumPy workspace reuses fixed geometry and accepted fits during optimization.
Use CPUReconstructionWorkspace directly to reuse an acquisition across calls.
"""

from __future__ import annotations

from typing import Literal, Sequence
import numpy as np
from utils.psf import Arbitrary_Grid, Microscope
from utils.zernike import Aberration
from optimization._reconstruction_common import (
    _KernelTransform, _derivative_geometry, _fourier_loss, _frequency_weights)
from optimization._reconstruction_cpu import CPUReconstructionWorkspace

ReconstructionWorkspace = CPUReconstructionWorkspace


def _workspace(microscope, grid, images, focal_z_levels, sample_z_levels,
               modes, rho, diversities, mode, frequency_batch_size, padding,
               *, reuse_svd=True):
    return CPUReconstructionWorkspace(
        microscope=microscope, grid=grid, images=images,
        focal_z_levels=focal_z_levels, sample_z_levels=sample_z_levels,
        modes=modes, rho=rho, diversities=diversities, mode=mode,
        frequency_batch_size=frequency_batch_size, padding=padding,
        reuse_svd=reuse_svd)


def estimate_sample_3D(microscope: Microscope, grid: Arbitrary_Grid,
                       images: np.ndarray,
                       focal_z_levels: Sequence[float] | np.ndarray,
                       sample_z_levels: Sequence[float] | np.ndarray,
                       aberration: Aberration, rho: float,
                       diversities: Sequence[Aberration] | None = None,
                       mode: Literal['scalar', 'vector'] = 'vector',
                       frequency_batch_size: int = 4096,
                       padding: int | tuple[int, int] | None = None,
                       return_info: bool = False):
    """
    Estimates a 3D sample using the regularized Gaussian object estimator in math.tex.
    Zero-pads the acquired stack, applies the frequency-wise regularized SVD solve,
    and crops the estimate to the original lateral grid.
    Zero padding assumes zero measured signal outside the acquired field, so we get
    an approximation for cropped data. Acquiring a larger field and cropping after
    reconstruction is preferable when appreciable signal reaches the original edges.
    The unconstrained estimate can be negative.
    No PSF normalization or axial integration weights are applied.
    Params:
        microscope (Microscope): optical parameters used to predict the PSFs
        grid (Arbitrary_Grid): shared lateral sampling of the acquired images
        images (np.ndarray): real, finite acquired data with shape (K, x, y)
        focal_z_levels (Sequence[float] | np.ndarray): acquisition focal positions [mm]
        sample_z_levels (Sequence[float] | np.ndarray): distinct reconstruction planes [mm]
        aberration (Aberration): guessed common aberration
        rho (float): strictly positive regularization in the raw PSF Fourier scale
        diversities (Sequence[Aberration] | None): known additional aberrations for each
            acquired image; None uses zero diversity for all K images
        mode (Literal["scalar", "vector"]): optical model matching the data
        frequency_batch_size (int): maximum spatial frequencies solved together
        padding (int | tuple[int, int] | None): pixels added to each side of x and y;
            None uses half the PSF size on each axis; zero gives the unpadded solver
        return_info (bool): whether to also return padded-domain fit diagnostics
    Returns:
        sample (Image_Mask_3D): real-valued estimate at the requested P planes
        info (dict): returned alongside sample only when return_info is True;
            includes padding, padded shape, relative data residuals, and the
            regularized full Fourier-sum loss before cropping
    """
    workspace = _workspace(microscope, grid, images, focal_z_levels,
                           sample_z_levels, aberration.modes, rho, diversities,
                           mode, frequency_batch_size, padding, reuse_svd=False)
    return workspace.estimate_sample(aberration.strengths, return_info=return_info)


def evaluate_loss_3D(microscope: Microscope, grid: Arbitrary_Grid,
                     images: np.ndarray,
                     focal_z_levels: Sequence[float] | np.ndarray,
                     sample_z_levels: Sequence[float] | np.ndarray,
                     aberration: Aberration, rho: float,
                     diversities: Sequence[Aberration] | None = None,
                     mode: Literal['scalar', 'vector'] = 'vector',
                     frequency_batch_size: int = 4096,
                     padding: int | tuple[int, int] | None = None) -> float:
    """Evaluate the regularized Gaussian loss at a given aberration.

    Uses the batched CPU workspace to fit the sample and returns
    sum(|D - S F_hat|**2) + rho * sum(|F_hat|**2), as in math.tex.
    Sums cover the full lateral Fourier grid using NumPy's unnormalized
    forward FFT convention. By Parseval's identity this is the spatial
    squared residual plus sample penalty, multiplied by the padded pixel
    count. Both terms use the padded estimate before cropping.

    Arguments and validation match ``estimate_sample_3D``. Keep padding
    fixed when comparing losses across aberrations.
    """
    workspace = _workspace(microscope, grid, images, focal_z_levels,
                           sample_z_levels, aberration.modes, rho, diversities,
                           mode, frequency_batch_size, padding, reuse_svd=False)
    return workspace.loss(aberration.strengths)


def evaluate_loss_derivatives_3D(microscope: Microscope, grid: Arbitrary_Grid,
                                 images: np.ndarray,
                                 focal_z_levels: Sequence[float] | np.ndarray,
                                 sample_z_levels: Sequence[float] | np.ndarray,
                                 aberration: Aberration, rho: float,
                                 diversities: Sequence[Aberration] | None = None,
                                 mode: Literal['scalar', 'vector'] = 'vector',
                                 frequency_batch_size: int = 4096,
                                 padding: int | tuple[int, int] | None = None
                                 ) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (loss, gradient, Hessian) of the fitted-sample Gaussian loss.

    Uses the analytic reduced-objective formulas in math.tex, including the
    Hessian correction for re-estimating the sample. Entries follow
    ``aberration.modes`` and differentiate ``aberration.strengths`` in waves.
    Diversities remain fixed. Include modes with zero strength to differentiate
    them. Gradient and Hessian have shapes (M,) and (M, M).

    Padding, regularization and full Fourier-sum scaling match evaluate_loss_3D.
    Supports scalar and vector optics. The exact Hessian need not be positive
    definite. Uses batched propagation, a shared SVD, and spatial contractions
    of second PSF derivatives. Stores M first-derivative OTF stacks and retains
    base/first fields within the call; no M-squared OTF stack is allocated.
    """
    workspace = _workspace(microscope, grid, images, focal_z_levels,
                           sample_z_levels, aberration.modes, rho, diversities,
                           mode, frequency_batch_size, padding)
    return workspace.derivatives(aberration.strengths)


def optimize_aberration_3D(microscope: Microscope, grid: Arbitrary_Grid,
                           images: np.ndarray,
                           focal_z_levels: Sequence[float] | np.ndarray,
                           sample_z_levels: Sequence[float] | np.ndarray,
                           modes: Sequence[Sequence[int]] | np.ndarray,
                           rho: float,
                           diversities: Sequence[Aberration] | None = None,
                           initial_strengths: Sequence[float] | np.ndarray | None = None,
                           mode: Literal['scalar', 'vector'] = 'vector',
                           frequency_batch_size: int = 4096,
                           padding: int | tuple[int, int] | None = None,
                           max_iterations: int = 50,
                           gradient_tolerance: float = 1e-6,
                           strength_tolerance: float = 1e-8,
                           max_step: float = 0.1,
                           max_backtracks: int = 20,
                           verbose: bool = True,
                           method: Literal['newton', 'gradient'] = 'newton'
                           ) -> tuple[Aberration, dict]:
    """Fit common Zernike strengths with Newton or gradient steps and backtracking.

    ``modes`` contains [m, n] pairs. Starting strengths default to zero waves;
    known diversities stay fixed while the sample is refitted at every trial.
    Prints the initial point and every accepted step unless verbose=False.
    max_step bounds the largest coefficient change per step, in waves.
    method='newton' uses the exact Hessian (default). method='gradient' uses
    steepest descent with Armijo backtracking and skips all Hessian work.

    Loss, gradient and Hessian are internally scaled by the initial loss to
    accommodate raw optical intensities. Convergence requires an infinity-norm
    scaled gradient <= gradient_tolerance and no significant negative Hessian
    eigenvalue. Indefinite Hessians are regularized for descent; stationary
    points with negative curvature trigger a curvature step instead.
    Small steps, a failed line search, or the iteration limit stop with
    success=False. This is a local optimizer, not a global-minimum guarantee.
    In gradient mode, convergence checks only the scaled gradient tolerance;
    curvature is not checked, so a stationary saddle or maximum can satisfy it.

    Returns (estimated_aberration, info). info contains success, message,
    iterations (accepted steps), loss, gradient, hessian and history. History
    stores unscaled losses and copies of strengths in mode order. Estimated
    strengths describe the unknown aberration; their negatives give a
    compensating phase in the same basis. One CPU workspace snapshots geometry,
    observation FFTs and diversity/defocus phases for this optimization, and
    reuses the accepted line-search fit at the next iteration. There is no
    global cache across different acquisitions. info['method'] records the
    selected method; info['hessian'] is None in gradient mode.
    """
    if method not in ('newton', 'gradient'):
        raise ValueError("method must be 'newton' or 'gradient'.")
    workspace = _workspace(microscope, grid, images, focal_z_levels,
                           sample_z_levels, modes, rho, diversities,
                           mode, frequency_batch_size, padding,
                           reuse_svd=(method == 'newton'))
    return workspace.optimize(
        initial_strengths=initial_strengths, max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
        strength_tolerance=strength_tolerance, max_step=max_step,
        max_backtracks=max_backtracks, verbose=verbose, method=method)


def _solve_reconstruction(microscope, grid, images, focal_z_levels,
                          sample_z_levels, aberration, rho, diversities,
                          mode, frequency_batch_size, padding):
    """Preserve the private raw-fit tuple used by existing scan notebooks."""
    workspace = _workspace(microscope, grid, images, focal_z_levels,
                           sample_z_levels, aberration.modes, rho, diversities,
                           mode, frequency_batch_size, padding, reuse_svd=False)
    fit = workspace._get_fit(workspace._strengths(aberration.strengths))[0]
    px, py = workspace.padding
    data = np.pad(np.asarray(images), ((0, 0), (px, px), (py, py)))
    return (fit.otfs, workspace.observed, fit.estimated, workspace.shape,
            workspace.crop, workspace.sample_z, data, workspace.padding)
