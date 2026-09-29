"""Regularized 3D reconstruction and analytic aberration optimization."""

from __future__ import annotations

from typing import Literal, Sequence
import numpy as np
from utils.psf import (Arbitrary_Grid, Image_Mask, Image_Mask_3D, Microscope,
                       _convolution_grid, _grid_at_z, _z_array)
from utils.zernike import Aberration, EmptyAberration


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
    otfs, observed, estimated, shape, crop, sample_z, data, (px, py) = _solve_reconstruction(
        microscope, grid, images, focal_z_levels, sample_z_levels,
        aberration, rho, diversities, mode, frequency_batch_size, padding)
    result = np.fft.irfft2(estimated.reshape(len(sample_z), shape[0], shape[1]//2+1), s=shape)
    sample = Image_Mask_3D([Image_Mask(_grid_at_z(grid, level), plane)
                           for level, plane in zip(sample_z, result[crop])])
    if return_info:
        predicted = np.einsum('kpf,pf->kf', otfs, estimated)
        residual = np.fft.irfft2((predicted-observed).reshape(observed.shape[0], shape[0], shape[1]//2+1), s=shape)
        norm_data = np.linalg.norm(data)
        measured_norm = np.linalg.norm(np.asarray(images))
        info = dict(padding=(px, py), padded_shape=shape,
                    loss=_fourier_loss(observed - predicted, estimated, rho, shape),
                    relative_residual=float(np.linalg.norm(residual)/norm_data) if norm_data else 0.0,
                    measured_relative_residual=float(np.linalg.norm(residual[crop])/measured_norm) if measured_norm else 0.0)
        # Residuals describe the padded estimate before cropping, including its
        # inferred exterior sample values; they are not a cropped forward refit.
        return sample, info
    return sample


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

    Fits the sample with the same solver as ``estimate_sample_3D`` and returns
    sum(|D - S F_hat|**2) + rho * sum(|F_hat|**2), as in math.tex.
    Sums cover the full lateral Fourier grid using NumPy's unnormalized
    forward FFT convention. By Parseval's identity this is the spatial
    squared residual plus sample penalty, multiplied by the padded pixel
    count. Both terms use the padded estimate before cropping.

    Arguments and validation match ``estimate_sample_3D``. Keep padding
    fixed when comparing losses across aberrations.
    """
    otfs, observed, estimated, shape, *_ = _solve_reconstruction(
        microscope, grid, images, focal_z_levels, sample_z_levels,
        aberration, rho, diversities, mode, frequency_batch_size, padding)
    residual = observed - np.einsum('kpf,pf->kf', otfs, estimated)
    return _fourier_loss(residual, estimated, rho, shape)


def _frequency_weights(shape):
    """Multiplicity of each stored real-FFT bin in the full Fourier sum."""
    weights = np.full((shape[0], shape[1] // 2 + 1), 2.0)
    weights[:, 0] = 1
    if shape[1] % 2 == 0:
        weights[:, -1] = 1
    return weights.ravel()


def _fourier_loss(residual, estimated, rho, shape):
    return float(np.sum(_frequency_weights(shape) * (
        np.sum(abs(residual)**2, axis=0)
        + rho * np.sum(abs(estimated)**2, axis=0))))


class _KernelTransform:
    """Reuse wrapping indices and a periodic buffer for a stack of PSFs."""

    def __init__(self, shape):
        self.shape = shape
        self.periodic = np.zeros(shape)
        self.kernel_shape = None

    def __call__(self, kernel):
        if kernel.shape != self.kernel_shape:
            self.kernel_shape = kernel.shape
            ix, iy = [(np.arange(n) - n // 2) % size
                      for n, size in zip(kernel.shape, self.shape)]
            self.indices = (ix[:, None], iy[None, :])
            self.unique = all(n <= size for n, size in zip(kernel.shape, self.shape))
        self.periodic.fill(0)
        if self.unique:
            self.periodic[self.indices] = kernel
        else:
            # On unpadded even axes the odd kernel has N+1 samples;
            # boundary samples represent the same periodic displacement.
            np.add.at(self.periodic, self.indices, kernel)
        return np.fft.rfft2(self.periodic).ravel()


def _solve_reconstruction(microscope, grid, images, focal_z_levels,
                          sample_z_levels, aberration, rho, diversities,
                          mode, frequency_batch_size, padding):
    focal_z, sample_z = _z_array(focal_z_levels), _z_array(sample_z_levels)
    data = np.asarray(images)
    shape = (grid.grid_ffp_x, grid.grid_ffp_y)
    if data.shape != (len(focal_z), *shape) or not np.isrealobj(data) or not np.all(np.isfinite(data)):
        raise ValueError('images must be a finite real array with shape (K, grid_ffp_x, grid_ffp_y).')
    if len(np.unique(sample_z)) != len(sample_z):
        raise ValueError('Reconstruction planes must be distinct.')
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError('rho must be finite and strictly positive.')
    if not isinstance(frequency_batch_size, int) or frequency_batch_size < 1:
        raise ValueError('frequency_batch_size must be a positive integer.')
    if diversities is None:
        diversities = [EmptyAberration()] * len(focal_z)
    if len(diversities) != len(focal_z) or not all(isinstance(a, Aberration) for a in diversities):
        raise ValueError('Provide one diversity Aberration per acquired image.')
    kernel_grid = _convolution_grid(grid)
    original_shape = shape
    if padding is None:
        padding = (kernel_grid.grid_ffp_x // 2, kernel_grid.grid_ffp_y // 2)
    elif isinstance(padding, (int, np.integer)):
        padding = (padding, padding)
    if (not isinstance(padding, (tuple, list)) or len(padding) != 2 or
            any(not isinstance(p, (int, np.integer)) or p < 0 for p in padding)):
        raise ValueError('padding must be a nonnegative integer or a pair of them.')
    px, py = map(int, padding)
    data = np.pad(data, ((0, 0), (px, px), (py, py)))
    shape = data.shape[1:]
    crop = (slice(None), slice(px, px + original_shape[0]), slice(py, py + original_shape[1]))
    # Store OTFs as (K, P, frequencies); frequency batches limit SVD workspace.
    otfs = np.empty((len(focal_z), len(sample_z), shape[0]*(shape[1]//2+1)), dtype=np.complex128)
    transform = _KernelTransform(shape)
    for k, (focus, diversity) in enumerate(zip(focal_z, diversities)):
        combined = aberration + diversity
        for i, level in enumerate(sample_z):
            _, _, kernel = microscope.compute_PSF(
                _grid_at_z(kernel_grid, level - focus), combined, mode)
            otfs[k, i] = transform(kernel)
    observed = np.fft.rfft2(np.asarray(data)).reshape(len(focal_z), -1)
    estimated = np.empty((len(sample_z), observed.shape[1]), dtype=np.complex128)
    for start in range(0, observed.shape[1], frequency_batch_size):
        stop = start + frequency_batch_size
        S = otfs[:, :, start:stop].transpose(2, 0, 1)
        D = observed[:, start:stop].T
        U, singular, Vh = np.linalg.svd(S, full_matrices=False)
        projected = (U.conj().transpose(0, 2, 1) @ D[..., None])[..., 0]
        weighted = singular / (singular**2 + rho) * projected
        F = (Vh.conj().transpose(0, 2, 1) @ weighted[..., None])[..., 0]
        estimated[:, start:stop] = F.T
    return otfs, observed, estimated, shape, crop, sample_z, data, (px, py)


def _derivative_geometry(microscope, grid, modes, mode):
    """Prepare phase-independent pupil geometry once per derivative evaluation."""
    from utils.rw import (get_bfp_grid, bfp_coord_convert,
                          gaussian_amplitude_s_perp, strength_angular)

    m = microscope
    spacing, x, y = get_bfp_grid(m.L_bfp, m.grid_bfp)
    mask, theta, phi, sx, sy, sz = bfp_coord_convert(m.f, m.n, m.alpha, x, y)
    pupil = np.zeros_like(x, dtype=complex)
    gauss = gaussian_amplitude_s_perp(m.mag, m.w_0, m.f, m.n, np.sqrt(sx**2 + sy**2))
    pupil[mask] = gauss[mask]
    if mode == 'scalar':
        pupils = pupil[None]
        scale = (sx[1, 0]-sx[0, 0]) * (sy[0, 1]-sy[0, 0])
    elif mode == 'vector':
        angular = np.asarray(strength_angular(theta, phi))
        pupils = np.zeros((3, *pupil.shape), dtype=complex)
        pupils[:, mask] = pupil[mask] * angular[:, mask] / np.sqrt(sz[mask])
        scale = -1j*m.k*m.f/(2*np.pi) * (spacing/(m.f*m.n))**2
    else:
        raise ValueError("mode must be 'scalar' or 'vector'.")
    x, y = grid.get_xy()
    Ax = np.exp(1j*m.k*np.outer(x, sx[:, 0]))
    Ay = np.exp(1j*m.k*np.outer(sy[0, :], y))

    def propagate(p):
        return scale * (Ax @ p @ Ay)

    # A unit strength constructs 2*pi*Z, including the waves-to-radians factor.
    basis = []
    for zmode in modes:
        z = np.zeros_like(theta)
        z[mask] = Aberration([zmode], [1]).construct_map(m.alpha)(theta[mask], phi[mask])
        basis.append(z)
    return pupils, mask, theta, phi, sz, basis, propagate


def _psf_derivatives(microscope, grid, aberration, modes, mode, geometry=None):
    """Yield first and upper-triangular second PSF derivatives in waves.

    Differentiate pupil propagation analytically, summing vector components
    before applying the multiphoton order. Geometry is local to this call's
    optical setup, so changes to mutable microscopes cannot leave stale caches.
    """
    if geometry is None:
        geometry = _derivative_geometry(microscope, grid, modes, mode)
    amplitudes, mask, theta, phi, sz, basis, propagate = geometry
    phase = np.zeros_like(theta, dtype=complex)
    phase[mask] = np.exp(1j * (
        aberration.construct_map(microscope.alpha)(theta[mask], phi[mask])
        + microscope.k * grid.z_level * sz[mask]))
    pupils = amplitudes * phase
    field = propagate(pupils)
    first = [propagate(1j*z*pupils) for z in basis]
    intensity = np.sum(np.abs(field)**2, axis=0)
    d_intensity = [2*np.real(np.sum(field.conj()*d, axis=0)) for d in first]
    N = microscope.N_order
    first_factor = N * intensity**(N-1)
    second_factor = N*(N-1)*intensity**(N-2) if N != 1 else None
    for j, d in enumerate(d_intensity):
        yield j, None, first_factor * d
    for j in range(len(modes)):
        for l in range(j, len(modes)):
            second = propagate(-basis[j]*basis[l]*pupils)
            dd_intensity = 2*np.real(np.sum(first[j].conj()*first[l] + field.conj()*second, axis=0))
            dd = first_factor * dd_intensity
            if N != 1:
                dd += second_factor*d_intensity[j]*d_intensity[l]
            yield j, l, dd


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
    definite. Stores M first-derivative OTF stacks; second derivatives are
    contracted one PSF at a time to avoid storing M squared OTF stacks.
    """
    S, D, F, shape, _, sample_z, _, _ = _solve_reconstruction(
        microscope, grid, images, focal_z_levels, sample_z_levels,
        aberration, rho, diversities, mode, frequency_batch_size, padding)
    r = D - np.einsum('kpf,pf->kf', S, F)
    weights = _frequency_weights(shape)
    loss = _fourier_loss(r, F, rho, shape)
    M = len(aberration)
    dS = np.empty((M, *S.shape), dtype=complex)
    gradient = np.zeros(M)
    hessian = np.zeros((M, M))
    kernel_grid = _convolution_grid(grid)
    if diversities is None:
        diversities = [EmptyAberration()] * len(D)
    transform = _KernelTransform(shape)
    geometry = _derivative_geometry(microscope, kernel_grid, aberration.modes, mode)
    for k, (focus, diversity) in enumerate(zip(_z_array(focal_z_levels), diversities)):
        combined = aberration + diversity
        for p, level in enumerate(sample_z):
            for j, l, kernel in _psf_derivatives(
                    microscope, _grid_at_z(kernel_grid, level-focus),
                    combined, aberration.modes, mode, geometry):
                otf = transform(kernel)
                if l is None:
                    dS[j, k, p] = otf
                else:
                    hessian[j, l] -= 2*np.real(np.sum(weights*r[k].conj()*F[p]*otf))
    hessian += np.triu(hessian, 1).T
    for start in range(0, D.shape[1], frequency_batch_size):
        sl = slice(start, start+frequency_batch_size)
        s = S[:, :, sl].transpose(2, 0, 1)
        ds = dS[:, :, :, sl].transpose(3, 0, 1, 2)
        f, residual = F[:, sl].T, r[:, sl].T
        q = np.einsum('bmkp,bp->bmk', ds, f)
        c = (np.einsum('bmkp,bk->bpm', ds.conj(), residual)
             - np.einsum('bkp,bmk->bpm', s.conj(), q))
        # Full right singular basis includes null-space directions when P>K.
        _, singular, vh = np.linalg.svd(s, full_matrices=True)
        v = vh.conj().swapaxes(-1, -2)
        projected = vh @ c
        denom = np.full((len(s), s.shape[2]), rho, dtype=float)
        denom[:, :singular.shape[1]] += singular**2
        response = v @ (projected / denom[..., None])
        w = weights[sl]
        gradient -= 2*np.real(np.einsum('b,bk,bmk->m', w, residual.conj(), q))
        hessian += 2*np.real(
            np.einsum('b,bmk,bnk->mn', w, q.conj(), q)
            - np.einsum('b,bpm,bpn->mn', w, c.conj(), response))
    return loss, gradient, (hessian+hessian.T)/2


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
                           verbose: bool = True) -> tuple[Aberration, dict]:
    """Fit common Zernike strengths with damped Newton steps and backtracking.

    ``modes`` contains [m, n] pairs. Starting strengths default to zero waves;
    known diversities stay fixed while the sample is refitted at every trial.
    Prints the initial point and every accepted step unless verbose=False.
    max_step bounds the largest coefficient change per step, in waves.

    Loss, gradient and Hessian are internally scaled by the initial loss to
    accommodate raw optical intensities. Convergence requires an infinity-norm
    scaled gradient <= gradient_tolerance and no significant negative Hessian
    eigenvalue. Indefinite Hessians are regularized for descent; stationary
    points with negative curvature trigger a curvature step instead.
    Small steps, a failed line search, or the iteration limit stop with
    success=False. This is a local optimizer, not a global-minimum guarantee.

    Returns (estimated_aberration, info). info contains success, message,
    iterations (accepted steps), loss, gradient, hessian and history. History
    stores unscaled losses and copies of strengths in mode order. Estimated
    strengths describe the unknown aberration; their negatives give a
    compensating phase in the same basis.
    """
    z_modes = np.asarray(modes)
    if (z_modes.ndim != 2 or z_modes.shape[1] != 2 or len(z_modes) == 0
            or not np.all(np.isfinite(z_modes)) or np.any(z_modes != np.floor(z_modes))):
        raise ValueError('modes must be a nonempty array of integer [m, n] pairs.')
    z_modes = z_modes.astype(int)
    azimuth, radial = z_modes.T
    if np.any(radial < 0) or np.any(abs(azimuth) > radial) or np.any((radial-abs(azimuth)) % 2):
        raise ValueError('Invalid Zernike [m, n] pair.')
    strengths = np.zeros(len(z_modes)) if initial_strengths is None else np.array(initial_strengths, dtype=float, copy=True)
    if strengths.shape != (len(z_modes),) or not np.all(np.isfinite(strengths)):
        raise ValueError('Provide one finite initial strength per mode.')
    for name, value in [('max_iterations', max_iterations), ('max_backtracks', max_backtracks)]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    for name, value in [('gradient_tolerance', gradient_tolerance),
                        ('strength_tolerance', strength_tolerance), ('max_step', max_step)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and strictly positive.')
    common = dict(microscope=microscope, grid=grid, images=images,
                  focal_z_levels=focal_z_levels, sample_z_levels=sample_z_levels,
                  rho=rho, diversities=diversities, mode=mode,
                  frequency_batch_size=frequency_batch_size, padding=padding)
    current = Aberration(z_modes, strengths)
    history = []
    success = False
    message = 'Maximum iterations reached.'
    scale = None
    for iteration in range(max_iterations + 1):
        loss, gradient, hessian = evaluate_loss_derivatives_3D(aberration=current, **common)
        if not (np.isfinite(loss) and np.all(np.isfinite(gradient)) and np.all(np.isfinite(hessian))):
            raise FloatingPointError('Nonfinite loss or derivatives during aberration optimization.')
        if scale is None:
            scale = max(abs(loss), np.finfo(float).tiny)
        history.append(dict(iteration=iteration, loss=loss, strengths=current.strengths.copy()))
        if verbose:
            print(f'Iteration {iteration}: loss={loss:.10e}, strengths (waves)='
                  f'{np.array2string(current.strengths, precision=8)}', flush=True)
        g = gradient / scale
        h = (hessian + hessian.T) / (2*scale)
        eigenvalues, eigenvectors = np.linalg.eigh(h)
        curvature_tol = 1e-10 * max(1.0, float(np.max(abs(eigenvalues))))
        stationary = np.linalg.norm(g, ord=np.inf) <= gradient_tolerance
        if stationary and eigenvalues[0] >= -curvature_tol:
            success = True
            message = 'Gradient and curvature tolerances satisfied.'
            break
        if iteration == max_iterations:
            break
        if stationary:
            # Try both signs to escape a stationary maximum or saddle.
            direction = eigenvectors[:, 0]
            direction *= max_step / np.max(abs(direction))
            directions = [direction, -direction]
        else:
            floor = max(1e-8, 1e-6*float(np.max(abs(eigenvalues))))
            direction = -eigenvectors @ ((eigenvectors.T @ g) / np.maximum(eigenvalues, floor))
            direction *= min(1.0, max_step / np.max(abs(direction)))
            directions = [direction]
        accepted = False
        for backtrack in range(max_backtracks):
            for direction in directions:
                step = direction * 0.5**backtrack
                trial = Aberration(z_modes, current.strengths + step)
                trial_loss = evaluate_loss_3D(aberration=trial, **common)
                # Armijo decrease, with strict decrease also for curvature steps.
                if (np.isfinite(trial_loss) and trial_loss < loss
                        and (trial_loss-loss)/scale <= 1e-4*min(float(g @ step), 0.0)):
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            message = 'Line search failed to find a decreasing step.'
            break
        if np.max(abs(step)) <= strength_tolerance:
            message = 'Step below strength tolerance before convergence.'
            break
        current = trial
    if verbose:
        print(f'Stopped: {message}', flush=True)
    return current, dict(success=success, message=message, iterations=len(history)-1,
                         loss=loss, gradient=gradient, hessian=hessian, history=history)
