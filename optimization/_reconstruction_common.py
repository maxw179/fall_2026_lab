"""Shared Fourier conventions and fixed pupil geometry for CPU reconstruction."""
from __future__ import annotations

import numpy as np
from utils.zernike import Aberration


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


def _derivative_geometry(microscope, grid, modes, mode, *, xp=np, propagation="auto", fft_workers=1):
    """Prepare phase-independent pupil geometry for an acquisition workspace."""
    from utils.rw import (get_bfp_grid, bfp_coord_convert,
                          gaussian_amplitude_s_perp, strength_angular)

    m = microscope
    spacing, x, y = get_bfp_grid(m.L_bfp, m.grid_bfp)
    mask, theta, phi, sx, sy, sz = bfp_coord_convert(m.f, m.n, m.alpha, x, y)
    pupil = np.zeros_like(x, dtype=complex)
    gauss = gaussian_amplitude_s_perp(m.mag, m.w_0, m.f, m.n, np.sqrt(sx**2 + sy**2))
    pupil[mask] = gauss[mask]
    # Optional fixed pupil transmission is shared by forward and inverse optics.
    transmission = getattr(m, 'pupil_transmission', None)
    if transmission is not None:
        weights = np.asarray(transmission(theta, phi, mask), dtype=float)
        if weights.shape != mask.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError('Pupil transmission must be finite, nonnegative, and match the pupil grid.')
        pupil *= weights
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
    if propagation not in ('auto', 'dense', 'zoom'):
        raise ValueError("propagation must be 'auto', 'dense', or 'zoom'.")
    # Whole-solver benchmarks vary with batch sizes and BLAS implementation.
    # Keep the proven dense default; ZoomFFT is explicitly selectable.
    use_zoom = xp is np and propagation == 'zoom'
    if propagation == 'zoom' and xp is not np:
        raise ValueError("zoom propagation is a CPU backend option.")
    if use_zoom:
        from scipy.signal import ZoomFFT
        axes = [(x, sx[:, 0]), (y, sy[0, :])]
        transforms = []
        for target, source in axes:
            # ZoomFFT computes exp(-i*frequency*j). The physical exponent
            # has the opposite sign, with source[j] = source[0] + j*spacing.
            source_step = source[1] - source[0]
            if not (np.allclose(np.diff(source), source_step, rtol=1e-12, atol=1e-15)
                    and (len(target) < 3 or np.allclose(np.diff(target), target[1]-target[0],
                                                       rtol=1e-12, atol=1e-15))):
                raise ValueError("zoom propagation requires uniformly sampled axes.")
            frequencies = -m.k*source_step*np.asarray(target)
            # A half-open range also handles a one-point output axis.
            step = frequencies[1]-frequencies[0] if len(target) > 1 else 0.
            transforms.append(ZoomFFT(len(source),
                [frequencies[0], frequencies[0]+len(target)*step],
                len(target), fs=2*np.pi))
        phase = scale * np.exp(1j*m.k*x*sx[0, 0])[:, None] * np.exp(1j*m.k*y*sy[0, 0])[None, :]

        def propagate(p):
            from scipy.fft import set_workers
            with set_workers(fft_workers):
                return transforms[1](transforms[0](p, axis=-2), axis=-1) * phase
    else:
        Ax = xp.asarray(np.exp(1j*m.k*np.outer(x, sx[:, 0])))
        Ay = xp.asarray(np.exp(1j*m.k*np.outer(sy[0, :], y)))

        def propagate(p):
            return scale * (Ax @ p @ Ay)

    # A unit strength constructs 2*pi*Z, including the waves-to-radians factor.
    basis = []
    for zmode in modes:
        z = np.zeros_like(theta)
        z[mask] = Aberration([zmode], [1]).construct_map(m.alpha)(theta[mask], phi[mask])
        basis.append(z)
    return pupils, mask, theta, phi, sz, basis, propagate
