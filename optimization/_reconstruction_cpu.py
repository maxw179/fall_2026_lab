"""Shared reconstruction workspace with an exact reduced-objective Hessian.

The CPU uses NumPy; the CUDA subclass selects CuPy for large runtime arrays.
It snapshots the optical setup and data: construct a new workspace after changing
images, planes, diversities, microscope parameters, modes, padding, or rho.
No PSF normalization, coordinate rounding, or precision reduction is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from optimization._reconstruction_common import _derivative_geometry, _frequency_weights
from utils.psf import Image_Mask, Image_Mask_3D, _convolution_grid, _grid_at_z, _z_array
from utils.zernike import Aberration, EmptyAberration


@dataclass
class _Fit:
    strengths: np.ndarray
    otfs: np.ndarray
    estimated: np.ndarray
    residual: np.ndarray
    loss: float
    factors: list
    derivatives: tuple | None = None
    gradient: np.ndarray | None = None


class CPUReconstructionWorkspace:
    """Reuse geometry, batched propagation and object fits across Newton steps.

    The three improvement switches support controlled benchmark ablations.
    ``reuse_svd`` retains right singular bases for the exact object-response
    solve, including null-space directions when sample planes outnumber images.
    ``spatial_hessian`` replaces only the second-PSF-derivative FFT contractions.
    ``deduplicate_psfs`` shares exact (defocus, diversity phase) combinations.

    Memory tradeoff: retained SVD factors occupy approximately
    frequencies * sample_planes * min(images, sample_planes) complex numbers,
    or frequencies * sample_planes**2 when a full right basis is needed.
    ``retain_fields`` also retains the unique PSFs' base/first fields during one
    derivative call. Disable it to recompute those fields using smaller memory.
    Frequency and propagation batches limit temporary arrays, not these caches.
    The workspace is mutable and should not be shared between concurrent calls.
    """

    xp = np
    backend = "cpu"

    def __init__(self, microscope, grid, images, focal_z_levels,
                 sample_z_levels, modes, rho, diversities=None, mode="vector",
                 frequency_batch_size=4096, padding=None, psf_batch_size=8,
                 derivative_batch_size=8, reuse_svd=True, spatial_hessian=True,
                 deduplicate_psfs=True, retain_fields=True, cache_last_fit=True, *, _xp=np):
        self.xp = xp = _xp
        self.backend = "cpu" if xp is np else "cuda"
        for name, value in [("frequency_batch_size", frequency_batch_size),
                            ("psf_batch_size", psf_batch_size),
                            ("derivative_batch_size", derivative_batch_size)]:
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        z_modes = np.asarray(modes)
        if (z_modes.ndim != 2 or z_modes.shape[1] != 2 or len(z_modes) == 0
                or not np.all(np.isfinite(z_modes))
                or np.any(z_modes != np.floor(z_modes))):
            raise ValueError("modes must be a nonempty array of integer [m, n] pairs.")
        self.modes = z_modes.astype(int, copy=True)
        azimuth, radial = self.modes.T
        if (np.any(radial < 0) or np.any(abs(azimuth) > radial)
                or np.any((radial - abs(azimuth)) % 2)):
            raise ValueError("Invalid Zernike [m, n] pair.")
        Aberration(self.modes, np.zeros(len(self.modes)))  # Check uniqueness.
        self.focal_z = _z_array(focal_z_levels).copy()
        self.sample_z = _z_array(sample_z_levels).copy()
        if len(np.unique(self.sample_z)) != len(self.sample_z):
            raise ValueError("Reconstruction planes must be distinct.")
        if not np.isfinite(rho) or rho <= 0:
            raise ValueError("rho must be finite and strictly positive.")
        original_shape = (grid.grid_ffp_x, grid.grid_ffp_y)
        data = xp.asarray(images, dtype=None)
        if (data.shape != (len(self.focal_z), *original_shape)
                or data.dtype.kind not in "buif" or not bool(xp.all(xp.isfinite(data)))):
            raise ValueError("images must be a finite real array with shape (K, x, y).")
        if diversities is None:
            diversities = [EmptyAberration()] * len(self.focal_z)
        if (len(diversities) != len(self.focal_z)
                or not all(isinstance(a, Aberration) for a in diversities)):
            raise ValueError("Provide one diversity Aberration per acquired image.")
        kernel_grid = _convolution_grid(grid)
        self.kernel_shape = (kernel_grid.grid_ffp_x, kernel_grid.grid_ffp_y)
        if padding is None:
            padding = tuple(n // 2 for n in self.kernel_shape)
        elif isinstance(padding, (int, np.integer)):
            padding = (padding, padding)
        if (not isinstance(padding, (tuple, list)) or len(padding) != 2
                or any(not isinstance(p, (int, np.integer)) or p < 0 for p in padding)):
            raise ValueError("padding must be a nonnegative integer or a pair of them.")
        self.padding = tuple(map(int, padding))
        px, py = self.padding
        data = xp.pad(data.astype(xp.float64), ((0, 0), (px, px), (py, py)))
        self.shape = data.shape[1:]
        self.grid = _grid_at_z(grid, grid.z_level)
        self.data_norm = float(xp.linalg.norm(data))
        self.crop = (slice(None), slice(px, px + original_shape[0]),
                     slice(py, py + original_shape[1]))
        self.observed = xp.fft.rfft2(data).reshape(len(self.focal_z), -1)
        self.weights = xp.asarray(_frequency_weights(self.shape))
        self.rho = float(rho)
        self.order = microscope.N_order
        self.frequency_batch_size = int(frequency_batch_size)
        self.psf_batch_size = int(psf_batch_size)
        self.derivative_batch_size = int(derivative_batch_size)
        self.reuse_svd = bool(reuse_svd)
        self.spatial_hessian = bool(spatial_hessian)
        self.deduplicate_psfs = bool(deduplicate_psfs)
        self.retain_fields = bool(retain_fields)
        self.cache_last_fit = bool(cache_last_fit)
        self._last_fit = None

        geometry = _derivative_geometry(microscope, kernel_grid, self.modes, mode, xp=xp)
        self.amplitudes, mask, theta, phi, sz, basis, self._propagate = geometry
        self.amplitudes = xp.asarray(self.amplitudes)
        self.basis = xp.asarray(np.asarray(basis))
        self._indices = tuple((np.arange(n) - n // 2) % size
                              for n, size in zip(self.kernel_shape, self.shape))
        self._unique_indices = all(n <= size for n, size
                                   in zip(self.kernel_shape, self.shape))
        axes = []
        for ids in self._indices:
            cuts = np.r_[0, np.flatnonzero(np.diff(ids) != 1) + 1, len(ids)]
            axes.append([(slice(int(a), int(b)),
                          slice(int(ids[a]), int(ids[b - 1]) + 1))
                         for a, b in zip(cuts[:-1], cuts[1:])])
        self._wrap_blocks = [(sx, sy, dx, dy) for sx, dx in axes[0]
                             for sy, dy in axes[1]]
        self._indices = tuple(xp.asarray(ids) for ids in self._indices)

        # Compare actual fixed phase arrays, including diversity modes outside
        # the optimized basis. Exact keys deliberately do not merge nearby z.
        diversity_ids, diversity_phases, phase_lookup = [], [], {}
        for diversity in diversities:
            phase = np.zeros_like(theta, dtype=complex)
            phase[mask] = np.exp(1j * diversity.construct_map(microscope.alpha)(
                theta[mask], phi[mask]))
            key = phase.tobytes()
            if key not in phase_lookup:
                phase_lookup[key] = len(diversity_phases)
                diversity_phases.append(phase)
            diversity_ids.append(phase_lookup[key])
        group_lookup, fixed_phases, self.groups = {}, [], []
        defocus_phases = {}
        for k, focus in enumerate(self.focal_z):
            for p, level in enumerate(self.sample_z):
                distance = float(level - focus)
                key = (distance, diversity_ids[k])
                if not self.deduplicate_psfs or key not in group_lookup:
                    if distance not in defocus_phases:
                        phase = np.zeros_like(theta, dtype=complex)
                        phase[mask] = np.exp(1j * microscope.k * distance * sz[mask])
                        defocus_phases[distance] = phase
                    group_index = len(self.groups)
                    group_lookup[key] = group_index
                    self.groups.append([])
                    fixed_phases.append(defocus_phases[distance]
                                        * diversity_phases[diversity_ids[k]])
                else:
                    group_index = group_lookup[key]
                self.groups[group_index].append((k, p))
        self.fixed_phases = xp.asarray(np.asarray(fixed_phases))
        self.pairs = list(combinations_with_replacement(range(len(self.modes)), 2))

    def clear_cache(self):
        """Discard the last fit; benchmarks call this to avoid timing cache hits."""
        self._last_fit = None

    def _strengths(self, strengths):
        strengths = np.asarray(strengths, dtype=float)
        if strengths.shape != (len(self.modes),) or not np.all(np.isfinite(strengths)):
            raise ValueError("Provide one finite initial strength per mode.")
        return strengths

    def _host(self, value):
        return np.asarray(value) if self.xp is np else self.xp.asnumpy(value)

    def _svd(self, matrices, full_matrices):
        if self.xp is np:
            return np.linalg.svd(matrices, full_matrices=full_matrices)
        import cupyx
        with cupyx.errstate(linalg="raise"):
            return self.xp.linalg.svd(self.xp.ascontiguousarray(matrices),
                                      full_matrices=full_matrices)

    def _transform(self, kernels):
        """Wrap contiguous blocks, summing even-axis aliases without atomics."""
        leading = kernels.shape[:-2]
        periodic = self.xp.zeros((*leading, *self.shape), dtype=float)
        for sx, sy, dx, dy in self._wrap_blocks:
            periodic[..., dx, dy] += kernels[..., sx, sy]
        return self.xp.fft.rfft2(periodic).reshape(*leading, len(self.weights))

    def _field_batches(self, strengths, first):
        common_phase = self.xp.exp(1j * self.xp.tensordot(self.xp.asarray(strengths), self.basis, axes=1))
        for start in range(0, len(self.groups), self.psf_batch_size):
            ids = range(start, min(start + self.psf_batch_size, len(self.groups)))
            pupils = (self.fixed_phases[start:start + len(ids), None]
                      * common_phase * self.amplitudes[None])
            field = self._propagate(pupils)
            intensity = self.xp.sum(abs(field)**2, axis=1)
            first_fields = d_intensity = None
            if first:
                first_fields = self.xp.empty((len(ids), len(self.modes), *field.shape[1:]),
                                        dtype=complex)
                for j in range(0, len(self.modes), self.derivative_batch_size):
                    z = self.basis[j:j + self.derivative_batch_size]
                    first_fields[:, j:j + len(z)] = self._propagate(
                        1j * z[None, :, None] * pupils[:, None])
                d_intensity = 2*self.xp.real(self.xp.sum(field[:, None].conj() * first_fields, axis=2))
            yield ids, pupils, field, intensity, first_fields, d_intensity


    def _build_otfs(self, strengths, first=False, include_base=True, retain_fields=None):
        shape = (len(self.focal_z), len(self.sample_z), len(self.weights))
        otfs = self.xp.empty(shape, dtype=complex) if include_base else None
        d_otfs = self.xp.empty((len(self.modes), *shape), dtype=complex) if first else None
        retain_fields = self.retain_fields if retain_fields is None else retain_fields
        records = [] if first and retain_fields else None
        for record in self._field_batches(strengths, first):
            ids, _, _, intensity, _, d_intensity = record
            base = self._transform(intensity**self.order) if include_base else None
            if first:
                derivatives = self._transform(
                    self.order * intensity[:, None]**(self.order - 1) * d_intensity)
            for b, group in enumerate(ids):
                for k, p in self.groups[group]:
                    if include_base:
                        otfs[k, p] = base[b]
                    if first:
                        d_otfs[:, k, p] = derivatives[b]
            if records is not None:
                records.append(record)
        return otfs, d_otfs, records


    def _solve(self, otfs):
        estimated = self.xp.empty((len(self.sample_z), len(self.weights)), dtype=complex)
        factors = []
        # A reduced SVD already has a complete right basis when K >= P.
        full = self.reuse_svd and len(self.sample_z) > len(self.focal_z)
        for start in range(0, len(self.weights), self.frequency_batch_size):
            stop = min(start + self.frequency_batch_size, len(self.weights))
            s = otfs[:, :, start:stop].transpose(2, 0, 1)
            d = self.observed[:, start:stop].T
            u, singular, vh = self._svd(s, full_matrices=full)
            rank = singular.shape[1]
            projected = (u[:, :, :rank].conj().swapaxes(-1, -2) @ d[..., None])[..., 0]
            weighted = singular / (singular**2 + self.rho) * projected
            estimated[:, start:stop] = (
                vh[:, :rank].conj().swapaxes(-1, -2) @ weighted[..., None])[..., 0].T
            if self.reuse_svd:
                factors.append((start, stop, singular, vh))
        return estimated, factors


    def _get_fit(self, strengths, first=False, retain_fields=None):
        if (self.cache_last_fit and self._last_fit is not None
                and np.array_equal(self._last_fit.strengths, strengths)):
            return self._last_fit, None, None
        build_options = dict(first=first)
        if retain_fields is not None:
            build_options['retain_fields'] = retain_fields
        otfs, d_otfs, records = self._build_otfs(strengths, **build_options)
        estimated, factors = self._solve(otfs)
        residual = self.observed - self.xp.einsum("kpf,pf->kf", otfs, estimated)
        loss = float(self.xp.sum(self.weights * (self.xp.sum(abs(residual)**2, axis=0)
                     + self.rho * self.xp.sum(abs(estimated)**2, axis=0))))
        fit = _Fit(strengths.copy(), otfs, estimated, residual, loss, factors)
        if self.cache_last_fit:
            self._last_fit = fit
        return fit, d_otfs, records


    def loss(self, strengths):
        return self._get_fit(self._strengths(strengths))[0].loss

    def loss_gradient(self, strengths):
        """Return loss and the analytic envelope gradient without Hessian work.

        Refits the object as usual, but requires no second PSF derivatives or
        object-response solve. First fields are released after each PSF batch.
        Construct with reuse_svd=False to also omit retained full right bases;
        the public gradient optimizer sets this automatically.
        """
        strengths = self._strengths(strengths)
        fit, d_otfs, _ = self._get_fit(strengths, first=True, retain_fields=False)
        if fit.derivatives is not None:
            return fit.loss, fit.derivatives[0].copy()
        if fit.gradient is not None:
            return fit.loss, fit.gradient.copy()
        if d_otfs is None:
            _, d_otfs, _ = self._build_otfs(
                strengths, first=True, include_base=False, retain_fields=False)
        gradient = self.xp.zeros(len(self.modes))
        for start in range(0, len(self.weights), self.frequency_batch_size):
            sl = slice(start, start + self.frequency_batch_size)
            ds = d_otfs[:, :, :, sl].transpose(3, 0, 1, 2)
            f, residual = fit.estimated[:, sl].T, fit.residual[:, sl].T
            q = self.xp.einsum('bmkp,bp->bmk', ds, f)
            gradient -= 2*self.xp.real(self.xp.einsum(
                'b,bk,bmk->m', self.weights[sl], residual.conj(), q))
        if self.cache_last_fit:
            fit.gradient = gradient.copy()
        return fit.loss, gradient


    def _second_psf_hessian(self, strengths, fit, records):
        hessian = self.xp.zeros((len(self.modes), len(self.modes)))
        batches = records if records is not None else self._field_batches(strengths, True)
        ix, iy = self._indices
        for ids, pupils, field, intensity, first_fields, d_intensity in batches:
            # Sum repeated PSFs' contractions before performing an inverse FFT.
            correlation_f = self.xp.zeros((len(ids), len(self.weights)), dtype=complex)
            for b, group in enumerate(ids):
                for k, p in self.groups[group]:
                    correlation_f[b] += fit.residual[k] * fit.estimated[p].conj()
            if self.spatial_hessian:
                correlation = self.xp.fft.irfft2(correlation_f.reshape(
                    len(ids), self.shape[0], self.shape[1]//2 + 1), s=self.shape)
                correlation = correlation[:, ix[:, None], iy[None, :]]
            first_factor = self.order * intensity**(self.order - 1)
            second_factor = (self.order*(self.order - 1)*intensity**(self.order - 2)
                             if self.order != 1 else None)
            for start in range(0, len(self.pairs), self.derivative_batch_size):
                pairs = self.pairs[start:start + self.derivative_batch_size]
                j, l = np.asarray(pairs).T
                second_fields = self._propagate(
                    -(self.basis[j] * self.basis[l])[None, :, None] * pupils[:, None])
                dd_intensity = 2*self.xp.real(self.xp.sum(
                    first_fields[:, j].conj() * first_fields[:, l]
                    + field[:, None].conj() * second_fields, axis=2))
                dd_psf = first_factor[:, None] * dd_intensity
                if second_factor is not None:
                    dd_psf += second_factor[:, None] * d_intensity[:, j] * d_intensity[:, l]
                if self.spatial_hessian:
                    values = -2*np.prod(self.shape) * self.xp.einsum(
                        "bxy,bjxy->j", correlation, dd_psf)
                else:
                    values = -2*self.xp.real(self.xp.einsum("f,bf,bjf->j", self.weights,
                        correlation_f.conj(), self._transform(dd_psf)))
                hessian[j, l] += values
        return hessian + self.xp.triu(hessian, 1).T


    def derivatives(self, strengths):
        """Return the same profiled (loss, gradient, exact Hessian) as the reference."""
        strengths = self._strengths(strengths)
        fit, d_otfs, records = self._get_fit(strengths, first=True)
        if fit.derivatives is not None:
            gradient, hessian = fit.derivatives
            return fit.loss, gradient.copy(), hessian.copy()
        if d_otfs is None:
            _, d_otfs, records = self._build_otfs(strengths, first=True, include_base=False)
        gradient = self.xp.zeros(len(self.modes))
        hessian = self._second_psf_hessian(strengths, fit, records)
        # Fields are not part of the persistent cache; release them before the
        # object-response contractions, which allocate frequency-batch arrays.
        del records
        for batch, start in enumerate(range(0, len(self.weights), self.frequency_batch_size)):
            stop = min(start + self.frequency_batch_size, len(self.weights))
            sl = slice(start, stop)
            s = fit.otfs[:, :, sl].transpose(2, 0, 1)
            ds = d_otfs[:, :, :, sl].transpose(3, 0, 1, 2)
            f, residual = fit.estimated[:, sl].T, fit.residual[:, sl].T
            q = self.xp.einsum("bmkp,bp->bmk", ds, f)
            c = (self.xp.einsum("bmkp,bk->bpm", ds.conj(), residual)
                 - self.xp.einsum("bkp,bmk->bpm", s.conj(), q))
            if self.reuse_svd:
                _, _, singular, vh = fit.factors[batch]
            else:
                _, singular, vh = self._svd(s, full_matrices=True)
            denom = self.xp.full((len(s), len(self.sample_z)), self.rho)
            denom[:, :singular.shape[1]] += singular**2
            response = vh.conj().swapaxes(-1, -2) @ ((vh @ c) / denom[..., None])
            w = self.weights[sl]
            gradient -= 2*self.xp.real(self.xp.einsum("b,bk,bmk->m", w, residual.conj(), q))
            hessian += 2*self.xp.real(
                self.xp.einsum("b,bmk,bnk->mn", w, q.conj(), q)
                - self.xp.einsum("b,bpm,bpn->mn", w, c.conj(), response))
        hessian = (hessian + hessian.T) / 2
        if self.cache_last_fit:
            fit.derivatives = (gradient.copy(), hessian.copy())
        return fit.loss, gradient, hessian


    def estimate_sample(self, strengths, return_info=False):
        """Return the cropped real sample, with optional padded-fit diagnostics."""
        fit = self._get_fit(self._strengths(strengths))[0]
        result = self.xp.fft.irfft2(fit.estimated.reshape(
            len(self.sample_z), self.shape[0], self.shape[1]//2 + 1), s=self.shape)
        sample = Image_Mask_3D([
            Image_Mask(_grid_at_z(self.grid, level), plane)
            for level, plane in zip(self.sample_z, self._host(result[self.crop]))])
        if not return_info:
            return sample
        residual = self.xp.fft.irfft2((-fit.residual).reshape(
            len(self.focal_z), self.shape[0], self.shape[1]//2 + 1), s=self.shape)
        # Padding adds only zeros, so padded and measured data norms are equal.
        info = dict(padding=self.padding, padded_shape=self.shape, loss=fit.loss, backend=self.backend,
                    relative_residual=float(self.xp.linalg.norm(residual)/self.data_norm)
                        if self.data_norm else 0.0,
                    measured_relative_residual=float(self.xp.linalg.norm(residual[self.crop])/self.data_norm)
                        if self.data_norm else 0.0)
        return sample, info


    def memory_summary(self):
        """Array storage estimates; these are not process peak-memory measurements."""
        fit = self._last_fit
        fit_bytes = 0
        if fit is not None:
            fit_bytes = sum(a.nbytes for a in [fit.strengths, fit.otfs,
                            fit.estimated, fit.residual])
            fit_bytes += sum(singular.nbytes + vh.nbytes
                             for _, _, singular, vh in fit.factors)
            if fit.derivatives is not None:
                fit_bytes += sum(a.nbytes for a in fit.derivatives)
            if fit.gradient is not None:
                fit_bytes += fit.gradient.nbytes
        components = len(self.amplitudes)
        pupil_pixels = int(np.prod(self.amplitudes.shape[-2:]))
        kernel_pixels = int(np.prod(self.kernel_shape))
        modes = len(self.modes)
        field_bytes = len(self.groups) * (
            16*components*(pupil_pixels + (1 + modes)*kernel_pixels)
            + 8*(1 + modes)*kernel_pixels)
        return dict(psf_requests=len(self.focal_z)*len(self.sample_z),
                    unique_psfs=len(self.groups), last_fit_bytes=fit_bytes,
                    known_array_bytes=sum(a.nbytes for a in [self.amplitudes,
                        self.basis, self.fixed_phases, self.observed, self.weights]),
                    derivative_otfs_bytes=16*modes*len(self.focal_z)
                        *len(self.sample_z)*len(self.weights),
                    retained_derivative_fields_bytes=field_bytes if self.retain_fields else 0)

    def optimize(self, initial_strengths=None, max_iterations=50,
                 gradient_tolerance=1e-6, strength_tolerance=1e-8,
                 max_step=0.1, max_backtracks=20, verbose=False, method='newton'):
        """Fit Zernike strengths with Newton or gradient steps and backtracking.

        Newton scaling, curvature steps and stopping rules match the reference.
        Gradient mode uses steepest descent, computes no Hessian, and stops on
        the scaled gradient tolerance alone. Both reuse accepted loss fits.
        """
        if method not in ('newton', 'gradient'):
            raise ValueError("method must be 'newton' or 'gradient'.")
        for name, value in [("max_iterations", max_iterations),
                            ("max_backtracks", max_backtracks)]:
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name, value in [("gradient_tolerance", gradient_tolerance),
                            ("strength_tolerance", strength_tolerance),
                            ("max_step", max_step)]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and strictly positive.")
        strengths = self._strengths(np.zeros(len(self.modes)) if initial_strengths is None
                                   else initial_strengths).copy()
        self.clear_cache()
        history, scale, success = [], None, False
        message = "Maximum iterations reached."
        for iteration in range(max_iterations + 1):
            if method == 'newton':
                loss, gradient, hessian = self.derivatives(strengths)
            else:
                loss, gradient = self.loss_gradient(strengths)
                hessian = None
            if not (np.isfinite(loss) and np.all(np.isfinite(gradient))
                    and (hessian is None or np.all(np.isfinite(hessian)))):
                raise FloatingPointError("Nonfinite loss or derivatives during optimization.")
            if scale is None:
                scale = max(abs(loss), np.finfo(float).tiny)
            history.append(dict(iteration=iteration, loss=loss, strengths=strengths.copy()))
            if verbose:
                print(f"Iteration {iteration}: loss={loss:.10e}, strengths (waves)="
                      f"{np.array2string(strengths, precision=8)}", flush=True)
            g = gradient / scale
            stationary = np.linalg.norm(g, ord=np.inf) <= gradient_tolerance
            if method == 'gradient':
                if stationary:
                    success, message = True, "Gradient tolerance satisfied; curvature not evaluated."
                    break
                if iteration == max_iterations:
                    break
                direction = -g
                direction *= min(1.0, max_step / np.max(abs(direction)))
                directions = [direction]
            else:
                h = (hessian + hessian.T) / (2*scale)
                eigenvalues, eigenvectors = np.linalg.eigh(h)
                curvature_tol = 1e-10 * max(1.0, float(np.max(abs(eigenvalues))))
                if stationary and eigenvalues[0] >= -curvature_tol:
                    success, message = True, "Gradient and curvature tolerances satisfied."
                    break
                if iteration == max_iterations:
                    break
                if stationary:
                    direction = eigenvectors[:, 0]
                    direction *= max_step / np.max(abs(direction))
                    directions = [direction, -direction]
                else:
                    floor = max(1e-8, 1e-6*float(np.max(abs(eigenvalues))))
                    direction = -eigenvectors @ ((eigenvectors.T @ g)
                                                 / np.maximum(eigenvalues, floor))
                    direction *= min(1.0, max_step / np.max(abs(direction)))
                    directions = [direction]
            accepted = False
            for backtrack in range(max_backtracks):
                for direction in directions:
                    step = direction * 0.5**backtrack
                    trial = strengths + step
                    trial_loss = self.loss(trial)
                    if (np.isfinite(trial_loss) and trial_loss < loss
                            and (trial_loss-loss)/scale <= 1e-4*min(float(g @ step), 0.0)):
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                message = "Line search failed to find a decreasing step."
                break
            if np.max(abs(step)) <= strength_tolerance:
                message = "Step below strength tolerance before convergence."
                break
            strengths = trial
        if verbose:
            print(f"Stopped: {message}", flush=True)
        return Aberration(self.modes.copy(), strengths), dict(
            success=success, message=message, iterations=len(history)-1,
            loss=loss, gradient=gradient, hessian=hessian, history=history,
            method=method, backend=self.backend)
