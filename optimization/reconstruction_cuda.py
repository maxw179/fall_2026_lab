"""CUDA implementation of all reconstruction operations using a persistent workspace.

CuPy is optional. Explicit CUDA requests fail when CUDA cannot be used. Geometry
is prepared once on the host; large iterative arrays remain on the current GPU.
"""
from __future__ import annotations
import numpy as np
from optimization._reconstruction_cpu import CPUReconstructionWorkspace


def _cupy():
    try:
        import cupy as cp
        if not cp.cuda.is_available():
            raise RuntimeError("No usable NVIDIA CUDA GPU is available.")
        # Verify the selected device can allocate and execute, not just enumerate.
        cp.zeros(1).sum().item()
        return cp
    except Exception as exc:
        raise RuntimeError("CUDA requires a working NVIDIA driver and CuPy installation; "
                           "see optimization/CUDA.md.") from exc


class CUDAReconstruction(CPUReconstructionWorkspace):
    """GPU workspace supporting Newton and gradient optimization in double precision.

    Construct and use on the same current CuPy device. Batches bound temporary
    memory; first derivative OTFs and retained fit factors still scale with the
    full acquisition. ``svd_method='qr'`` opts into batched QR reduction for
    eligible rectangular operators; direct SVD remains the default and the
    weak-regularization fallback. Screening defaults to bounded loss-only
    linear solves; ``screening_method='serial'`` restores the original scoring
    path. ``_xp`` is for portable numerical tests only.
    """
    def __init__(self, *args, _xp=None, svd_method='direct',
                 screening_method='linear', screening_batch_size=4, **kwargs):
        if svd_method not in ('direct', 'qr'):
            raise ValueError("svd_method must be 'direct' or 'qr'.")
        self.svd_method = svd_method
        self.svd_counts = dict(qr=0, direct=0)
        if screening_method not in ('linear', 'svd', 'serial'):
            raise ValueError("screening_method must be 'linear', 'svd', or 'serial'.")
        if (isinstance(screening_batch_size, (bool, np.bool_))
                or not isinstance(screening_batch_size, (int, np.integer))
                or screening_batch_size < 1):
            raise ValueError('screening_batch_size must be a positive integer.')
        self.screening_method = screening_method
        self.screening_batch_size = int(screening_batch_size)
        super().__init__(*args, _xp=_cupy() if _xp is None else _xp, **kwargs)
        self.backend = 'cuda'
        self._screen_group_batches = []
        for start in range(0, len(self.groups), self.psf_batch_size):
            stop = min(start + self.psf_batch_size, len(self.groups))
            pairs = [(k, p, group-start) for group in range(start, stop)
                     for k, p in self.groups[group]]
            k, p, group = np.asarray(pairs).T
            self._screen_group_batches.append((start, stop, self.xp.asarray(k),
                                               self.xp.asarray(p), self.xp.asarray(group)))

    def _svd(self, matrices, full_matrices):
        rows, columns = matrices.shape[-2:]
        if self.svd_method == 'qr' and min(rows, columns) <= 32 < max(rows, columns):
            from optimization._batched_svd import qr_reduced_svd
            # Upper bound on conditioning of S^H S + rho I. Even stable SVD
            # variants can differ in derivatives when regularization is tiny;
            # retain the reference factorization for those frequencies.
            safe = self._host(self.xp.sum(abs(matrices)**2, axis=(1, 2)) / self.rho <= 1e6)
            self.svd_counts['qr'] += int(np.count_nonzero(safe))
            self.svd_counts['direct'] += int(np.count_nonzero(~safe))

            def reduced(a):
                if self.xp is np:
                    return qr_reduced_svd(a, full_matrices, self.xp)
                import cupyx
                with cupyx.errstate(linalg='raise'):
                    return qr_reduced_svd(a, full_matrices, self.xp)

            if np.all(safe):
                return reduced(matrices)
            if not np.any(safe):
                return super()._svd(matrices, full_matrices)
            rank = min(rows, columns)
            batch = len(matrices)
            u = self.xp.empty((batch, rows, rows if full_matrices else rank), dtype=matrices.dtype)
            singular = self.xp.empty((batch, rank), dtype=matrices.real.dtype)
            vh = self.xp.empty((batch, columns if full_matrices else rank, columns), dtype=matrices.dtype)
            safe_ids = self.xp.asarray(np.flatnonzero(safe))
            direct_ids = self.xp.asarray(np.flatnonzero(~safe))
            u[safe_ids], singular[safe_ids], vh[safe_ids] = reduced(matrices[safe_ids])
            u[direct_ids], singular[direct_ids], vh[direct_ids] = super()._svd(
                matrices[direct_ids], full_matrices)
            return u, singular, vh
        self.svd_counts['direct'] += len(matrices)
        return super()._svd(matrices, full_matrices)

    def _screen_batch_size(self, count):
        requested = min(self.screening_batch_size, count)
        if self.xp is np or count == 0:
            return requested
        # Allow reuse of released pool allocations. This is a conservative
        # array estimate, not a peak-memory guarantee; allocation errors propagate.
        free, _ = self.xp.cuda.runtime.memGetInfo()
        available = free + self.xp.get_default_memory_pool().free_bytes()
        k, p, q = len(self.focal_z), len(self.sample_z), len(self.weights)
        rank = min(k, p)
        frequency = 16*self.frequency_batch_size*(4*k*p + 4*rank**2 + k**2 + p**2)
        pupil = int(np.prod(self.amplitudes.shape[-2:]))
        kernel = int(np.prod(self.kernel_shape))
        field = self.psf_batch_size * (16*len(self.amplitudes)*(pupil+kernel)
                    + 8*kernel + 24*int(np.prod(self.shape)))
        per_candidate = 16*k*p*q + frequency + field + 16*pupil
        return min(requested, max(1, int(.5*available/max(per_candidate, 1))))

    def _screen_otfs(self, candidates):
        """Propagate a candidate axis and scatter each PSF batch in one assignment."""
        xp = self.xp
        phase = xp.exp(1j*xp.tensordot(xp.asarray(candidates), self.basis, axes=1))
        otfs = xp.empty((len(candidates), len(self.focal_z), len(self.sample_z),
                         len(self.weights)), dtype=complex)
        for start, stop, k, p, group in self._screen_group_batches:
            pupils = (phase[:, None, None] * self.fixed_phases[None, start:stop, None]
                      * self.amplitudes[None, None])
            field = self._propagate(pupils)
            intensity = xp.sum(abs(field)**2, axis=2)
            transformed = self._transform(intensity**self.order)
            otfs[:, k, p] = transformed[:, group]
        return otfs

    def _screen_frequency_losses(self, matrices, data, counts):
        from optimization._screening import linear_screening_loss, spectral_screening_loss
        xp = self.xp
        if self.screening_method == 'svd':
            counts['svd_fallback_frequencies'] += len(matrices)
            return spectral_screening_loss(matrices, data, self.rho, self._svd, xp)
        bound = xp.sum(abs(matrices)**2, axis=(1, 2))/self.rho
        safe = self._host(bound <= 1e6)
        if xp is not np:
            from cupy.cublas import get_batched_gesv_limit
            if min(matrices.shape[-2:]) > get_batched_gesv_limit():
                safe[:] = False
        counts['linear_frequencies'] += int(np.count_nonzero(safe))
        counts['svd_fallback_frequencies'] += int(np.count_nonzero(~safe))

        def linear(s, d):
            if xp is np:
                return linear_screening_loss(s, d, self.rho, xp)
            import cupyx
            with cupyx.errstate(linalg='raise'):
                return linear_screening_loss(s, d, self.rho, xp)

        if np.all(safe):
            return linear(matrices, data)
        if not np.any(safe):
            return spectral_screening_loss(matrices, data, self.rho, self._svd, xp)
        values = xp.empty(len(matrices))
        good, fallback = xp.asarray(np.flatnonzero(safe)), xp.asarray(np.flatnonzero(~safe))
        values[good] = linear(matrices[good], data[good])
        values[fallback] = spectral_screening_loss(
            matrices[fallback], data[fallback], self.rho, self._svd, xp)
        return values

    def _screen_candidates(self, candidates, on_completed=None):
        if self.screening_method == 'serial':
            return super()._screen_candidates(candidates, on_completed)
        xp = self.xp
        count = len(candidates)
        batch_size = self._screen_batch_size(count)
        counts = dict(linear_frequencies=0, svd_fallback_frequencies=0)
        scores = np.empty(count)
        for start in range(0, count, max(batch_size, 1)):
            stop = min(start + batch_size, count)
            current = candidates[start:stop]
            otfs = self._screen_otfs(current)
            totals = xp.zeros(len(current))
            for frequency in range(0, len(self.weights), self.frequency_batch_size):
                sl = slice(frequency, frequency + self.frequency_batch_size)
                bins = len(self.weights[sl])
                s = otfs[..., sl].transpose(0, 3, 1, 2).reshape(
                    -1, len(self.focal_z), len(self.sample_z))
                d = xp.broadcast_to(self.observed[:, sl].T,
                    (len(current), bins, len(self.focal_z))).reshape(-1, len(self.focal_z))
                losses = self._screen_frequency_losses(s, d, counts).reshape(len(current), bins)
                totals += xp.sum(losses * self.weights[sl], axis=1)
            scores[start:stop] = self._host(totals)
            del otfs, s, d, losses, totals
            if stop < count and on_completed is not None:
                on_completed(stop)

        # Gram solves have a bounded conditioning risk, but near-tied scores
        # should not change seed selection merely through reduction roundoff.
        # Recheck the winner and every close challenger using canonical loss.
        valid = np.isfinite(scores)
        finalists = np.empty(0, dtype=int)
        if np.any(valid):
            best = np.min(scores[valid])
            tolerance = 4e-8*max(abs(best), np.finfo(float).tiny)
            finalists = np.flatnonzero(valid & (scores <= best + tolerance))
            for index in finalists:
                scores[index] = self.loss(candidates[index])
        self._last_screening_info = dict(method=self.screening_method, batch_size=batch_size,
                                        refined_candidates=len(finalists), **counts)
        if count and on_completed is not None:
            on_completed(count)
        return scores

    def derivatives(self, strengths):
        loss, gradient, hessian = super().derivatives(strengths)
        return loss, self._host(gradient), self._host(hessian)

    def loss_gradient(self, strengths):
        loss, gradient = super().loss_gradient(strengths)
        return loss, self._host(gradient)

    def sample(self, strengths, return_info=False, *, return_device=False):
        if not return_device:
            return self.estimate_sample(strengths, return_info=return_info)
        fit = self._get_fit(self._strengths(strengths))[0]
        volume = self.xp.fft.irfft2(fit.estimated.reshape(
            len(self.sample_z), self.shape[0], self.shape[1] // 2 + 1), s=self.shape)
        result = volume[self.crop].copy()
        if return_info:
            # Diagnostics only: do not transfer the reconstructed volume.
            residual = self.xp.fft.irfft2(fit.residual.reshape(
                len(self.focal_z), self.shape[0], self.shape[1] // 2 + 1), s=self.shape)
            info = dict(padding=self.padding, padded_shape=self.shape, loss=fit.loss,
                        backend=self.backend,
                        relative_residual=float(self.xp.linalg.norm(residual))/self.data_norm
                            if self.data_norm else 0.,
                        measured_relative_residual=float(self.xp.linalg.norm(residual[self.crop]))/self.data_norm
                            if self.data_norm else 0.)
            return result, info
        return result


CUDAReconstructionWorkspace = CUDAReconstruction


def estimate_sample_3D(*args, psf_batch_size=8, return_device=False, **kwargs):
    from optimization import reconstruction
    if not return_device:
        return reconstruction.estimate_sample_3D(
            *args, backend='cuda', psf_batch_size=psf_batch_size, **kwargs)
    # Bind the public signature to preserve every positional argument.
    import inspect
    bound = inspect.signature(reconstruction.estimate_sample_3D).bind(*args, **kwargs)
    bound.apply_defaults()
    options = dict(bound.arguments)
    aberration = options.pop('aberration')
    return_info = options.pop('return_info')
    options.pop('backend')
    options['psf_batch_size'] = psf_batch_size
    workspace = CUDAReconstruction(modes=aberration.modes, **options)
    return workspace.sample(aberration.strengths, return_info, return_device=True)


def evaluate_loss_3D(*args, **kwargs):
    from optimization import reconstruction
    return reconstruction.evaluate_loss_3D(*args, backend='cuda', **kwargs)


def evaluate_loss_derivatives_3D(*args, **kwargs):
    from optimization import reconstruction
    return reconstruction.evaluate_loss_derivatives_3D(*args, backend='cuda', **kwargs)


def optimize_aberration_3D(*args, **kwargs):
    from optimization import reconstruction
    return reconstruction.optimize_aberration_3D(*args, backend='cuda', **kwargs)
