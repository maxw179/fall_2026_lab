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
    weak-regularization fallback. ``_xp`` is for portable numerical tests only.
    """
    def __init__(self, *args, _xp=None, svd_method='direct', **kwargs):
        if svd_method not in ('direct', 'qr'):
            raise ValueError("svd_method must be 'direct' or 'qr'.")
        self.svd_method = svd_method
        self.svd_counts = dict(qr=0, direct=0)
        super().__init__(*args, _xp=_cupy() if _xp is None else _xp, **kwargs)
        self.backend = 'cuda'

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
