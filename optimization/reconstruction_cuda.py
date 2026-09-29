"""Single-GPU CUDA reconstruction with a persistent CuPy workspace.

The public functions mirror reconstruction.py. Reuse CUDAReconstruction for
sweeps/optimization to keep observations, geometry and Zernike bases on device.
All large runtime arrays use float64/complex128; only setup geometry and the
small Newton optimizer run on the host. See CUDA.md for cluster usage.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import numpy as np

from utils.psf import Image_Mask, Image_Mask_3D, _convolution_grid, _grid_at_z, _z_array
from utils.rw import get_bfp_grid, bfp_coord_convert, gaussian_amplitude_s_perp, strength_angular
from utils.zernike import Aberration, EmptyAberration
from optimization.reconstruction import _frequency_weights


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError('CUDA reconstruction requires CuPy; see optimization/CUDA.md.') from exc
    if not cp.cuda.is_available():
        raise RuntimeError('No CUDA GPU is available. Run on an allocated NVIDIA GPU node.')
    return cp


def _validate_modes(modes):
    modes = np.asarray(modes)
    if (modes.ndim != 2 or modes.shape[1] != 2 or not len(modes)
            or not np.all(np.isfinite(modes)) or np.any(modes != np.floor(modes))):
        raise ValueError('modes must be a nonempty array of integer [m, n] pairs.')
    modes = modes.astype(int)
    m, n = modes.T
    if np.any(n < 0) or np.any(abs(m) > n) or np.any((n-abs(m)) % 2):
        raise ValueError('Invalid Zernike [m, n] pair.')
    if len(np.unique(modes, axis=0)) != len(modes):
        raise ValueError('Zernike modes must be unique.')
    return modes


class CUDAReconstruction:
    """Snapshot a fixed acquisition and optical model on the current CUDA device.

    ``modes`` fixes coefficient order; methods accept strengths in waves.
    ``psf_batch_size`` bounds simultaneous pupil/field arrays;
    ``frequency_batch_size`` bounds SVD/Hessian workspace. OTF storage still
    scales as K*P*frequencies, and derivatives add M such stacks.
    Instances are mutable workspaces and must not be shared between threads.
    Create/use inside the same ``with cupy.cuda.Device(index):`` context.
    ``_xp`` is an internal NumPy numerical-test hook, never an automatic fallback.
    """

    def __init__(self, microscope, grid, images, focal_z_levels, sample_z_levels,
                 modes, rho, diversities=None, mode='vector',
                 frequency_batch_size=4096, padding=None, *, psf_batch_size=16,
                 _xp=None):
        self.xp = xp = _cupy() if _xp is None else _xp
        self.modes = _validate_modes(modes)
        self.grid = deepcopy(grid)
        self.sample_z = _z_array(sample_z_levels).copy()
        focal_z = _z_array(focal_z_levels)
        if len(np.unique(self.sample_z)) != len(self.sample_z):
            raise ValueError('Reconstruction planes must be distinct.')
        if not np.isfinite(rho) or rho <= 0:
            raise ValueError('rho must be finite and strictly positive.')
        for name, value in [('frequency_batch_size', frequency_batch_size),
                            ('psf_batch_size', psf_batch_size)]:
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if mode not in ('scalar', 'vector'):
            raise ValueError("mode must be 'scalar' or 'vector'.")
        self.rho, self.order = float(rho), microscope.N_order
        self.frequency_batch_size, self.psf_batch_size = frequency_batch_size, psf_batch_size
        self.K, self.P, self.M = len(focal_z), len(self.sample_z), len(self.modes)
        original_shape = (grid.grid_ffp_x, grid.grid_ffp_y)
        data = xp.asarray(images)
        if (data.shape != (self.K, *original_shape) or data.dtype.kind not in 'buif'
                or not bool(xp.all(xp.isfinite(data)))):
            raise ValueError('images must be finite real data with shape (K, x, y).')
        kg = _convolution_grid(grid)
        self.kernel_shape = (kg.grid_ffp_x, kg.grid_ffp_y)
        if padding is None:
            padding = tuple(n // 2 for n in self.kernel_shape)
        elif isinstance(padding, (int, np.integer)):
            padding = (padding, padding)
        if (not isinstance(padding, (tuple, list)) or len(padding) != 2
                or any(not isinstance(p, (int, np.integer)) or p < 0 for p in padding)):
            raise ValueError('padding must be a nonnegative integer or pair.')
        self.padding = px, py = tuple(map(int, padding))
        data = xp.pad(data.astype(xp.float64), ((0, 0), (px, px), (py, py)))
        self.shape = data.shape[1:]
        self.crop = (slice(None), slice(px, px+original_shape[0]), slice(py, py+original_shape[1]))
        self.data_norm = xp.linalg.norm(data)
        self.D = xp.fft.rfft2(data).reshape(self.K, -1)
        self.nf = self.D.shape[1]
        self.weights = xp.asarray(_frequency_weights(self.shape))
        diversities = [EmptyAberration()] * self.K if diversities is None else list(diversities)
        if len(diversities) != self.K or not all(isinstance(a, Aberration) for a in diversities):
            raise ValueError('Provide one diversity Aberration per acquired image.')
        self._prepare_geometry(microscope, kg, focal_z, diversities, mode)
        self._cached = None

    def _host(self, value):
        return np.asarray(value) if self.xp is np else self.xp.asnumpy(value)

    def _prepare_geometry(self, m, kg, focal_z, diversities, mode):
        xp = self.xp
        spacing, x, y = get_bfp_grid(m.L_bfp, m.grid_bfp)
        mask, theta, phi, sx, sy, sz = bfp_coord_convert(m.f, m.n, m.alpha, x, y)
        gauss = gaussian_amplitude_s_perp(m.mag, m.w_0, m.f, m.n, np.sqrt(sx*sx+sy*sy))
        amplitudes = np.zeros((1 if mode == 'scalar' else 3, *x.shape), complex)
        if mode == 'scalar':
            amplitudes[0, mask] = gauss[mask]
            scale = (sx[1, 0]-sx[0, 0])*(sy[0, 1]-sy[0, 0])
        else:
            angular = np.asarray(strength_angular(theta, phi))
            amplitudes[:, mask] = gauss[mask]*angular[:, mask]/np.sqrt(sz[mask])
            scale = -1j*m.k*m.f/(2*np.pi)*(spacing/(m.f*m.n))**2
        gx, gy = kg.get_xy()
        self.Ax = xp.asarray(scale*np.exp(1j*m.k*np.outer(gx, sx[:, 0])))
        self.Ay = xp.asarray(np.exp(1j*m.k*np.outer(sy[0, :], gy)))
        self.amplitudes = xp.asarray(amplitudes)
        self.kz = xp.asarray(m.k*sz)
        self.defocus = xp.asarray((self.sample_z[None, :]-focal_z[:, None]).ravel())
        self.image_indices = xp.asarray(np.repeat(np.arange(self.K), self.P))

        def phase(a):
            result = np.zeros_like(theta)
            result[mask] = a.construct_map(m.alpha)(theta[mask], phi[mask])
            return result

        self.basis = xp.asarray(np.stack([phase(Aberration([z], [1])) for z in self.modes]))
        self.diversity_phase = xp.asarray(np.stack([phase(a) for a in diversities]))
        # Split any wrapped boundary into disjoint rectangular assignments.
        # This avoids floating-point atomic scatter-adds and preserves aliases.
        axes = []
        for n, size in zip(self.kernel_shape, self.shape):
            ids = (np.arange(n)-n//2) % size
            cuts = np.r_[0, np.flatnonzero(np.diff(ids) != 1)+1, n]
            axes.append([(slice(int(a), int(b)), slice(int(ids[a]), int(ids[b-1])+1))
                         for a, b in zip(cuts[:-1], cuts[1:])])
        self.wrap_blocks = [(sx, sy, dx, dy) for sx, dx in axes[0] for sy, dy in axes[1]]

    def _strengths(self, strengths):
        strengths = np.asarray(strengths, dtype=float)
        if strengths.shape != (self.M,) or not np.all(np.isfinite(strengths)):
            raise ValueError('Provide one finite strength per mode.')
        return strengths

    def _propagate(self, pupil):
        return self.Ax @ pupil @ self.Ay

    def _fields(self, strengths):
        xp = self.xp
        common = xp.einsum('m,mxy->xy', xp.asarray(strengths), self.basis)
        for start in range(0, self.K*self.P, self.psf_batch_size):
            sl = slice(start, min(start+self.psf_batch_size, self.K*self.P))
            phase = (common + self.diversity_phase[self.image_indices[sl]]
                     + self.defocus[sl, None, None]*self.kz)
            pupil = xp.exp(1j*phase)[:, None]*self.amplitudes
            field = self._propagate(pupil)
            intensity = xp.sum(abs(field)**2, axis=1)
            yield sl, pupil, field, intensity

    def _transform(self, kernels):
        xp = self.xp
        periodic = xp.zeros((*kernels.shape[:-2], *self.shape), dtype=xp.float64)
        for sx, sy, dx, dy in self.wrap_blocks:
            periodic[..., dx, dy] += kernels[..., sx, sy]
        return xp.fft.rfft2(periodic).reshape(*kernels.shape[:-2], self.nf)

    def _otfs(self, strengths, derivatives):
        xp = self.xp
        S = xp.empty((self.K*self.P, self.nf), dtype=xp.complex128)
        dS = xp.empty((self.M, self.K*self.P, self.nf), dtype=xp.complex128) if derivatives else None
        for sl, pupil, field, intensity in self._fields(strengths):
            S[sl] = self._transform(intensity**self.order)
            if derivatives:
                factor = self.order*intensity**(self.order-1)
                for j in range(self.M):
                    first = self._propagate(1j*self.basis[j]*pupil)
                    di = 2*xp.real(xp.sum(field.conj()*first, axis=1))
                    dS[j, sl] = self._transform(factor*di)
        return S.reshape(self.K, self.P, self.nf), (
            dS.reshape(self.M, self.K, self.P, self.nf) if derivatives else None)

    def _svd(self, s, full):
        # Raise on cuSOLVER failures instead of silently using invalid factors.
        if self.xp is np:
            context = nullcontext()
        else:
            import cupyx
            context = cupyx.errstate(linalg='raise')
        with context:
            return self.xp.linalg.svd(s, full_matrices=full)

    def _fit(self, S, dS=None):
        """One factorization per frequency, shared by fit and Hessian response."""
        xp = self.xp
        F = xp.empty((self.P, self.nf), dtype=xp.complex128)
        gradient, hessian = xp.zeros(self.M), xp.zeros((self.M, self.M))
        for start in range(0, self.nf, self.frequency_batch_size):
            sl = slice(start, min(start+self.frequency_batch_size, self.nf))
            s = xp.ascontiguousarray(S[:, :, sl].transpose(2, 0, 1))
            d = self.D[:, sl].T
            # Full right basis explicitly retains null-space directions for P>K.
            u, singular, vh = self._svd(s, full=dS is not None)
            rank = min(self.K, self.P)
            projected = (u[:, :, :rank].conj().swapaxes(-1, -2) @ d[..., None])[..., 0]
            f = (vh[:, :rank].conj().swapaxes(-1, -2)
                 @ (singular/(singular**2+self.rho)*projected)[..., None])[..., 0]
            F[:, sl] = f.T
            if dS is None:
                continue
            residual = d-(s @ f[..., None])[..., 0]
            ds = dS[:, :, :, sl].transpose(3, 0, 1, 2)
            q = xp.einsum('bmkp,bp->bmk', ds, f)
            c = (xp.einsum('bmkp,bk->bpm', ds.conj(), residual)
                 - xp.einsum('bkp,bmk->bpm', s.conj(), q))
            denom = xp.full((len(s), self.P), self.rho)
            denom[:, :rank] += singular**2
            response = vh.conj().swapaxes(-1, -2) @ ((vh @ c)/denom[..., None])
            w = self.weights[sl]
            gradient -= 2*xp.real(xp.einsum('b,bk,bmk->m', w, residual.conj(), q))
            hessian += 2*xp.real(
                xp.einsum('b,bmk,bnk->mn', w, q.conj(), q)
                - xp.einsum('b,bpm,bpn->mn', w, c.conj(), response))
        residual = self.D-xp.einsum('kpf,pf->kf', S, F)
        loss = xp.sum(self.weights*(xp.sum(abs(residual)**2, axis=0)
                                   + self.rho*xp.sum(abs(F)**2, axis=0)))
        return F, residual, loss, gradient, hessian

    def loss(self, strengths):
        """Return a host scalar, retaining the fitted sample for reconstruction."""
        strengths = self._strengths(strengths)
        if self._cached is not None and np.array_equal(strengths, self._cached[0]):
            return self._cached[3]
        self._cached = None
        S, _ = self._otfs(strengths, False)
        F, r, loss, _, _ = self._fit(S)
        loss = float(loss)
        self._cached = (strengths.copy(), F, r, loss)
        return loss

    def derivatives(self, strengths):
        """Return host (loss, gradient, exact reduced Hessian), in waves."""
        xp = self.xp
        strengths = self._strengths(strengths)
        self._cached = None
        S, dS = self._otfs(strengths, True)
        F, r, loss, gradient, hessian = self._fit(S, dS)
        del S, dS
        # Stream second derivatives; never allocate an M*M OTF stack.
        for sl, pupil, field, intensity in self._fields(strengths):
            first = [self._propagate(1j*self.basis[j]*pupil) for j in range(self.M)]
            di = [2*xp.real(xp.sum(field.conj()*d, axis=1)) for d in first]
            factor = self.order*intensity**(self.order-1)
            ids = xp.arange(sl.start, sl.stop)
            contraction = self.weights*r[ids//self.P].conj()*F[ids % self.P]
            for j in range(self.M):
                for l in range(j, self.M):
                    second = self._propagate(-self.basis[j]*self.basis[l]*pupil)
                    ddi = 2*xp.real(xp.sum(first[j].conj()*first[l]+field.conj()*second, axis=1))
                    kernel = factor*ddi
                    if self.order != 1:
                        kernel += self.order*(self.order-1)*intensity**(self.order-2)*di[j]*di[l]
                    term = -2*xp.real(xp.sum(contraction*self._transform(kernel)))
                    hessian[j, l] += term
                    if j != l:
                        hessian[l, j] += term
        loss = float(loss)
        self._cached = (strengths.copy(), F, r, loss)
        return loss, self._host(gradient), self._host((hessian+hessian.T)/2)

    def sample(self, strengths, return_info=False, *, return_device=False):
        """Return Image_Mask_3D, or a cropped CuPy array with return_device=True."""
        xp = self.xp
        loss = self.loss(strengths)
        _, F, r, _ = self._cached
        fourier_shape = (self.shape[0], self.shape[1]//2+1)
        volume = xp.fft.irfft2(F.reshape(self.P, *fourier_shape), s=self.shape)
        cropped = volume[self.crop].copy()
        if return_device:
            sample = cropped
        else:
            sample = Image_Mask_3D([Image_Mask(_grid_at_z(self.grid, z), plane)
                                   for z, plane in zip(self.sample_z, self._host(cropped))])
        if not return_info:
            return sample
        residual = xp.fft.irfft2(r.reshape(self.K, *fourier_shape), s=self.shape)
        norm = float(self.data_norm)
        info = dict(padding=self.padding, padded_shape=self.shape, loss=loss,
                    relative_residual=float(xp.linalg.norm(residual))/norm if norm else 0.,
                    measured_relative_residual=float(xp.linalg.norm(residual[self.crop]))/norm if norm else 0.)
        return sample, info

    def optimize(self, initial_strengths=None, max_iterations=50,
                 gradient_tolerance=1e-6, strength_tolerance=1e-8,
                 max_step=.1, max_backtracks=20, verbose=True):
        """CPU Newton control with GPU loss/gradient/Hessian evaluations.

        Convergence and line-search rules match reconstruction.py. Only small
        coefficient arrays and scalar objectives cross the device boundary.
        """
        z_modes = self.modes
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
        current = Aberration(z_modes, strengths)
        history = []
        success = False
        message = 'Maximum iterations reached.'
        scale = None
        for iteration in range(max_iterations + 1):
            loss, gradient, hessian = self.derivatives(current.strengths)
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
                    trial_loss = self.loss(trial.strengths)
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


def _workspace(microscope, grid, images, focal_z_levels, sample_z_levels,
               aberration, rho, diversities, mode, frequency_batch_size,
               padding, psf_batch_size):
    return CUDAReconstruction(microscope, grid, images, focal_z_levels,
                              sample_z_levels, aberration.modes, rho,
                              diversities, mode, frequency_batch_size, padding,
                              psf_batch_size=psf_batch_size)


def estimate_sample_3D(microscope, grid, images, focal_z_levels, sample_z_levels,
                       aberration, rho, diversities=None, mode='vector',
                       frequency_batch_size=4096, padding=None, return_info=False,
                       *, psf_batch_size=16, return_device=False):
    """CUDA equivalent of reconstruction.estimate_sample_3D.

    return_device=True returns a CuPy volume instead of Image_Mask_3D.
    """
    workspace = _workspace(microscope, grid, images, focal_z_levels, sample_z_levels,
                           aberration, rho, diversities, mode, frequency_batch_size,
                           padding, psf_batch_size)
    return workspace.sample(aberration.strengths, return_info, return_device=return_device)


def evaluate_loss_3D(microscope, grid, images, focal_z_levels, sample_z_levels,
                     aberration, rho, diversities=None, mode='vector',
                     frequency_batch_size=4096, padding=None, *, psf_batch_size=16):
    """CUDA equivalent of reconstruction.evaluate_loss_3D."""
    workspace = _workspace(microscope, grid, images, focal_z_levels, sample_z_levels,
                           aberration, rho, diversities, mode, frequency_batch_size,
                           padding, psf_batch_size)
    return workspace.loss(aberration.strengths)


def evaluate_loss_derivatives_3D(microscope, grid, images, focal_z_levels, sample_z_levels,
                                 aberration, rho, diversities=None, mode='vector',
                                 frequency_batch_size=4096, padding=None, *, psf_batch_size=16):
    """CUDA equivalent of reconstruction.evaluate_loss_derivatives_3D."""
    workspace = _workspace(microscope, grid, images, focal_z_levels, sample_z_levels,
                           aberration, rho, diversities, mode, frequency_batch_size,
                           padding, psf_batch_size)
    return workspace.derivatives(aberration.strengths)


def optimize_aberration_3D(microscope, grid, images, focal_z_levels, sample_z_levels,
                           modes, rho, diversities=None, initial_strengths=None,
                           mode='vector', frequency_batch_size=4096, padding=None,
                           max_iterations=50, gradient_tolerance=1e-6,
                           strength_tolerance=1e-8, max_step=.1, max_backtracks=20,
                           verbose=True, *, psf_batch_size=16):
    """CUDA Newton fit; reuses a single GPU workspace across all iterations."""
    workspace = CUDAReconstruction(microscope, grid, images, focal_z_levels,
                                   sample_z_levels, modes, rho, diversities, mode,
                                   frequency_batch_size, padding,
                                   psf_batch_size=psf_batch_size)
    return workspace.optimize(initial_strengths, max_iterations, gradient_tolerance,
                              strength_tolerance, max_step, max_backtracks, verbose)
