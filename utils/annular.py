"""Annular illumination with matching forward and reconstruction pupil geometry."""
import numpy as np
from utils.psf import Microscope
from utils.rw import gaussian_amplitude_s_perp


class AnnularMicroscope(Microscope):
    """Finite-width Bessel-like beam; radius and width are fractions of pupil radius."""
    def __init__(self, reference, radius=0.8, width=0.1, match_power=False):
        self.__dict__.update(reference.__dict__)
        self.inner = float(radius) - float(width)/2
        self.outer = float(radius) + float(width)/2
        if not (0 <= self.inner < self.outer <= 1):
            raise ValueError('Require 0 <= radius-width/2 < radius+width/2 <= 1.')
        self.match_power = bool(match_power)
        self._geometry_cache = {}

    def pupil_transmission(self, theta, phi, mask):
        radius = np.sin(theta)/np.sin(self.alpha)
        ring = mask & (radius >= self.inner) & (radius <= self.outer)
        if not np.any(ring):
            raise ValueError('Annulus has no sampled points; increase pupil sampling or ring width.')
        gauss = gaussian_amplitude_s_perp(self.mag, self.w_0, self.f, self.n, np.sin(theta[mask]))
        fraction = float(np.sum(gauss[ring[mask]]**2)/np.sum(gauss**2))
        self.transmitted_power_fraction = fraction
        return ring.astype(float) / (np.sqrt(fraction) if self.match_power else 1.)

    def compute_PSF(self, grid, aberration, mode='vector'):
        from optimization._reconstruction_common import _derivative_geometry
        key = (grid.L_ffp_x, grid.L_ffp_y, grid.grid_ffp_x, grid.grid_ffp_y,
               grid.x_offset, grid.y_offset, mode)
        if key not in self._geometry_cache:
            self._geometry_cache[key] = _derivative_geometry(self, grid, [], mode, propagation='dense')
        amplitudes, mask, theta, phi, sz, _, propagate = self._geometry_cache[key]
        phase = np.zeros_like(theta, dtype=complex)
        phase[mask] = np.exp(1j*(aberration.construct_map(self.alpha)(theta[mask], phi[mask])
                                + self.k*grid.z_level*sz[mask]))
        field = propagate(amplitudes * phase)
        x, y = grid.get_xy()
        return x, y, np.sum(np.abs(field)**2, axis=0)**self.N_order
