"""Reproducible phase candidates for loss-only initialization screening."""
import numpy as np
from utils.zernike import Aberration, zernike_RMS


def random_aberration_starts(modes, alpha, count, rms_range=(0.005, 0.50), seed=0):
    """Gaussian coefficient directions, rescaled to log-uniform pupil RMS waves.

    Drawing direction and radius together preserves prefixes as count increases.
    Only the supplied correctible modes are populated; no true phase is used.
    """
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError('num_random_starts must be a nonnegative integer.')
    bounds = np.asarray(rms_range, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or not 0 < bounds[0] <= bounds[1]:
        raise ValueError('random_start_rms_range must satisfy 0 < lower <= upper, in waves.')
    rng = np.random.default_rng(seed)
    starts = np.empty((count, len(modes)))
    for index in range(count):
        direction = rng.normal(size=len(modes))
        radius = np.exp(rng.uniform(np.log(bounds[0]), np.log(bounds[1])))
        norm = zernike_RMS(Aberration(modes, direction), alpha)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('Correctible modes must span a nonzero pupil phase.')
        starts[index] = direction * radius / norm
    return starts
