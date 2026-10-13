"""Calibrate expected global signal/error SNR including shot noise and artifacts."""
import numpy as np


def calibrated_noise(clean, kind, level, target_snr, poisson_seed, added_seed, *, gaussian_mean_sigma=3.0):
    clean = np.maximum(np.asarray(clean, dtype=float), 0)
    if not np.isfinite(target_snr) or target_snr <= 0 or not np.isfinite(level) or level < 0:
        raise ValueError('SNR must be positive and level nonnegative.')
    if not np.isfinite(gaussian_mean_sigma) or gaussian_mean_sigma < 0:
        raise ValueError("Gaussian mean in sigma units must be finite and nonnegative.")
    peak = float(clean.max())
    if peak <= 0:
        raise ValueError('Signal must be nonzero.')
    signal = clean / peak
    signal_power = float(np.sum(signal**2))
    budget = signal_power / target_snr**2
    x, y = np.indices(signal.shape[-2:])
    patterns = {'rows': (-1.)**y, 'columns': (-1.)**x,
                'checkerboard': (-1.)**(x+y)}
    amplitude = 0.
    if kind == 'none':
        modifier = np.ones_like(signal)
        extra_power = 0.
    elif kind in ('gaussian', *patterns):
        # Level is extra-noise RMS / Poisson-noise RMS, rather than a fixed amplitude.
        extra_power = budget * level**2 / (1 + level**2)
        # Gaussian mean contributes squared bias as well as variance to the error budget.
        mean_factor = 1 + gaussian_mean_sigma**2 if kind == "gaussian" else 1
        amplitude = np.sqrt(extra_power / (signal.size * mean_factor))
        modifier = np.ones_like(signal)
    elif kind == 'multiplicative_rows':
        # Level is the fraction of the error-power budget assigned to row modulation.
        if level > 1:
            raise ValueError('Modulation budget fraction must be <= 1.')
        # Retain some photon-noise budget even for requested fraction 1.
        fraction = min(level, .95)
        amplitude = np.sqrt(fraction * budget / signal_power)
        modifier = 1 + amplitude * patterns['rows']
        extra_power = float(np.sum((signal * (modifier-1))**2))
    else:
        raise ValueError(kind)
    photon_budget = budget - extra_power
    # Multiplication occurs after Poisson sampling, so photon variance scales with modifier^2.
    exposure_peak = float(np.sum(signal * modifier**2)) / photon_budget
    exposure = exposure_peak / peak
    counts = np.random.default_rng(poisson_seed).poisson(exposure_peak * signal)
    observed = counts / exposure_peak
    if kind == 'gaussian':
        observed += amplitude * (gaussian_mean_sigma +
                                  np.random.default_rng(added_seed).normal(size=signal.shape))
    elif kind in patterns:
        observed += amplitude * patterns[kind]
    elif kind == 'multiplicative_rows':
        observed *= modifier
    expected = np.sqrt(signal_power / (np.sum(signal*modifier**2)/exposure_peak + extra_power))
    realized = np.sqrt(signal_power / np.sum((observed-signal)**2))
    return dict(image=observed*peak, counts=counts, exposure=exposure,
                expected_snr=float(expected), realized_snr=float(realized),
                gaussian_mean=float(amplitude*peak*gaussian_mean_sigma if kind == "gaussian" else 0),
                artifact_amplitude=float(amplitude*peak if kind != 'multiplicative_rows' else amplitude))
