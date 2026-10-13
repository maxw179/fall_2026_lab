"""Simple fixed-exposure camera model in photoelectrons and ADU.

The returned raw image is analog ADU; ADC rounding/saturation are deliberately
omitted. Patterns and sensitivity maps are fixed over all frames of a stack.
"""
import numpy as np


def simulate_camera(signal_e, *, read_sigma_e=0., background_e=0., dark_e=0.,
                    pedestal_adu=64., gain_adu_per_e=1., pattern='none',
                    offset_pattern_e=0., sensitivity_fraction=0.,
                    poisson_seed=0, read_seed=1):
    signal = np.asarray(signal_e, dtype=float)
    parameters = [read_sigma_e, background_e, dark_e, pedestal_adu,
                  gain_adu_per_e, offset_pattern_e, sensitivity_fraction]
    if signal.ndim != 3 or not np.all(np.isfinite(signal)) or np.any(signal < 0):
        raise ValueError('signal_e must be a finite nonnegative [frame,x,y] stack.')
    if not np.all(np.isfinite(parameters)) or any(p < 0 for p in parameters):
        raise ValueError('Camera parameters must be finite and nonnegative.')
    if gain_adu_per_e <= 0 or sensitivity_fraction >= 1:
        raise ValueError('Gain must be positive and sensitivity fraction less than one.')
    x, y = np.indices(signal.shape[-2:])
    patterns = {'none': np.zeros_like(x, dtype=float), 'rows': (-1.)**y,
                'columns': (-1.)**x, 'checkerboard': (-1.)**(x+y)}
    if pattern not in patterns:
        raise ValueError(pattern)
    fixed_pattern = patterns[pattern]
    sensitivity = 1 + sensitivity_fraction*fixed_pattern
    offset_map = offset_pattern_e*fixed_pattern
    background_mean = sensitivity*background_e + dark_e
    streams = np.random.SeedSequence(poisson_seed).spawn(2)
    # Independent signal/background streams preserve matched signal draws
    # when adding background, offsets or read noise to the same scene.
    signal_counts = np.random.default_rng(streams[0]).poisson(signal*sensitivity)
    background_counts = np.random.default_rng(streams[1]).poisson(
        np.broadcast_to(background_mean, signal.shape))
    read = read_sigma_e*np.random.default_rng(read_seed).normal(size=signal.shape)
    electrons = signal_counts + background_counts + read + offset_map
    raw = pedestal_adu + gain_adu_per_e*electrons
    # Usual global pedestal/gain correction; spatial/background errors remain.
    pedestal_only = (raw-pedestal_adu)/gain_adu_per_e
    # Ideal calibration uses known fixed offsets, sensitivity and background means.
    known_maps = (pedestal_only-offset_map-background_mean)/sensitivity
    variance = signal*sensitivity + background_mean + read_sigma_e**2
    bias = signal*(sensitivity-1) + background_mean + offset_map
    return dict(raw_adu=raw, pedestal_only_e=pedestal_only, known_maps_e=known_maps,
                signal_counts=signal_counts, background_counts=background_counts,
                sensitivity=sensitivity, offset_map_e=offset_map,
                background_mean_e=background_mean, variance_e=variance,
                bias_e=bias, known_maps_variance_e=variance/sensitivity**2)


def camera_metrics(signal_e, observation, calibration, roi):
    """Per-frame expected ROI temporal SNR and whole-stack signal/error ratio."""
    signal = np.asarray(signal_e, dtype=float)
    if calibration == 'pedestal_only':
        variance = observation['variance_e']
        bias = observation['bias_e']
    elif calibration == 'known_maps':
        variance = observation['known_maps_variance_e']
        bias = np.zeros_like(signal)
    else:
        raise ValueError(calibration)
    power = np.sum(signal[:, roi]**2, axis=1)
    temporal_snr = np.sqrt(power/np.sum(variance[:, roi], axis=1))
    total_error_snr = np.sqrt(np.sum(signal**2)/np.sum(variance+bias**2))
    return temporal_snr, float(total_error_snr)
