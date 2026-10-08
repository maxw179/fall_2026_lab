"""Simulated SLM feedback loops with explicit acquisition budgets.

The stack method fits the current residual aberration, applies its negative,
and acquires new data. The Booth baseline shares one zero-bias measurement
across two probes per mode and commits all modal updates after the sweep.
See Facomprez et al., Optics Express 20, 2598 (2012):
https://doi.org/10.1364/OE.20.002598

SLM phases add ideally to a static unknown wavefront. Truth is used only by
the forward simulator and diagnostic RMS calculation, never to choose updates.
"""

from __future__ import annotations

import numpy as np

from optimization import reconstruction
from utils import zernike


def exposure_for_snr(clean_images, snr):
    """Photons per intensity unit for a reference global RMS shot-noise SNR.

    Calibrate once, then reuse this exposure for every probe and correction.
    Recalibrating each image would change its photon budget with aberration.
    """
    images = np.asarray(clean_images, dtype=float)
    if not np.all(np.isfinite(images)) or images.size == 0 or images.max() <= 0:
        raise ValueError("Reference images must be finite with positive signal.")
    if not np.isfinite(snr) or snr <= 0:
        raise ValueError("Reference SNR must be finite and positive.")
    peak = images.max()
    if images.min() < -1e-12 * peak:
        raise ValueError("Reference images must be nonnegative.")
    unit = np.clip(images / peak, 0, None)
    return float(snr**2 * unit.sum() / (np.sum(unit**2) * peak))


class SimulatedSLM:
    """Forward acquisition under an accumulated correction and temporary probes.

    Every focal plane counts as one image. Optimizer iterations and diagnostic
    evaluations do not count as images. Exposure is shared by both algorithms;
    observations are returned in the original optical intensity units.
    """

    def __init__(self, microscope, sample, truth, correction_modes, exposure,
                 rng, optical_mode="vector"):
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("Exposure must be finite and positive.")
        self.microscope = microscope
        self.sample = sample
        self._truth = truth
        self.modes = [list(mode) for mode in correction_modes]
        if not self.modes or len({tuple(mode) for mode in self.modes}) != len(self.modes):
            raise ValueError("Correction modes must be nonempty and unique.")
        self.exposure = float(exposure)
        self.rng = rng
        self.optical_mode = optical_mode
        self.slm_strengths = np.zeros(len(self.modes))
        self.images_taken = 0
        self.photons_detected = 0

    def set_slm(self, strengths):
        strengths = np.asarray(strengths, dtype=float)
        if strengths.shape != self.slm_strengths.shape or not np.all(np.isfinite(strengths)):
            raise ValueError("Provide one finite SLM coefficient per correction mode.")
        self.slm_strengths = strengths.copy()

    def _residual(self):
        return self._truth + zernike.Aberration(self.modes, self.slm_strengths)

    def residual_rms(self):
        """Simulation-only diagnostic; not available to the correction estimator."""
        return float(zernike.zernike_RMS(self._residual(), self.microscope.alpha))

    def acquire(self, focal_z_levels, diversities=None):
        _, _, _, clean = self.microscope.compute_image_3D(
            image=self.sample, aberration=self._residual(),
            focal_z_levels=focal_z_levels, diversities=diversities,
            mode=self.optical_mode,
        )
        if not np.all(np.isfinite(clean)) or clean.min() < -1e-12 * max(clean.max(), 0):
            raise ValueError("Simulated images must be finite and nonnegative.")
        counts = self.rng.poisson(self.exposure * np.clip(clean, 0, None))
        self.images_taken += len(counts)
        self.photons_detected += int(counts.sum())
        return counts / self.exposure


def _record(simulator, history, **details):
    history.append(dict(
        images_taken=simulator.images_taken,
        photons_detected=simulator.photons_detected,
        slm_strengths=simulator.slm_strengths.copy(),
        residual_rms=simulator.residual_rms(),
        **details,
    ))


def _validate_control(image_budget, gain, max_update):
    if not isinstance(image_budget, (int, np.integer)) or image_budget < 0:
        raise ValueError("Image budget must be a nonnegative integer.")
    if not np.isfinite(gain) or not 0 < gain <= 1:
        raise ValueError("Correction gain must lie in (0, 1].")
    if not np.isfinite(max_update) or max_update <= 0:
        raise ValueError("Maximum modal update must be finite and positive.")


def stack_feedback(simulator, grid, sample_z_levels, focal_positions,
                   diversity_mode, diversity_strengths, rho, image_budget,
                   gain=1.0, max_update=0.1, optimizer_kwargs=None,
                   on_update=None):
    """Reacquire a diverse stack and fit the residual after every SLM update.

    Each fit starts at zero residual, not at the previous estimate of the
    original wavefront. Even an iteration-limited fit can provide a useful
    correction; its status and diagnostics are retained in the history.
    """
    _validate_control(image_budget, gain, max_update)
    positions = np.asarray(focal_positions, dtype=float)
    strengths = np.asarray(diversity_strengths, dtype=float)
    if positions.ndim != 1 or positions.size == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("Provide a nonempty finite vector of focal positions.")
    if strengths.ndim != 1 or strengths.size == 0 or not np.all(np.isfinite(strengths)):
        raise ValueError("Provide a nonempty finite vector of diversity strengths.")
    focal_z = np.repeat(positions, len(strengths))
    diversities = [
        zernike.Aberration([diversity_mode], [strength])
        for _ in positions for strength in strengths
    ]
    kwargs = dict(optimizer_kwargs or {})
    kwargs.setdefault("verbose", False)
    history = []
    _record(simulator, history)
    while simulator.images_taken + len(focal_z) <= image_budget:
        images = simulator.acquire(focal_z, diversities)
        estimated, info = reconstruction.optimize_aberration_3D(
            microscope=simulator.microscope, grid=grid, images=images,
            focal_z_levels=focal_z, sample_z_levels=sample_z_levels,
            modes=simulator.modes, rho=rho, diversities=diversities,
            initial_strengths=np.zeros(len(simulator.modes)),
            mode=simulator.optical_mode, **kwargs,
        )
        if not np.all(np.isfinite(estimated.strengths)):
            raise ValueError("The estimated correction contains nonfinite coefficients.")
        update = -gain * np.clip(estimated.strengths, -max_update, max_update)
        simulator.set_slm(simulator.slm_strengths + update)
        _record(simulator, history, estimated_strengths=estimated.strengths.copy(),
                update=update.copy(), optimizer_info=info)
        if on_update is not None:
            on_update(history[-1])
    return history


def mean_brightness(images):
    """Mean detected intensity, without normalization by image brightness."""
    return float(np.mean(images))


def booth_feedback(simulator, focal_positions, bias, image_budget,
                   gain=1.0, max_update=0.1, metric=mean_brightness,
                   curvature_tolerance=1e-6, on_update=None):
    """Repeated 2N+1 sweeps, using three-point modal parabolic maxima.

    The persistent SLM command stays fixed during a sweep. One shared baseline
    and the +/- bias of each mode are acquired, then the modal updates are
    applied together. A stack of K focal positions costs K*(2N+1) images.
    A flat/convex fit uses the best measured bias; extrapolation is limited by
    max_update. This is a local quadratic baseline, with no mode calibration.
    """
    _validate_control(image_budget, gain, max_update)
    if not np.isfinite(bias) or bias <= 0:
        raise ValueError("Probe bias must be finite and positive.")
    if not np.isfinite(curvature_tolerance) or curvature_tolerance < 0:
        raise ValueError("Curvature tolerance must be finite and nonnegative.")
    positions = np.asarray(focal_positions, dtype=float)
    if positions.ndim != 1 or positions.size == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("Provide a nonempty finite vector of focal positions.")
    frames_per_sweep = len(positions) * (2 * len(simulator.modes) + 1)
    history = []
    _record(simulator, history)
    while simulator.images_taken + frames_per_sweep <= image_budget:
        baseline = float(metric(simulator.acquire(positions)))
        probe_metrics = np.empty((len(simulator.modes), 3))
        modal_updates = np.empty(len(simulator.modes))
        used_parabola = np.zeros(len(simulator.modes), dtype=bool)
        for i, mode in enumerate(simulator.modes):
            negative = zernike.Aberration([mode], [-bias])
            positive = zernike.Aberration([mode], [bias])
            minus = float(metric(simulator.acquire(positions, [negative] * len(positions))))
            plus = float(metric(simulator.acquire(positions, [positive] * len(positions))))
            if not np.all(np.isfinite([minus, baseline, plus])):
                raise ValueError("The image metric must return finite values.")
            probe_metrics[i] = [minus, baseline, plus]
            curvature = minus + plus - 2 * baseline
            scale = max(abs(minus), abs(baseline), abs(plus), np.finfo(float).tiny)
            if curvature < -curvature_tolerance * scale:
                # Signed SLM bias at the maximum of the fitted parabola.
                modal_updates[i] = bias * (minus - plus) / (2 * curvature)
                used_parabola[i] = True
            else:
                # Put zero first so ties do not introduce an arbitrary bias.
                modal_updates[i] = [0.0, -bias, bias][np.argmax([baseline, minus, plus])]
        update = gain * np.clip(modal_updates, -max_update, max_update)
        simulator.set_slm(simulator.slm_strengths + update)
        _record(simulator, history, update=update.copy(), probe_metrics=probe_metrics,
                used_parabola=used_parabola)
        if on_update is not None:
            on_update(history[-1])
    return history


def rms_at_image_counts(history, image_counts):
    """Hold the last committed correction until the next update is available."""
    counts = np.asarray([point["images_taken"] for point in history])
    rms = np.asarray([point["residual_rms"] for point in history])
    requested = np.asarray(image_counts)
    if not history or np.any(requested < counts[0]):
        raise ValueError("Requested image counts must follow the initial state.")
    indices = np.searchsorted(counts, requested, side="right") - 1
    return rms[indices]
