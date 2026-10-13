# Physical detector model for the PD noise experiment

This design replaces the previous SNR-budget-driven image corruptions. It assumes a camera; confirm the detector before implementation. If the three-photon images represent a raster scan measured by a PMT/APD, pixel patterns instead require a scan/electronics model.

## Model and units

The optical simulator returns an uncalibrated intensity distribution. Its saved `exposure` is a counts-per-simulation-unit multiplier, not an independently measured exposure duration in seconds. Define one fixed conversion to expected photoelectrons from a reference image. Specify integration time separately if modeling dark current or physical optical background rates.

For frame k and pixel i:

- s_ki = nominal signal photoelectrons predicted from the optical image at the selected integration time.
- q_i = relative optical sensitivity, mean one, fixed across the experiment.
- b_ki = expected optical-background photoelectrons.
- d_i t = expected dark-current electrons.
- N_ki ~ Poisson(q_i (s_ki + b_ki) + d_i t).
- R_ki ~ Normal(0, sigma_read_i^2), independently drawn per pixel/frame.
- Optional C_kr is a row-correlated temporal readout term, freshly drawn per frame.
- raw_ki = quantize_and_clip(o_i + g_i (N_ki + R_ki + C_kr)), in ADU.

Electronic gain g_i is ADU/electron. Offset o_i is ADU. Draw sensitivity, gain, offset and dark-current maps once, then retain them across all frames and AO loops. Temporal fluctuations are resampled each exposure. Sensitivity changes the Poisson mean before sampling; an electronic gain map scales the sampled charge afterward. These are different physical mechanisms.

A positive raw background can arise from a digital pedestal, optical background, or dark current. Keep these distinct. Gaussian read noise remains zero-mean around the expected detector output; a nonzero Gaussian mean represents a separate offset.

## Initial conditions to compare

1. Photon shot noise alone.
2. Shot noise plus Gaussian read noise, specified in electrons RMS.
3. Shot noise plus optical background, specified in expected electrons/pixel/exposure.
4. Shot noise plus fixed additive row/column offsets, specified in electrons or ADU.
5. Shot noise plus fixed sensitivity variation, specified as a fractional response map.
6. Optional mixed case, hot pixels, and time-varying correlated row readout.

Alternating stripes and checkerboards are explicit synthetic stress cases. Use measured dark/flat maps, or declared hypothetical pixel/row/column map distributions, for a camera-characterization experiment. Do not choose amplitudes based on apparent darkness of the display.

## Exposure and metrics

Use fixed signal flux, exposure duration, camera maps and conversion gain for the primary comparison. Vary one detector parameter; let SNR change. Preserve paired underlying signal-photoelectron samples where the physical signal rate is unchanged. Background/dark Poisson draws, readout draws and fixed maps use separate RNG streams.

For a uniform calibrated detector without correlated noise, temporal per-pixel SNR is

    SNR_ki = s_ki / sqrt(s_ki + b_ki + d_i t + sigma_read_i^2).

A removable electronic pedestal does not add shot noise. Subtracting an optical-background mean leaves its photon shot noise. Report remaining fixed-pattern bias separately. If desired, also report a total signal/error measure that explicitly includes squared bias and covariance.

Report signal-region SNR per frame, background mean/std, fixed-pattern RMS and signal electron counts. A whole-stack RMS ratio is a supplementary scalar; it depends on empty-pixel fraction and cannot predict visible background brightness by itself.

A secondary matched-SNR comparison may vary photon flux or integration time while keeping detector parameters fixed. State the reference frame/region and the metric; solve for exposure rather than shrinking all artifacts to fill a prescribed error budget. If exposure time changes, optical background and dark current must change consistently. Correlation and systematic bias can matter even at equal scalar SNR.

## Data supplied to correction

Store raw ADU, calibrated images, all detector maps/parameters, RNG seeds and acquisition metadata. Compare realistic calibration (offset/dark subtraction and flat-fielding), omitted calibration, and imperfect calibration. Keep finite calibration-frame uncertainty optional and explicit.

Convert preprocessed measurements back to the forward model's intensity units before using the existing loss and rho. Otherwise the regularization scaling changes. Bias-subtracted observations can be negative; do not clip them. ADC clipping belongs in the raw-data model, and saturated pixels require explicit handling.

The current reconstruction uses an unweighted regularized least-squares loss. Initially keep it fixed to study robustness. A camera-aware likelihood/weighting is a separate comparison; a Poisson-plus-Gaussian detector model does not automatically change the reconstruction objective.

## Cheap diagnostics and validation

Run loss evaluations on fixed observed stacks along a zero-to-truth path and a few small phase perturbations near truth. Refit the object at every phase point. Compare where sampled minima lie and whether artifacts introduce barriers or bias; do not infer optimizer success from one decreasing path.

Before costly fits, validate the detector simulator on repeated uniform bright and dark frames: electron-domain mean/variance, fixed map persistence, read-noise amplitude, pedestal subtraction, gain conversion and reproducible RNG streams. For the simple ideal model, mean bright electrons are s+b+d*t and temporal variance is s+b+d*t+sigma_read^2.

## References

- EMVA 1288 linear camera characterization, including temporal noise, gain, sensitivity and spatial nonuniformity: https://www.emva.org/wp-content/uploads/EMVA1288Linear_4.0Release.pdf
- Hamamatsu SNR guide, signal/background photon noise and read noise: https://camera.hamamatsu.com/us/en/learn/technical_information/thechnical_guide/calculating_snr.html
- Hamamatsu camera simulation guide, electronic offset, read noise and dark current: https://www.hamamatsu.com/sp/sys/en/camera_simulator/index.html
- Konnik and Welsh, high-level CCD/CMOS noise modeling: https://arxiv.org/abs/1412.4031
