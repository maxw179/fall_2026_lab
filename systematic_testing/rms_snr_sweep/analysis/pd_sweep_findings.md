# Diagnosis of the completed PD RMS–SNR sweep

## Data and scope

Analyzed `systematic_testing/rms_snr_sweep/results/pd_rms_snr_sweep_results_large_sweep.npz`: 189 runs, three full AO loops each, all 567 loop results present. The other file, `pd_rms_snr_sweep_results.npz`, contains only two completed loops and was excluded from aggregate conclusions. Notebook output contains an older 25-run summary; this report uses the saved arrays.

The RMS axis contains two 0.15 entries. These use identical wavefronts but independent noise seeds. Heatmaps and CSV group these entries at the same RMS. They provide six noise observations per SNR, but still only three independent wavefront directions. Other RMS values have three observations per SNR. This is too few wavefronts for a precise general failure probability.

Representative forward simulations were rebuilt using the current 32×32 image/pupil setup and saved sample, exposure, diversity, seed, and modal parameters. The unaberrated exposure calibration matches the archive; three unclipped first-loop fitted losses were reproduced exactly. Original results and optimization code were not changed.

## Patterns

| AO loop | RMS improves over previous state | RMS worsens | Optimizer reports convergence |
|---|---:|---:|---:|
| 1 | 128/189 | 61/189 | 189/189 |
| 2 | 82/189 | 107/189 | 185/189 |
| 3 | 47/189 | 142/189 | 189/189 |

105 of the 107 worsening second loops also report convergence. All 142 worsening third loops report convergence. Convergence means a stationary solution of the fitted-image objective, not a correct phase estimate or a beneficial AO update. More optimizer steps alone cannot be expected to repair these cases.

Mean residual RMS, averaging over all SNRs and trials (grouping duplicate 0.15 entries):

| Initial RMS | After loop 1 | After loop 2 | After loop 3 |
|---:|---:|---:|---:|
| 0.05 | 0.0147 | 0.0153 | 0.0193 |
| 0.10 | 0.0598 | 0.0944 | 0.1046 |
| 0.15 | 0.0668 | 0.0735 | 0.0785 |
| 0.20 | 0.1402 | 0.1300 | 0.1337 |
| 0.25 | 0.2308 | 0.2456 | 0.3120 |
| 0.30 | 0.2803 | 0.2862 | 0.2918 |
| 0.40 | 0.3807 | 0.3929 | 0.4136 |
| 0.50 | 0.4929 | 0.4908 | 0.5083 |

At 0.05-wave initial RMS and reference SNR 100, the final mean is 0.00310 waves. At 0.25 waves and above, capture becomes unreliable. The nonmonotonic behavior at 0.10 versus 0.15 is associated with particular wavefront directions and noise-dependent solution basins; it is not evidence that 0.15-wave aberrations are intrinsically easier.

Trial 3 is especially fragile: mean residual/initial RMS over the stored sweep is 0.941, 1.191, 1.332 after successive loops. Trial 2 yields 0.484, 0.442, 0.473. The same three wavefront directions are scaled at every sweep point, so these are conditional patterns for those directions.

## Confirmed cause: local minima of the phase/object objective

For initial RMS 0.10, reference SNR 100, trial 3, loop 1:

- The archived converged solution produces residual RMS **0.16234 waves**.
- None of its updates is clipped, so the actual estimated phase is recoverable as minus the saved SLM command.
- Its replayed fitted-image loss is **1.73480e55**, exactly the archived loss.
- The loss at the known true phase, on exactly the same noisy images, is **7.34768e54**: the failed solution's loss is **2.36 times larger**.
- At the failed solution, scaled gradient infinity norm is **5.36e-7**, below the 1e-6 convergence threshold. The minimum eigenvalue of the scaled, symmetrized Hessian is positive (**4.48**). This supports a genuine local minimum rather than an unfinished descent or an obvious saddle.
- A straight-line scan from the failed estimate toward truth crosses a loss barrier approximately **8.13 times** the failed solution's loss. Backtracking therefore cannot simply step along that path toward the correct basin.
- Starting the same noisy fit at the true phase, as a diagnostic only, converges in three steps to loss **7.32189e54**, with phase error **0.00146 waves**.

This experiment proves that the images and existing objective contain a much better solution, but the original zero-start optimization did not reach it. Using truth as an initializer is not a proposed operating method. A repeat optimization from zero on clean data was interrupted; no claim about its endpoint is made here.

There is a second useful observation from the duplicate 0.15 entries. At reference SNR 100, trial 3, the same true wavefront gives final RMS **0.00535** for one noise realization and **0.55020** for the other. This extreme divergence is consistent with noise changing which basin is selected.

## Confirmed cause: nominal SNR greatly overstates available signal for large aberrations

The SNR axis is intentionally defined using the unaberrated z=0 reference. Exposure then stays fixed. It does not enforce the same SNR on each aberrated stack.

Recomputed clean first-loop stacks give the expected global shot-noise SNR

`SNR_stack = sqrt(exposure * sum(clean**2) / sum(clean))`.

At reference SNR 100, the ranges across the three wavefront directions are:

| Initial RMS | Expected initial-stack SNR |
|---:|---:|
| 0.05 | 63.2–64.5 |
| 0.10 | 43.4–50.5 |
| 0.15 | 24.2–38.2 |
| 0.20 | 12.7–27.6 |
| 0.25 | 7.85–19.0 |
| 0.30 | 5.10–12.7 |
| 0.40 | 2.71–6.10 |
| 0.50 | 1.35–3.82 |

These are global stack values, not per-pixel SNR. They scale linearly with reference SNR. At 0.50-wave RMS, a nominal SNR of 5 therefore corresponds to approximately 0.067–0.191 for the first stack. Collected intensity is only 0.30–1.21% of the corresponding unaberrated diverse stack. Aberration can spread excitation and reduce nonlinear fluorescence; the finite FOV also limits collected signal. These calculations do not separate those contributions.

Consequently the sweep varies both phase difficulty and actual photon information. Raising reference SNR helps, but even the reference-SNR-100 column can be photon-starved at large RMS.

## Why later AO loops can worsen the result

The correction sign and RMS readout were checked. Library `zernike_RMS` reproduces archived residual RMS for nine representative loop states. The simulator correctly measures truth plus accumulated SLM command, while the optimizer fits the current residual from newly acquired images. There is no evidence here of a repeated-original-aberration subtraction bug.

With gain 1 and no clipping, `residual_next = residual_current - estimate`. A wrong estimate therefore becomes the entire next residual. In the 0.10-wave, reference-SNR-15, trial-3 example, residual grows **0.100 → 0.167 → 0.265 → 0.441**. None of these three updates is clipped.

Diversity decreases from ±0.10 to ±0.08 to ±0.064 waves. This is a plausible contributor to reduced information near successful corrections, but it was not isolated by a controlled rerun. Fresh noise can also dominate when the remaining aberration is already small. In a successful 0.05-wave/SNR-100 example, residuals are **0.00182 → 0.00316 → 0.00230**: later iterations are fluctuating around a small estimation floor rather than improving monotonically.

At high initial RMS, modal clipping also constrains how much phase can be removed in three updates. It is not the primary explanation for the unclipped failures above, and can limit damage from large wrong estimates as well as limit useful correction.

## Secondary hypotheses, not established causes

- The likelihood is regularized Gaussian least squares while acquisition is Poisson. This can be inefficient in photon-starved regimes. A microscopy study comparing the models reports advantages for a Poisson formulation in several conditions, but does not establish its benefit for this specific three-photon 3D model: [Reiser et al., 2023](https://arxiv.org/abs/2308.00734).
- Jointly fitting an unconstrained 21-plane object to data simulated with 41 sample planes adds possible object/phase tradeoffs and model approximation. The fact that truth gives a much lower loss than the failed estimate shows that this mismatch alone does not explain the demonstrated bad basin.
- The RMS metric weights the full clear pupil, while excitation illumination and image data weight modes differently. The largest aggregate third-loop coefficient errors include (-4,4), (-3,5), and (2,2), but these are also substantial components of some input wavefronts. This ranking by itself does not establish an observability defect in those modes.
- Neither Gaussian-object regularization nor the full Hessian proves global identifiability. The precise physical origin of each secondary basin was not fully isolated.

## Tests to prioritize

1. **Multiple starts on the same acquired stack**, select the solution by the data objective. Keep the exact Hessian. This targets the directly demonstrated basin failure. Never select by ground-truth RMS in a practical method.
2. **Measure actual stack signal/SNR and record photon counts**, rather than interpret the reference SNR as the measured stack SNR. Preserve fixed exposure if comparing equal illumination conditions.
3. **Compare fixed ±0.10 diversity against decay**, using matched wavefronts and reproducible noise. This isolates whether probe shrinkage harms later loops.
4. **Compare gain 1 against a smaller gain and/or a measured-image validation acquisition.** Optimizer convergence should not be treated as proof that applying the correction improves the optics. Do not compare raw optimizer losses across newly acquired stacks as a validation metric; each loss belongs to different data.
5. **Known-object and matched-axial-grid diagnostic fits**, to distinguish phase/object ambiguity from the optimizer's capture range. Then consider Poisson fitting if photon-starved performance remains limiting.

Repeat across more independent wavefront directions before estimating general success rates. No sweep parameters or optimizer implementation were changed in this analysis.

## Artifacts

- `pd_sweep_residual_ratios.png`: grouped RMS/SNR heatmaps; 1 means no net improvement.
- `pd_sweep_failure_examples.png`: representative trajectories and improvement counts.
- `pd_sweep_local_minimum.png`: loss barrier between one failed estimate and truth.
- `pd_sweep_signal_loss.png`: actual expected stack SNR and collected signal loss.
- `pd_sweep_modal_errors.png`: aggregate residual coefficient magnitudes.
- `pd_sweep_summary.csv`: grouped numeric summaries.
- `pd_case_diagnostics.json`, `pd_basin_diagnostics.json`, and diagnostic NPZ files: targeted replay measurements.
- `analyze_pd_sweep.py`: reproducible saved-array analysis, without re-running the sweep.
