# Single-case AO replay

Open `ao_replay.ipynb` to select a saved SNR-sweep case by RMS, SNR, and trial. `snr_case_replay.html` and `demo.html` show RMS 0.10 waves, SNR 20, trial 1 from the completed cumulative three-loop sweep. The viewer has play/pause, previous/next, a scrubber, playback speed, with acquisition steps hidden. Its standalone HTML export works offline.

For an actual run, replace `correction.SimulatedSLM(...)` with `RecordingSLM(...)` before running the existing feedback function. No optimizer changes are needed:

```python
from systematic_testing.ao_animation.viewer import RecordingSLM, replay_history, show_replay

simulator = RecordingSLM(
    microscope=microscope, sample=mask_3D, truth=truth,
    correction_modes=CORRECTION_MODES, exposure=fixed_exposure,
    rng=np.random.default_rng(NOISE_SEED), optical_mode=OPTICAL_MODE,
)
# Run your existing correction.stack_feedback(...) or correction.booth_feedback(...)
# with this simulator and retain the returned history.
frames, points = replay_history(history, simulator, focus=0.0, max_updates=3)
display(show_replay(frames, points, output_path='ao_case_replay.html'))
```

An ordinary `SimulatedSLM` also works, but old histories do not contain actual acquisition images. In that case replay shows only saved correction states, available estimates, and recomputed noiseless diagnostic images. It does not invent measurements. Use the same truth, sample, optics and modes as the original run. Replay does not change SLM state or consume RNG draws.

The default interprets three loops as three successive corrections of one method. Estimates come from the stored fit; Booth histories have no joint residual estimate. The persistent residual phase excludes temporary probes. RMS plots include intermediate optimizer estimates and committed states. Display intensity and phase scales are shared across the entire replay, throughout replay. Image counts include algorithm exposures only; diagnostics are not acquisitions.

Frames are pre-rendered for portability. Large acquisition stacks produce larger files and take longer to render; use a small case initially. The SNR replay uses saved commands and every archived optimizer iterate. The viewer hides acquisition steps and does not regenerate measurement images. No optimizer is rerun. The loader checks image counts, applied updates, and residual RMS against the archive. The phase and image panels use `utils.plotting` functions, including their polar plasma phase plots in radians and grayscale images with physical axes. After a correction, the displayed estimate is transformed into the new SLM frame; during fitting, the physical residual remains unchanged.

The RMS curve updates at every saved optimizer step. Its axes remain fixed. Panel and colorbar positions are explicit and identical in every frame, including before any estimate exists.

Intermediate RMS is the pupil RMS of true current residual minus each saved fitted estimate (the hypothetical residual after full, unclipped correction). Green squares show actual applied states, including gain/clipping. The horizontal axis follows saved optimization steps, since image count stays constant while fitting.

## Multiple-bead cases

In `ao_replay.ipynb`, set `SWEEP="beads"`, `BEAD_COUNT` to 1–5, and choose `RMS_INDEX` and `TRIAL`. Calibration SNR is 15 for this sweep. The default standalone `beads_case_replay.html` shows three beads, initial RMS 0.10 waves, trial 1.

```python
from systematic_testing.ao_animation.snr_replay import load_bead_case
frames, points, title = load_bead_case(rms_index=1, bead_count=3, trial=0)
display(show_replay(frames, points, title=title))
```

Samples use the archived 3D centers and radii, including overlapping spheres. Saved estimates and applied corrections are checked against the matching checkpoint. No optimization or acquisition is rerun.
