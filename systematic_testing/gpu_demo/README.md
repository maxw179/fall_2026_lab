# GPU cumulative AO demonstration

This runs the bead sweep's three-loop cumulative feedback experiment at 128 ×
128 image pixels, recording core computation time and a portable reconstruction
history. The input is a simulated three-photon vector image stack of five random
beads with an initial pupil RMS of 0.20 waves. Plotting and animations run later
on your Mac using saved arrays, without optimization or CPU object refits.

## Experiment

Defaults match `bead_count_sweep/pd_rms_bead_count_sweep.ipynb` except for the
128 × 128 lateral image grid. The pupil grid remains 32 × 32, as in the sweep;
`--pupil-size 128` is available for a separate higher-resolution pupil experiment.

- 6.5 µm lateral field; five beads of **1 µm radius**, with overlaps allowed.
- 17 fitted Zernike modes; deterministic aberration seed 0 and bead seed 2,
  zero-based trial 0. Pupil RMS uses the existing annular geometry helper.
- Shot noise at calibration SNR **15**, measured from the initial aberrated
  z=0 image with no diversity. That exposure stays fixed across all loops.
- Four acquisition focal positions: −2, −2/3, +2/3, +2 µm.
- 41 truth-object planes and 21 reconstruction planes, both spanning −2 to +2 µm.
- Three normalized diversity modes per loop, each at −0.10, 0, +0.10 waves RMS.
  Loop 1: `[-2,2], [0,4], [2,2]`; loop 2: `[-3,3], [0,4], [3,3]`;
  loop 3: `[-4,4], [0,4], [4,4]`.
- 36 fresh images per loop, **36 / 72 / 108 images in cumulative fits**. Duplicate
  zero-diversity frames remain independent noisy measurements, as in the sweep.
- Gain 1; each modal correction is clipped at ±0.10 waves before application.
- `rho=1e46`; exact Newton Hessians; up to 50 accepted steps per loop.
- 100 random candidates plus zero per fit, with the sweep's RMS range
  [0.005, 0.50] and seed 0. Use `--random-starts 1000` to try more candidates.
- CUDA QR SVD and guarded loss-only linear screening; requested candidate batch
  size 8. The existing memory guard can reduce the effective batch size.

The shared `correction.stack_feedback` controller implements the feedback and
cumulative phase bookkeeping: historical known phases equal the acquisition
diversity plus acquisition SLM phase minus the current SLM phase. The true
aberration never enters screening, derivatives or update selection.

## Queue on Bouchet

Commit and push the source changes on your Mac first. On Bouchet:

```bash
cd /nfs/roberts/project/pi_cr876/mcw77/fall_2026_lab
git pull --ff-only
mkdir -p logs
sbatch systematic_testing/gpu_demo/bouchet_demo.sbatch
```

The script requests one H100, H200 or B200, whichever partition can schedule it.
It first checks the small cumulative CUDA workflow and offline playback tests.
To request H100 specifically:

```bash
sbatch --partition=gpu_h100 systematic_testing/gpu_demo/bouchet_demo.sbatch
```

To screen 1,000 random starts in each loop:

```bash
sbatch systematic_testing/gpu_demo/bouchet_demo.sbatch --random-starts 1000
```

Track the returned job ID:

```bash
squeue -j JOB_ID
tail -f logs/pd-ao-demo-JOB_ID.out
cat results/bouchet/gpu_demo/JOB_ID/run.json
```

Simulation runs on the CPU and is reported separately from GPU reconstruction.
This job uses 17 fitted modes and cumulative acquisitions, so its runtime is
measured independently of the earlier two-mode synthetic benchmarks. Logs show
when acquisition, each cumulative fit, and subsequent object replay begin.

## What is timed and saved

`run.json` checkpoints after each AO loop and records the actual GPU, software,
Git revision, optimizer success/message, effective screening batch, and SVD/
screening fallback counts. `core_seconds` is synchronized host-visible wall time
for **workspace construction + candidate screening + Newton optimization**,
summed over the three loops. Each loop also reports `workspace_seconds`,
`optimization_seconds` (screening included), and `screening_seconds`.

CUDA context initialization occurs before the core timer. There is no full-size
warmup: first-use kernel compilation encountered within a fit is included.
Acquisition simulation, calibration, RMS diagnostics, disk checkpoints and object
replay are excluded. `wall_seconds` includes all of those parts except local
plotting. Lightweight timestamp callbacks remain in the measured optimizer.

- `scene.npz`: true object, original wavefront coefficients, bead centers,
  physical axes, calibration image and microscope pupil angle.
- `loop_1_input.npz` through `loop_3_input.npz`: the exact cumulative noisy
  observations and diversity phases seen by each fit, including SLM offsets.
- `loop_*_screening.npz`: every candidate and its loss, plus the selected index.
- `history.npz`: every recorded optimizer state, coefficient vector, loss,
  estimation RMS error and cumulative core timestamp; applied SLM coefficients,
  applied residual RMS and loop boundary times are also saved.
- `object_history.npy`: float32 copies of estimated 3D objects, reconstructed on
  the selected backend **after all three loops**, with no Hessians. Optimization
  remains double precision. Local plotting memory maps this file.
- `object_frames.npz`: the history indices corresponding to the saved objects.
- `correction_preview.npz`: the four-state image panels used by the sweep example
  helper. Before correction / after loops 1 and 2 use actual zero-diversity
  measurements. The after-loop-3 column is a separate simulated diagnostic,
  labeled `(sim.)`, generated after optimization and excluded from core timing.
- `environment.txt` and `git-status.txt`: batch-job environment and source state.

Every iteration is saved by default. `--object-stride 3` reduces post-run object
replay and file size while always retaining each loop's last state. It does not
change optimization or RMS history. Partial runs retain completed-loop numerical
checkpoints; object playback becomes available after replay completes. A completed
job can have iteration-limited optimizers: check `all_optimizers_converged` and
each loop's message separately from the applied residual RMS.

## Download and plot on your Mac

Run from your local repository, replacing `JOB_ID`. Use Yale VPN or the campus
network for the transfer host:

```bash
mkdir -p results/bouchet/gpu_demo/JOB_ID
rsync -avP \
  mcw77@transfer-bouchet.ycrc.yale.edu:/nfs/roberts/project/pi_cr876/mcw77/fall_2026_lab/results/bouchet/gpu_demo/JOB_ID/ \
  results/bouchet/gpu_demo/JOB_ID/

conda activate phase_diversity_env
python -m systematic_testing.gpu_demo.plot_demo results/bouchet/gpu_demo/JOB_ID
```

This writes `figures/rms_time.{png,pdf}`, `rms_iterations.{png,pdf}`,
`acquired_images.{png,pdf}`, `final_aberration.{png,pdf}`, `final_object.{png,pdf}`,
and two GIFs: `aberration.gif` and `object.gif`. It directly reuses the sweep
example plotting helpers and the original testing notebooks' polar-phase and
3D-object animation helper: `bmh`, LaTeX text, 300 DPI static exports, 6 × 6-inch
RMS/correction figures, polar phase insets, shared normalization and the existing
loop annotations and external legend. The time-axis plot changes only the
horizontal coordinate. The saved estimates retain their unconstrained,
potentially negative values.

The RMS curves show the same two diagnostics as the sweep examples: **estimation
error inside each fit**, and **residual RMS after the clipped AO correction**.
The time axis excludes acquisition, replay and export; it measures cumulative
core computation time rather than total job wall time. Animation titles include
these timestamps, but the GIF uses fixed-duration frames rather than real-time
playback. Phase animations subtract the current SLM phase from the fitted
residual estimate, showing the inferred original static aberration across loops.

Animations use the original notebook's 92nd-percentile object threshold.
To reduce local rendering time while retaining each loop's final state:

```bash
python -m systematic_testing.gpu_demo.plot_demo results/bouchet/gpu_demo/JOB_ID \
  --stride 3
```

This uses `utils.plotting.animate_reconstruction` with precomputed samples;
it performs no inverse fits on the Mac. `--no-animations` exports only static
plots. LaTeX is enabled by default, matching the examples; `--no-tex` is an
explicit fallback for machines without LaTeX.

Open `demo_analysis.ipynb` for interactive local inspection and notebook playback.
The notebook selects the most recently downloaded demo by default; its path is
printed so you can choose another run explicitly.

## Local smoke check

This small CPU run validates the complete export/playback workflow, not GPU speed:

```bash
python -m systematic_testing.gpu_demo.run_demo --backend cpu \
  --size 8 --pupil-size 12 --sample-planes 9 --reconstruction-planes 3 \
  --random-starts 2 --max-iterations 1 --output /tmp/pd_demo_smoke
python -m systematic_testing.gpu_demo.plot_demo /tmp/pd_demo_smoke
```

The destination must be new, so previous runs are never overwritten. CUDA tests:

```bash
REQUIRE_CUDA=1 python -m unittest discover -s tests -p test_gpu_demo.py -v
```
