# GPU cumulative AO demonstration

This runs the bead sweep's three-loop cumulative feedback experiment at 128 ×
128 image pixels, recording core computation time and a portable reconstruction
history. The input is a simulated three-photon vector image stack of five random
beads with an initial pupil RMS of 0.20 waves. Local playback uses yesterday's
AO image/wavefront viewer, without optimization, inverse object fits or voxels.
It simulates four diagnostic images from saved applied SLM states and adds
Poisson noise at the archived fixed exposure.

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
- CUDA QR SVD and guarded loss-only linear screening, with batched augmented QR
  for frequencies outside the linear guard; requested candidate batch
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
when acquisition and each cumulative fit begin. Dense object-history export is
optional; the image/wavefront replay does not use it.

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
- `object_history.npy` (only with `--save-objects`): float32 copies of estimated 3D objects, reconstructed on
  the selected backend **after all three loops**, with no Hessians. Optimization
  remains double precision. Local plotting memory maps this file.
- `object_frames.npz` (only with `--save-objects`): the history indices corresponding to the saved objects.
- `correction_preview.npz`: the four-state image panels used by the sweep example
  helper. Before correction / after loops 1 and 2 use actual zero-diversity
  measurements. The after-loop-3 column is a separate simulated diagnostic,
  labeled `(sim.)`, generated after optimization and excluded from core timing.
- `environment.txt` and `git-status.txt`: batch-job environment and source state.

Every optimizer iteration is saved by default. `--save-objects` opts into dense
object-history export; `--object-stride 3` then reduces that optional export while
retaining each loop's final state. Neither changes optimization or RMS history.
Partial runs retain completed-loop numerical checkpoints. A completed
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
`acquired_images.{png,pdf}`, `final_aberration.{png,pdf}`, and the standalone
`figures/ao_replay.html`. Static plots directly reuse the sweep example helpers:
`bmh`, LaTeX text, 300 DPI static exports, 6 × 6-inch
RMS/correction figures, polar phase insets, shared normalization and the existing
loop annotations and external legend. The time-axis plot changes only the
horizontal coordinate. The saved estimates retain their unconstrained,
potentially negative values.

The RMS curves show the same two diagnostics as the sweep examples: **estimation
error inside each fit**, and **residual RMS after the clipped AO correction**.
The time axis excludes acquisition, replay and export; it measures cumulative
core computation time rather than total job wall time. Animation titles include
these timestamps, but playback uses fixed-duration frames rather than real time.
The existing AO viewer shows true residual, estimated current residual and
accumulated SLM phase in three polar panels, initial/current noisy simulated
images, and the saved RMS progress trace. The estimated wavefront updates at
every saved optimizer step. Images and physical residual change only at applied
corrections, because the SLM is fixed during each fit. After application, the
last estimate is expressed in the new SLM frame, as in yesterday's demonstration.
Image and phase scales remain fixed throughout. Diagnostic images do not count
as acquisitions and do not consume the original noise RNG. Their fresh Poisson
draws use a separate reproducible seed, one realization per applied SLM state,
and the original exposure calibrated to the initial reference SNR.

Open the HTML file directly in your browser for play/pause, previous/next,
scrubbing and playback-speed controls:

```bash
open results/bouchet/gpu_demo/JOB_ID/figures/ao_replay.html
```

To show fewer intermediate optimizer states while retaining each loop's last state:

```bash
python -m systematic_testing.gpu_demo.plot_demo results/bouchet/gpu_demo/JOB_ID \
  --stride 3
```

This uses `systematic_testing.ao_animation.viewer` from the existing AO demo.
It performs no inverse fits on the Mac. `--no-animations` exports only static
plots. LaTeX is enabled by default, matching the examples; `--no-tex` is an
explicit fallback for machines without LaTeX.

Open `demo_analysis.ipynb` for interactive local inspection and notebook playback.
The notebook selects the most recently downloaded demo by default; its path is
printed so you can choose another run explicitly. Interrupt any older voxel
cell, rerun the setup cell to reload the helpers, then run the new replay cell.

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

## Shared-baseline 64 × 64 camera-noise run

The runner also accepts `--shared-zero --diversity-settings -.10 -.05 .05 .10`.
This acquires one zero-diversity frame plus four settings for each of three
modes at each of four z positions: 52 fresh frames per loop, and 52 / 104 / 156
frames in the cumulative fits. `--size 64` changes only the image grid.

Camera parameters `--read-sigma-e`, `--background-e` and `--checkerboard-e`
combine Gaussian read noise, Poisson background and fixed alternating checker
pixel offsets with signal Poisson noise. They use the noise-testing
`simulate_camera` model. Only the known electronic pedestal/gain is removed;
background and checkerboard remain in the observations. Negative values are
retained. `--snr` calibrates signal-only shot-noise exposure from the initial
aberrated z=0 image; it is not the total SNR after adding the other components.
Exposure and camera parameters remain fixed through the AO loops.

Add `--prepare-only` to generate the exact first 52 noisy frames locally without
CUDA or optimization. This writes `scene.npz`, `loop_1_input.npz` and `run.json`
with status `prepared`. Repeat the same command without `--prepare-only`, using
a new output directory, for the full feedback run. The first loop is identical
for the same configuration and NumPy version. Later acquisitions depend on the
preceding fitted corrections and therefore cannot be prepared before fitting.

In `demo_analysis.ipynb`, set `RUN_DIR` to the prepared directory for a preview
of all 52 actual frames in the existing image/wavefront viewer. Set it to the
completed cluster directory for RMS plots and AO replay. Shared-baseline runs
show archived noisy zero-diversity frames in replay, at the acquired plane
nearest `FOCUS_MM`; no denoising or inverse refitting is performed. After the
third correction there is no fourth acquisition loop: its displayed image is
an independently noised diagnostic with the same detector parameters and is
labeled as simulated. Probe frames remain in each `loop_*_input.npz`.
