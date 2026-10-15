# Single-bead RMS cliff baseline

Twenty seeded random aberration directions are each tested at twenty initial
pupil RMS levels: **400 independent cases**, with three cumulative AO loops per
case. One Slurm array task uses one GPU; the default concurrency limit is four.

## Experiment

- A fixed, centered, single fluorescent bead, radius **1 µm**, in a 6.5 µm field.
  Every acquired frame is **64 × 64 pixels**; the pupil grid remains 32 × 32.
- The 17 fitted Zernike modes and three-loop diversity schedule match the
  existing RMS/SNR and bead-count sweeps. Gaussian modal draws use seed
  `[0, trial]`; each direction is reused across every RMS, with the existing
  annular-pupil `RMSAberration` normalization. Trial indices are zero-based.
- Initial RMS levels, in waves:
  `0.05 0.10 0.15 0.20 0.225 0.25 0.26 0.27 0.28 0.29 0.30 0.31 0.32 0.33 0.34 0.35 0.375 0.40 0.45 0.50`.
- **Combined expected RMS SNR = 10** on each case's initial aberrated,
  zero-diversity z=0 reference image. Signal Poisson and independent, zero-mean
  Gaussian noise each contribute **half the expected error power**. The model
  is `Poisson(exposure * image) / exposure + Normal(0, sigma)`.
  Calibration satisfies
  `SNR² = sum(image²) / (sum(image)/exposure + N_pixels * sigma²)`.
  Exposure and sigma are calibrated separately for each initial condition, then
  held fixed through its three loops. Probe stacks and corrected images can
  consequently have different SNRs; expected and realized values are saved.
  There is one noise realization per condition, with separate reproducible
  Poisson/Gaussian streams. Negative observations are retained.
- Four acquisition focal planes over −2 to +2 µm; 41 truth-object planes and
  21 fitted object planes over the same interval; three-photon vector optics.
- Three normalized probe modes at −0.10, 0, +0.10 waves per loop:
  `[-2,2], [0,4], [2,2]`, then `[-3,3], [0,4], [3,3]`, then
  `[-4,4], [0,4], [4,4]`. Each loop acquires 36 fresh frames; cumulative fits use
  **36 / 72 / 108 frames**, preserving past noise and SLM phase offsets.
- Newton optimization, `rho=1e46`, up to 50 steps per loop, 100 random starts
  plus zero, initialization seed 0 and RMS range [0.005, 0.50]. CUDA uses the
  existing QR SVD and guarded linear screening with candidate batches of eight.
  Applied corrections use gain 1 and modal clipping at ±0.10 waves.

The trial axis represents different aberration directions. It does not provide
independent noise replicates of each direction. Paired traces and between-trial
sample SD describe this baseline, rather than determining a precise universal
failure threshold.

## Transfer with Git and submit

The experiment is on the Git branch `codex/rms-cliff-sweep`. After that branch
has been pushed to `origin`, log into Bouchet yourself and fetch it from the
existing repository. Create a separate checkout so the job uses a fixed source
revision alongside your existing work:

```bash
cd /nfs/roberts/project/pi_cr876/mcw77/fall_2026_lab
git fetch origin
git worktree add --detach ../rms_cliff_baseline origin/codex/rms-cliff-sweep
cd ../rms_cliff_baseline
mkdir -p logs
sbatch systematic_testing/rms_cliff_sweep/bouchet_rms_cliff.sbatch
```

Use a new directory name if `../rms_cliff_baseline` already exists. Git transfers
the experiment source, tests and analysis scripts; generated numerical results
are retrieved separately below. Each job records the Git revision and any
tracked working-tree changes in its provenance directory. Keep this checkout at
the submitted revision until the whole array has finished.

To allow **eight simultaneous GPU tasks**, replace that last command with:

```bash
sbatch --array=0-399%8 systematic_testing/rms_cliff_sweep/bouchet_rms_cliff.sbatch
```

Slurm schedules tasks as GPUs and account limits allow. Each task has a
two-hour limit and 64 GB host memory. Task zero also runs the small CPU/CUDA
workflow checks; every task requires a working CUDA device. Change the partition,
time limit or concurrency on the `sbatch` command line as needed. The batch
defaults use the existing `pi_cr876` account and `phase_diversity_env` environment.

The case mapping is `case_index = rms_index * 20 + trial`. Logs include the
master array job ID and task index. Monitor the array with:

```bash
squeue -j ARRAY_JOB_ID
tail -f logs/pd-rms-cliff-ARRAY_JOB_ID_200.out
```

Results are under `results/bouchet/rms_cliff_sweep/ARRAY_JOB_ID/case_0000/` etc.
Each case saves `run.json`, scene/calibration, three fresh `acquisition_*.npz`
stacks with their acquisition SLM states, candidate-screening records,
`history.npz` and `results.npz`. Numerical results checkpoint atomically after
each loop. A completed optimizer call is saved even if it reports nonconvergence;
the convergence flag is a separate diagnostic from correction accuracy.
Existing case directories are never overwritten. Failed or timed-out cases
retain completed loops, while unfinished entries remain missing.

## Download and analyze

From your Mac, retrieve the result root (including all case subdirectories):

```bash
mkdir -p results/bouchet/rms_cliff_sweep/ARRAY_JOB_ID
rsync -avP \
  mcw77@transfer-bouchet.ycrc.yale.edu:/nfs/roberts/project/pi_cr876/mcw77/rms_cliff_baseline/results/bouchet/rms_cliff_sweep/ARRAY_JOB_ID/ \
  results/bouchet/rms_cliff_sweep/ARRAY_JOB_ID/
conda activate phase_diversity_env
python -m systematic_testing.rms_cliff_sweep.analyze \
  results/bouchet/rms_cliff_sweep/ARRAY_JOB_ID
```

The analysis writes these to the result root's `analysis/` directory:

- `residual_vs_initial_rms.{png,pdf}`: applied RMS after loop 3 versus initial
  RMS, all 20 paired traces and mean ± sample SD, the uncorrected diagonal and
  a 0.30-wave reference line.
- `estimation_vs_initial_rms.{png,pdf}`: loop-3 estimation error **before modal
  clipping**, to distinguish inference error from limited applied updates.
- `trials.csv`: every scheduled condition and loop, including missing entries,
  applied/estimation RMS, residual ratio, status and convergence flag.
- `summary.csv`: mean, sample SD, median, fraction at least the initial RMS,
  converged count, completed count and missing count for every RMS and loop.
- `sweep.npz`: arrays with axes `(initial_rms, trial, ao_loop)` and configuration.

Both RMS metrics use the established pupil geometry. The analysis includes
every completed finite result, including inaccurate and nonconverged fits.
By default it requires all cases; `--allow-partial` explicitly plots available
loops and reports missing task indices and per-RMS counts. `--all-loops` plots
loops 1–3 together. Plotting extends the shared RMS helper with its `bmh` style,
6 × 6-inch layout, external legend, LaTeX text and 300 DPI exports. `--no-tex`
is an explicit fallback. Analysis performs no optimization or image simulation.

## Local validation and input preview

```bash
python -m systematic_testing.rms_cliff_sweep.run_sweep --describe
python -m unittest discover -s tests -p test_rms_cliff_sweep.py -v
python -m systematic_testing.rms_cliff_sweep.run_sweep \
  --prepare-only --case-index 200 --output /tmp/rms_cliff_preview
```

The preview saves the exact first 36 noisy **64 × 64** frames at RMS 0.30,
trial 0, without CUDA startup or fitting. A matching preview is already saved
locally at `results/rms_cliff_sweep_setup/preview/case_0200/`.
Small-grid CPU tests exercise real three-loop fits, noise calibration and replay,
paired directions, historical measurements/SLM offsets, clipping and RMS
diagnostics, partial-data accounting and offline plot export. Set `REQUIRE_CUDA=1`
on a GPU node to additionally exercise the same checks with CUDA reconstruction.

Custom `--rms-values` or `--trials` change the required array bounds; use
`run_sweep --describe` with the same arguments, then override `sbatch --array`.
Keep a separate output root for each configuration.
