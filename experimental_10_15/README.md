See [WORKFLOW.md](WORKFLOW.md) for the concise acquire → compare → solve → correct checklist.

Run this workflow on your local computer from the repository root, in
`phase_diversity_env`, after copying a MATLAB acquisition folder. The folder must
contain `metadata.mat` and `acquisition.mat`; TIFFs are previews, not fitting inputs.
The notebook `bead_phase_diversity.ipynb` provides the same comparison interactively.
Current acquisitions carry `cfg.analysis` and `cfg.solver` settings from MATLAB;
Python commands and the notebook use these by default. Only paths and the SSH
login alias need to be supplied on the laptop. CLI flags below are optional overrides.
The notebook only requires editing `run_dir`.

Compare every acquired image with a simulated bead:

```sh
conda activate phase_diversity_env
python -m experimental_10_15.bead_pd compare /path/to/copied/run --output /path/to/analysis
```

This reads optics and all total SLM commands from metadata and simulates a
uniform fluorescent sphere of the recorded **diameter** (1 µm by default) using
the existing vector 3P model. Figures in `analysis/figures/` show Z rows with
measured and simulated columns, shared intensity normalization and polar
commanded-phase insets. They preserve the acquired-plane plotting style,
6 × 6-inch size, LaTeX and 300 DPI PNG/PDF exports. Comparisons are paginated at
four Z rows per page, one phase-command slot per figure. `comparison.npz` stores
raw and scaled simulations, measured data and per-image RMSE;
`comparison.json` records the global brightness scale and comparison settings.
The notebook displays the exported figures side by side without uploading data.

Simulation intensities are in arbitrary optical units. One nonnegative global
least-squares gain maps the whole modeled stack to measured brightness; individual
planes are never separately normalized. The forward simulation uses the recorded
command, not an unknown native microscope phase. Differences can therefore
reflect native aberration, bead alignment, optical calibration or experimental
noise. The polar inset is the commanded phase, not a measured wavefront.

Relevant options:

```sh
python -m experimental_10_15.bead_pd compare /path/to/copied/run --output /path/to/analysis \
  --size 128 --pupil-size 64 --bead-planes 21 \
  --bead-center-um 0 0 0 --background 0 --z-sign 1
```

- `--size`: block-average the measured image to this size (default 128). It must
  divide the acquired square image size. Set 512 to preserve a 512-pixel scan.
  The simulator samples the PSF at the resulting pixel centers; compare finer
  sizes to check the approximation introduced by coarse sampling.
- `--pupil-size`: propagation pupil grid, default 64. Check convergence with a
  larger value when quantitatively validating the optics.
- `--bead-planes`: odd sample-plane count spanning the bead diameter, default 21.
  Sample-plane sums follow the existing forward model; one global gain absorbs
  unknown fluorophore density and axial discretization scale.
- `--bead-center-um X Y Z`: modeled bead center relative to field/focus center.
  For current runs, an empty MATLAB `cfg.analysis.beadCenterUm` estimates the
  lateral bead center from the brightest connected object in the closest-to-focus
  reference image after removing a border pedestal. Other objects are not included
  in the single-bead simulation. An explicit center overrides that estimate.
  Images are not translated or mirrored; stage-reference Z=0 is preserved.
- `--background`: subtract a known scalar PMT pedestal, default 0. Signed
  intensities are retained. This same preprocessing is used for reconstruction.
- `--z-sign`: +1 or −1 maps the stage Z readout to the model's focal coordinate.
  The model uses PSF defocus `sample_z - focal_z`; calibrate this sign and the
  relay's phase handedness on the instrument. Z metadata are manually confirmed
  requested positions, not stage measurements.
- `--no-tex`: explicitly disable LaTeX for a computer without a LaTeX installation.

MATLAB v7 imports work with SciPy. Earlier acquisition versions wrote v7.3,
which requires `h5py` (now listed in `environment.yml`). If unavailable, you can
re-save an existing small acquisition in MATLAB without altering its contents:

```matlab
data = load('acquisition.mat');
save('acquisition.mat', '-struct', 'data', '-v7');
```

Prepare fitting input without running simulations:

```sh
python -m experimental_10_15.bead_pd prepare /path/to/copied/run --output /path/to/analysis
```

Both `prepare` and `compare` create `input.npz` and `preparation.json`. Only
acquired PD images enter the fit; references and missing/failed images are
excluded. Repeated aberration-only frames are retained as independent averaged
measurements. Images are mapped from MATLAB `[row,column,image]` to model
`[image,x(column),y(row)]`. Pixel spacing is physical FOV divided by the image
size. The known diversity is `baseline + diversity`; the imposed aberration is
not supplied to the solver or its initialization. It is stored separately to
assess the result afterward. Thus the estimated unknown is the imposed phase
plus any residual native microscope phase and the cumulative correction already
applied. The cumulative correction stays out of known diversities so that later
loops estimate the remaining phase rather than repeatedly estimating the original
aberration. The solver fits an unconstrained 3D object jointly with aberration using
the project's existing reduced-loss optimizer, rather than assuming a known bead.

Submit, watch and fetch explicitly:

```sh
python -m experimental_10_15.submit_bead_pd submit /path/to/copied/run \
  --output /path/to/submission --host mcw77@YOUR_BOUCHET_LOGIN_ALIAS \
  --rho 1e46 --random-starts 100

python -m experimental_10_15.submit_bead_pd watch /path/to/submission/job_JOBID.json
python -m experimental_10_15.submit_bead_pd fetch /path/to/submission/job_JOBID.json \
  --output /path/to/fetched_solve
```

For one command that submits, watches and then fetches:

```sh
python -m experimental_10_15.submit_bead_pd submit /path/to/copied/run \
  --output /path/to/submission --host mcw77@YOUR_BOUCHET_LOGIN_ALIAS \
  --fetch-output /path/to/fetched_solve
```

No remote commands run unless you explicitly invoke submit/watch/fetch. Use an
SSH login host or configured alias that can execute `sbatch`; no host is assumed.
The remote run root defaults to
`/nfs/roberts/project/pi_cr876/mcw77/fall_2026_lab/results/bouchet/bead_experiments`;
override it with `--remote-root` if needed. SSH/SCP use your local authentication
setup. The commands inherit the existing Bouchet account `pi_cr876`, GPU
partitions, miniconda module and `phase_diversity_env`; adjust the supplied
`bouchet_bead_pd.sbatch` for a different account/environment.

Each submission uploads a unique ZIP with the original MAT acquisition files,
prepared solver input, optical metadata, a snapshot of the current local Python
sources, and SHA-256 hashes. It does not update the remote repository. The remote
GPU job uses that exact snapshot and requires working CuPy/CUDA in its environment.
Use `bundle` in place of `submit` to create/review the ZIP without connecting.
A submission receipt records the job ID, host and remote directory for reconnecting.
If interrupted while watching, rerun `watch`; it does not submit another job.
`watch --once` prints a single status update. Watching reports queue state,
random-start screening progress, optimizer stages and coefficient/loss updates.
Ctrl-C stops local watching and leaves the remote job running.

Solver settings can be passed to `submit` or `bundle`:
`--rho` (default 1e46 in the raw PSF scale), `--sample-planes` (21),
`--sample-extent-um` (4, centered about focus), `--max-iterations` (50), and
`--random-starts` (100, seeded initialization screening). Regularization must be
assessed for the optical grid and experiment; 1e46 is a starting value from the
simulation workflow. All preparation options (`--size`, `--background`,
`--pupil-size`, `--z-sign`) also apply to submission: use the same settings as
comparison if you want to compare equivalent inputs.

Optional `--min-snr 5` on `prepare`, `compare`, `bundle` or `submit` screens each
PD image after block binning at the solver resolution. SNR is the brightest
interior 3 × 3 patch mean above the median edge background, divided by robust
edge noise (MAD / 0.67449 from the outer 15% band). It uses the saved average
directly, with no extra square-root-of-exposure factor. This spatial background
contrast estimate does not measure temporal or bead shot noise. Thresholding
does not change retained pixels or normalize their brightness. The input saves
retained SNR values; `preparation.json` and `snr_qc.csv` audit every candidate and
rejection, and the submission ZIP includes the CSV and its hash. Reference frames
never enter the fit. Comparison plots continue to show all acquired images.

To inspect exclusions, near-cutoff examples and prepare filtered inputs locally
before any upload, use `python -m experimental_10_15.quality RUN1 RUN2 ...
--output NEW_REVIEW_FOLDER --min-snr 5 --z-um -1 0 1`. Open the generated
`index.html`; no remote connection or alignment preprocessing runs.

New experimental submissions automatically use the GPU demo's optimized execution:
QR-reduced CUDA SVD (with direct-SVD fallback for numerical conditioning), batched
loss-only random-start screening with augmented QR fallback, screening batches of
8, and PSF/derivative batches of 8 with 4096 frequencies per solve batch. The
persistent workspace retains fields, SVD factors and accepted fits, shares exactly
repeated PSFs, and uses the spatial Hessian contraction. Progress-file writes are
limited to once a second within a stage; each evaluated optimizer state is always
saved. Double precision, the exact Newton Hessian, all fitted modes, image/model
grids, initialization candidate count and convergence criteria are preserved.
Existing submitted jobs keep their original source snapshots.

Execution tuning can be overridden on `submit`, `bundle` or `solve` using
`--svd-method qr|direct`, `--screening-batch-size`, `--psf-batch-size`,
`--derivative-batch-size` and `--frequency-batch-size`. Larger batches use more GPU
memory; defaults match the established demo rather than assuming a particular GPU.
These are Python execution settings, separate from MATLAB's physical/model settings.
Submissions save them in `bundle.json`; results save the effective backend/options
and QR/direct factorization counts in `runtime.json` and `result.json`. The existing
`gpu.txt` records the allocated GPU. `optimizer.json` history now includes elapsed
seconds at each evaluated state, including initialization/setup time, to diagnose
slow iterations versus slow convergence. Local validation checks correctness;
compare actual GPU elapsed times on Bouchet.

Fetched `results/` contains:

- `correction.mat`: MATLAB-readable complete cumulative correction for the next
  loop, equal to previous cumulative correction minus the estimated residual;
  also records parent run/loop, imposed aberration, baseline, optics, focus reference
  and optimizer convergence. Set MATLAB `cfg.correctionFile` to this file.
- `solution.npz`: mode order, estimated wave coefficients, negative correction
  coefficients, reconstructed object, reconstruction/focal Z positions and the
  imposed coefficients for validation.
- `coefficients.csv`: readable per-mode imposed/estimated/correction coefficients
  in waves, plus error against imposed phase + current cumulative correction.
- `optimizer.json`: full convergence diagnostics, screened candidates and timed history.
- `runtime.json`: effective execution settings and QR/direct SVD counts.
- `result.json`: settings, elapsed time and optimizer convergence status.
- `progress.json` and `latest_estimate.npz`: current status and last evaluated
  estimate; useful if a job fails before completing.
- `environment.txt` and `gpu.txt`: remote software/GPU records.

Fetch also copies the source/data hash manifest and Slurm stdout/stderr. A
completed job can have `optimizer_success=false` (e.g. an iteration limit); check
`result.json` before using the coefficients. The negative residual coefficients are an incremental correction; `correction.mat`
contains the cumulative correction. Copy that file to the microscope and select
it in MATLAB `cfg.correctionFile` to apply it on the next run. The original
imposed aberration remains unchanged; do not add exported coefficients by hand. Fetch destinations
must be new or empty so older results cannot be overwritten accidentally.

For a local CPU smoke fit of prepared data:

```sh
python -m experimental_10_15.bead_pd solve /path/to/analysis/input.npz \
  --output /path/to/local_solve --backend cpu --random-starts 0 --max-iterations 2
```

The experimental workflow tests use synthetic acquisitions, a real tiny CPU fit,
and mocked SSH/SCP/Slurm calls. They do not establish instrument agreement or
validate the remote environment. The legacy v7.3 test is skipped if h5py is absent.

For runs using MATLAB `cfg.systemAOFigure`, the applied raw SLM map is retained
in metadata, prepared input and `correction.mat`. Every reference and PD image
has that fixed calibration applied inside the pupil. Forward comparison assumes
it cancels native system aberration; the fit estimates residual native + system
AO + imposed + cumulative phase. The map is not added again as a known diversity.
MATLAB checks the map matches when loading a correction for another loop.

SSH/SCP connections now share a private OpenSSH control socket. Authenticate once;
uploads, status polling and downloads reuse that connection. It remains available
for ten minutes after the last command, so standalone watch/fetch commands can
reuse it too. Authentication may be required again if the connection drops or
expires. Ctrl-C while watching does not cancel the Slurm job: resume with its
existing `job_JOBID.json` receipt rather than running submit again.

Replay a downloaded solve with the established case-study HTML viewer:

```sh
python -m experimental_10_15.replay /path/to/fetched/results \
  --input /path/to/submission/input.npz \
  --output /path/to/fetched/figures/correction_replay.html
```

The self-contained HTML reuses the case-study layout and playback controls.
System AO is assumed already applied and native aberration corrected. The three
polar panels show the metadata imposed wavefront, the current estimated imposed
wavefront, and estimation error. The bottom panels show a fixed actual acquired
zero-diversity image, optimizer loss, and RMS estimation error relative to metadata.
There is no proposed-correction simulation. Later-loop estimates have the known
cumulative correction subtracted before comparison to the original imposed phase.
Iteration times were not recorded, so the horizontal axis is saved iteration,
not elapsed seconds. `--focus-um` selects the nearest acquired zero-diversity
frame (default 0); `--stride` reduces frame count. The matching input must retain
the same plane exclusions as the solve.

The watcher retries failed status reads up to five consecutive times, prints SSH's
stderr, and then stops locally with the saved job-file path. It never cancels or
resubmits a job on a polling failure. Resume with `watch job_JOBID.json`, then
fetch and render the replay. `watch --max-failures N` changes the retry limit.

For reference-driven lateral drift correction before fitting, see
[ALIGNMENT.md](ALIGNMENT.md) and run `python -m experimental_10_15.align_stack --help`.
It writes a separate acquisition folder and uses only before/after reference
images to estimate motion.
# Saved estimate animations

Run `python -m experimental_10_15.animate_estimates` to export the completed
rho=1e49 many-bead result as polar aberration and 3D voxel object animations.
This reuses `utils.plotting.animate_reconstruction`, `zernike_plot` and
`plot_sample_3D`; each object is refitted at its saved phase with the recorded
regularization. No optimization or cluster connection is performed.
Outputs are two GIFs and `animations.html` under the case's
`solve/figures/animations/`. Optional `--case`, `--output`, `--stride`,
`--percentile`, `--dpi` and `--interval` control export. The fixed voxel
threshold comes from the final object's selected percentile; colors are shared
across frames. Acquisitions and generated figures/results are kept locally,
outside Git tracking.
