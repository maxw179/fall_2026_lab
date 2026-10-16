# Experimental phase diversity handoff — 2026-10-15

## Working agreement

- **Never connect to, upload to, or submit to Bouchet yourself.** Prepare locally, then give the user both command blocks: (1) submit and (2) wait → download → build replay → open visualization. This also applies when the user says “run on Bouchet.”
- Keep replies concise; reuse the existing preparation, submission, collection and plotting code.
- Preserve signed MATLAB averages and raw acquisitions. TIFF exports clip negative values; use `acquisition.mat` + `metadata.mat` for inference.
- Software registration in `align_stack.py` is on hold until explicitly requested. The separately authorized global X mirror remains part of the current run.
- Follow root `AGENTS.md` plotting conventions. Static diagnostics are PNG/PDF; optimizer replays use the existing HTML viewer.

## Workflow and conventions

1. Microscope: configure optics, manual imposed Zernike coefficients, Z targets and averaging in `matlab_oct15/bead_phase_diversity_acquire.m` (relative to this directory). Align/focus the bead, run MATLAB, manually move to each prompted Z target.
2. Copy the whole acquisition folder to the laptop. Settings come from MATLAB metadata. Inspect measured/simulated images before solving.
3. Prepare a new local batch; the user submits and collects it using commands below. Never overwrite an earlier solve.
4. Inspect replay/coefficient error; a future physical correction loop can import the exported correction into MATLAB.

Optics: 1300 nm, 3P vector model, NA 1.05, water index 1.333, FOV 6.5 µm. Zernike coefficients are **waves**, normalized real modes `[m,n]`; SLM raw phase has 256 units per cycle. System AO is applied to all images. Treat its cancellation of native aberration as an assumption, not measured truth.

Python arrays: `[image,x(column),y(row)]`. Binning is a **block mean**, preserving FOV and averaged-intensity units. Current camera convention: mirror model X, without transposing or changing phase commands.

## Current many-bead case

Source, relative to this directory:

`3d_many_fluorobeads_random_aligned_0p15w/tp76e42de4_ca3a_4172_9562_d6cf4b94918c`

- 162 averaged images; 144 PD + 18 references; 25 exposures per saved image.
- Nine Z targets: −2 to +2 µm in 0.5 µm steps; ±0.1-wave probes.
- Exclude original **1-based image numbers** `2,4,20,24,44,57,112,116,130,135`, leaving **134 PD images**. No SNR or Z clipping. Do not apply the older single-bead plane-05 exclusion here.
- All 17 aberration modes free, including coma/trefoil; free signed object with 21 planes over 4 µm; rho `1e46`; max 50 Newton iterations.
- Fast CUDA QR with fallback; screening/PSF/derivative batches 8, frequency batch 4096. **1,000 random starts means screening 1,000 candidates, then refining the selected candidate**, not 1,000 full optimizations.

## Main finding: per-image backgrounds explain much of the fit gap

Original fixed-wavefront loss check refitted the object at truth, zero and the saved estimate. The incorrect experimental estimate had 10.05% lower loss than truth; the noisy simulation estimate improved only 0.057%.

Subsequent read-only diagnostics found large imagewise background offsets. Subtract one median from each final 64×64 image, using its outer **10-pixel border** (central 44×44 excluded). Preserve negative values; no rescaling, clipping, smoothing or registration.

| At the known wavefront, object refitted | Residual RMS |
|---|---:|
| Original experiment | 16.156 |
| Experiment, per-image background removed | 6.315 |
| Noisy simulation, background removed | 6.339 |
| Simulation with experimental offsets added | 16.039 |

After subtraction, the old experimental estimate has **33.27% higher loss than truth**. Borders 5/10/16 pixels and corner-only estimates agree. Reference-only background interpolation fails: estimate each image separately. Corrected scaling profiles minimize at α=1 on the sampled grid. This supports background offsets as a major cause; **successful recovery still requires a new corrected-data optimization**.

Other checks: repeats/brightness/centroids, noise correlations, Z sign/scale/offset, FOV/aspect ratio, NA, pupil rotations/reflections, beam waist, object sampling, regularization, and historical zero controls. None produced a comparable improvement. Software Zernike conventions match. Actual stage positions, SLM phase response and residual system wavefront are not independently measured. The System AO control shares the current AO map; the “no aberration” control uses a different map.

Evidence: [full report](image_model_diagnostics/REPORT.md), [16-page figures](image_model_diagnostics/diagnostics.pdf), tables/residual arrays in the same directory. Reproduce locally with `python -m experimental_10_15.image_model_diagnostics`.

## Current background-subtracted rerun

Batch: `multiple_beads_64x64_flipx_background_subtracted_20261015`.
Completed by the user as job **29708308**; results and replay are downloaded in the batch. Error against imposed truth is **0.099212 waves RMS**, versus **0.134368** for the previous X-mirrored solve. The initial agent submission attempt failed before upload; the user submitted successfully afterward.

**Commands from repository root on the user's Mac:**

```bash
conda activate phase_diversity_env
python experimental_10_15/multiple_beads_64x64_flipx_background_subtracted_20261015/submit_jobs.py --submit
```

```bash
python experimental_10_15/multiple_beads_64x64_flipx_background_subtracted_20261015/collect_results.py && \
open experimental_10_15/multiple_beads_64x64_flipx_background_subtracted_20261015/replays.html
```

Collector waits, downloads, and builds replays; `open` runs only on success. Submission alone does not open a visualization. Host used by the scripts: `mcw77@bouchet.ycrc.yale.edu`.

## Follow-up: old/new/truth comparison

`background_solution_comparison/REPORT.md` and `comparison.pdf` (17 pages), reproduced with `python -m experimental_10_15.compare_background_solutions`. Local diagnostics only; no new cluster run prepared.

- On identical background-subtracted PD data, old / truth loss = **1.3327**, new / truth = **0.8278**. The objective still prefers the new incorrect phase. Both saved optimizer losses replay correctly.
- Alternating object and positive image gains, with mean(gain²)=1 and fixed phase, reduces the new estimate's advantage over truth from **17.2% to 1.9%** (up to 12 updates, not guaranteed global joint minima). At truth, experimental gain p10–p90 ≈ **0.65–1.40**; matched noisy simulation ≈ **0.97–1.03**. Gain freedom can hide phase errors; these are not measured laser-power corrections.
- Fixed-object command-shared gains improve truth RMS 6.315 → **5.814**, versus only 6.297 for Z-shared gains. Probe command and within-plane acquisition order are confounded. Relative brightness/throughput/model mismatch is the next strong lead, not proven hardware causation.
- References test two distinct hypotheses. Modeling estimate-minus-truth as a common system residual strongly worsens joint PD+reference loss. Holding references fixed/unaberrated instead still favors the new PD phase in experiment (new/truth **0.6474**), while noisy simulation favors truth (**1.3894**). Do not conflate these hypotheses or claim references alone fix recovery.
- Modes **(-3,3), (-3,5)** account for **60.3%** of remaining squared coefficient error (coefficient-space fraction).
- Suggested next test: constrained gain-aware phase retrieval, first command-shared versus imagewise, with global scale fixed and a matched simulation control. Not yet implemented in the production optimizer.

## Latest priority: regularization before intensity factors

User prefers investigating regularization first. `background_regularization_sweep/` contains a 25-point half-decade sweep, 1e40–1e52, on corrected PD data at fixed new estimate versus truth with the object refit each time. Reproduce locally with `python -m experimental_10_15.sweep_background_regularization`; exact SVD reuse is checked against canonical solves.

New/truth loss ratio: 0.8278 at current rho 1e46, 0.8291 at 1e48, 0.8528 at 1e50, 0.9698 at 1e51, **1.0484 at 3.16e51**, 1.0687 at 1e52. Strong regularization reverses the ranking between these fixed phases. At 3.16e51 the penalty drives the reversal; truth/new residual RMS rise to 7.125/7.042, with fitted image norms still 68%/65% of measured. Fresh optimizations were subsequently prepared as described below. This sweep does not establish the new optimum or independently validate a rho chosen using truth. See `regularization_sweep.pdf`, `loss_ratio.png`, `REPORT.md` and `sweep.csv`.

Prepared next: `multiple_beads_64x64_regularization_20261015/` has three independent rho cases **1e51, 3.16227766e51, and 1e52**, otherwise identical corrected inputs and fast settings. User runs its `submit_jobs.py --submit`, then `collect_results.py && open .../replays.html`. No dependencies; may execute concurrently as GPUs become available. Check local receipts/results for subsequent status; agent did not submit. For downstream diagnostics, read effective rho from downloaded `result.json`, not the acquisition-default `input.npz` solver JSON; `image_model_diagnostics.load_case` now handles this override.

Before submission, user requested a naive-zero check at all 25 rho values. Added to the same sweep/CSV/PDF: **truth/naive loss = 0.5043 at 1e46, 0.6764 at 1e51, 0.9272 at 3.16e51, 1.3154 at 1e52**. Known probes remain applied and the object is refit for naive zero (distinct from zero object). At **3.16e51 truth beats both new estimate and zero**; at **1e52 zero beats both**, indicating overregularization bias. The three-job batch was left unchanged; the high-rho case is useful as an overregularization control. `truth_vs_naive.png/pdf` shows truth/zero and new/zero; combined PDF now has five figures.

## Reuse these files

**Saved-estimate animations (2026-10-16):** `python -m experimental_10_15.animate_estimates` exports the rho=1e49 case by default. All 23 saved iterates use the established polar phase + 3D voxel object styles (`utils.plotting.animate_reconstruction`, `zernike_plot`, `plot_sample_3D`). Objects are refitted with effective rho and known probes; the last refit is checked against the saved solution. Outputs: `rho_1e49/solve/figures/animations/{aberration_estimate.gif,object_estimate.gif,animations.html}` under the regularization batch. Shared voxel colors and a fixed 98th-percentile final-object threshold; signed data preserved. CLI accepts other cases/stride/threshold percentile/dpi/interval. No optimization or Bouchet connection. Generated data, animations, replays, archives and logs are ignored by Git; notebooks are committed without outputs while local outputs are retained.

**End-of-day figures:** completed many-bead rho=1e49 solve has zero-estimate error **0.14921 → 0.07595 waves RMS** (22 saved Newton updates). Seven 300-DPI PNGs are in `multiple_beads_64x64_regularization_20261015/rho_1e49/solve/figures/presentation_pngs/`: final three-polar-panel AO overview, RMS/loss convergence, signed coefficients, zero-diversity Z stack, representative probe fits, and residual vs Z. Reproduce locally with `python -m experimental_10_15.export_rho49_figures`; uses shared plotting helpers and an optional AO viewer figure-export callback. These show estimation and fitted predictions, not a physical post-correction acquisition. User requested PNGs only and to wrap up for the day.

**Other six acquisition reruns:** `all_stacks_64x64_flipx_background_20261015/` prepares the original no-aberration (32 PD), System AO (40), spherical 0.15w (40), random 0.15w (40), random aligned 0.15w (60) and random aligned 0.25w (60): **272 PD images** total. X mirror + per-image 10-pixel border subtraction, 64x64 means, no SNR or Z-range clipping, no software registration or inherited many-bead image exclusions. Retain only documented failed plane 05 in no-aberration. One independent standard **rho=1e46** run each, all 17 modes free, 1,000 screened starts, same fast QR settings. These are separate from the five-rho many-bead batch. User requested ignoring earlier SNR/Z-clipped attempts; do not use those derived inputs. User runs this batch's `submit_jobs.py --submit`, then `collect_results.py && open .../replays.html`; agent does not connect to Bouchet. Prepared locally, not submitted by agent; inspect receipts for subsequent status.

**Latest run selection:** user replaced the proposed three-rho batch with **1e47, 1e48, 1e49, 1e50, 1e51**. The same `multiple_beads_64x64_regularization_20261015/submit_jobs.py` now prepares/submits only these five independent cases; its collector builds their combined replay page. Old unsubmitted 3.16e51 and 1e52 preparation folders are inactive. No network operations performed by the agent. Historical three-value notes above explain the investigation, not the current batch selection.

- `bead_pd.py`, `bead_pd_io.py`: prepare/solve; optional `--background-border-pixels 10`; offsets retained in `input.npz` and `preparation.json`.
- `submit_bead_pd.py`: snapshot/bundle/submit/watch/fetch; shared SSH connection. User runs remote operations.
- Batch `submit_jobs.py` + `collect_results.py`: existing lifecycle/replay skeleton.
- `reference_bead_simulation.py`: reference-only finite 1 µm sphere position/brightness fits; `reference_bead_orientation_scan/flip_x/` holds the chosen comparison.
- `empirical_noise.py`: effective Poisson/Gaussian noise from repeat pairs; gain ≈4.09, median Gaussian σ≈3.44 at final 64×64 resolution. Do not divide again by bin area or exposure count.
- `multiple_beads_64x64_simulated_noisy_matched_20261015/`: frozen noise seed 20261015, matched exclusions; all 162 experimental/noisy simulation comparisons in `experimental_vs_noisy/all_images.pdf`.
- Local interpreter: `/opt/miniconda3/envs/phase_diversity_env/bin/python`; plotting tools may need `/Library/TeX/texbin` on PATH. Root `MEMORY.md` contains older simulation history, not current experimental status.
