# Full-stack binning comparisons: 64 × 64 and 32 × 32

The local results page is **`replays.html`**. It lists all 12 jobs and supports
filtering by resolution, switching between the established optimizer replays,
comparing RMS error against the imposed wavefront, and downloading corrections.
Cases awaiting download remain visible as placeholders; no results are invented.

To collect only jobs that have finished and refresh the page immediately:

```sh
conda activate phase_diversity_env
python experimental_10_15/binning_all_images_20261015_161419/collect_results.py --ready-only
```

Repeat that command as results arrive. It checks all jobs before downloading,
so an unfinished first job does not block completed jobs later in the list.
The page refreshes after each replay is created. Use `--local-only` to render
already downloaded results without SSH. The original command without flags
watches each job to completion, fetches, renders, and refreshes the site.

To refresh only the page from existing replay artifacts:

```sh
python experimental_10_15/binning_all_images_20261015_161419/build_site.py
```

Both this page and the SNR ≥ 10 page use the same shared builder,
`experimental_10_15/results_site.py`, preserving the scientific viewer layout.

Twelve independent GPU solves: all six acquisitions at both lateral image sizes.
No SNR threshold or Z-window selection is applied. The previously flagged bad
plane 05 in the no-aberration acquisition remains discarded; that run has four
valid Z planes, while the other five acquisitions use all five planes.

Each resolution uses 32, 40, 40, 40, 60 and 60 PD images respectively (272 images
per resolution). References never enter the fit. Binning uses 2 × 2 or 4 × 4
block means of the original signed averaged frames. Pixel spacing changes with
resolution; the 6.5 µm FOV and intensity units are preserved. Model pupil sampling,
17 fitted modes, 21 reconstruction planes, regularization, initialization count
and convergence settings come from the original metadata. No software alignment
is applied. All jobs use optimized CUDA QR, batched screening and cached fits.

These jobs have **no Slurm dependencies**, so they may run alongside the SNR ≥ 10
batch as resources and account limits permit. No existing job is cancelled.

From the repository root:

```sh
conda activate phase_diversity_env
python experimental_10_15/binning_all_images_20261015_161419/submit_all.py
```

The command verifies the frozen inputs and records receipts in `jobs.json`.
Existing receipts are reused when the command is rerun. Use `--prepare-only` to
create local ZIPs without an SSH connection. `prepared_bundles.json` lists those
ZIPs, and each case's `submission/bundle.json` records the source/data hashes
and fast execution settings. A fresh source snapshot is made at submission.

To watch, fetch and render completed optimizer histories as HTML:

```sh
python experimental_10_15/binning_all_images_20261015_161419/collect_results.py
```

Replays go under `CASE/solve/figures/correction_replay.html`.

Submitted without dependencies:

| Acquisition | 64 × 64 job | 32 × 32 job |
|---|---:|---:|
| no_aberration | 29676158 | 29676264 |
| system_AO | 29676249 | 29676267 |
| spherical_0p15w | 29676251 | 29676269 |
| random_0p15w | 29676254 | 29676271 |
| random_aligned_0p15w | 29676256 | 29676272 |
| random_aligned_0p25w | 29676261 | 29676281 |
