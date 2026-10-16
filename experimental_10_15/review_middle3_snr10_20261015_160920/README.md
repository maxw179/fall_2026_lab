# Three-plane reruns with SNR ≥ 10

All six results are downloaded. Open **`replays.html`** for the local results
site: switch between embedded optimizer replays, compare wavefront RMS error
against metadata, see retained images by Z, and download corrections/coefficients.
`site_data.json` contains the displayed summary data. The established scientific
replay layout and plotting style are preserved. No external hosting is used.

`collect_results.py` now refreshes this site after collecting/rendering. To rebuild
only the page from local replay artifacts:

```sh
python experimental_10_15/review_middle3_snr10_20261015_160920/build_site.py
```

Open `index.html` for inclusion/exclusion examples, counts and per-image CSVs.
The six acquisitions retain 17, 23, 5, 10, 28 and 11 PD images respectively:
94 of the 168 candidates at Z = −1, 0, +1 µm. The SNR definition matches the
SNR ≥ 5 review. Only the cutoff changes. No software alignment is applied.
The SNR ≥ 15 review was abandoned without submitting any jobs.

After filtering, spherical has only five retained images. Random aligned 0.25w
has no retained +1 µm images. These fits implement the requested cutoff exactly;
the remaining image/mode counts are visible in the review.

`prepared_bundles.json` lists all six locally validated upload ZIPs. Each case's
`submission/` contains the exact fitting input, screening audit CSV, preparation
metadata, source snapshot, Slurm script and SHA-256 manifest. CUDA QR with
conditioning fallback, batched loss-only initialization and the persistent
workspace's caches remain enabled; model grids and fitting tolerances are retained.

From the repository root:

```sh
conda activate phase_diversity_env
python experimental_10_15/review_middle3_snr10_20261015_160920/submit_all.py
```

This verifies the inputs against the review, submits with the fast CUDA defaults,
and records job receipts in `jobs.json`. Re-running it skips saved job receipts.
It makes a fresh source snapshot at submission. Once submitted, watch, fetch and
render the replays with:

```sh
python experimental_10_15/review_middle3_snr10_20261015_160920/collect_results.py
```

Each output goes under `CASE/solve/figures/correction_replay.html`. Replays use
only retained measured frames; when zero-diversity frames are unavailable, the
displayed retained probe is explicitly labelled with its diversity.

Submitted jobs (SNR ≥ 10, middle three requested Z planes):

| Acquisition | Images | Job |
|---|---:|---:|
| no_aberration | 17 | 29675454 |
| system_AO | 23 | 29675459 |
| spherical_0p15w | 5 | 29675631 |
| random_0p15w | 10 | 29675668 |
| random_aligned_0p15w | 28 | 29675697 |
| random_aligned_0p25w | 11 | 29675698 |
