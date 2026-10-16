Run from the repository root in `phase_diversity_env`.

Prepared locally; the initial submission attempt was blocked by SSH
`Permission denied (publickey)`. No jobs were submitted in that attempt.

Submit all three resolutions (existing receipts prevent duplicate submission):

```bash
python experimental_10_15/multiple_beads_native_64_32_20261015/submit_jobs.py --submit
```

Fetch finished jobs and update the separate HTML page:

```bash
python experimental_10_15/multiple_beads_native_64_32_20261015/collect_results.py --ready-only
```

Omit `--ready-only` to watch until completion, download and render all three.
Open `replays.html` in this folder. All 144 fitting images and all nine Z planes
are included, with no SNR cut. Native resolution is 128×128. Binning uses means,
preserving FOV. These fits reconstruct the object freely and fit all 17 modes.
