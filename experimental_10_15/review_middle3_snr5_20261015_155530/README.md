# Reviewed reruns: Z = −1, 0, +1 µm and SNR ≥ 5

Open `index.html` for the pre-submission counts, included/excluded examples,
and CSV scores. `summary.json` records the six source acquisitions and selections.
Only original signed acquisition MAT files are used; no software alignment runs.

There are 272 acquired PD images across the six acquisitions. The Z selection
removes 104; SNR screening removes 25 of the remaining 168, leaving 143 for six
independent fits. Reference frames are separate and never enter the fit.

SNR is the maximum interior 3 × 3 patch mean minus the median background,
divided by the background MAD / 0.6744897501960817. The background uses the outer
15% band of each frame at the solver resolution. Scores refer to the saved
average; they estimate spatial background contrast rather than temporal or bead
shot noise. Retained pixels, diversity commands and model settings are preserved.

From the repository root, after reviewing:

```sh
conda activate phase_diversity_env
python experimental_10_15/review_middle3_snr5_20261015_155530/submit_all.py
```

This verifies that each input exactly matches the review before submitting with
the optimized CUDA QR/batching/caching defaults. Already-saved job receipts are
reused, so rerunning this command skips existing jobs. Receipts and exact source
snapshots live under each case's `submission/`; `jobs.json` lists the job IDs.

Then watch, fetch, and generate all six HTML replays:

```sh
python experimental_10_15/review_middle3_snr5_20261015_155530/collect_results.py
```

Each replay goes under `CASE/solve/figures/correction_replay.html`. Submitted jobs
can execute concurrently regardless of the order in which they are watched.
The correction MAT and original optimizer/result diagnostics accompany each solve.
