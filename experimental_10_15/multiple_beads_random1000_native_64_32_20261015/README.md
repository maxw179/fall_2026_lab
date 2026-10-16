Run from the repository root in `phase_diversity_env`.

This independent batch repeats the multiple-bead native (128×128), 64×64 and
32×32 fits with **1,000 random initial candidates** instead of 100. Screening
selects an initialization for the Newton solve; it does not run 1,000 full solves.
All 144 PD images, nine Z planes, 17 free modes and fast CUDA settings are retained.
Original runs and their results are preserved.

```bash
python experimental_10_15/multiple_beads_random1000_native_64_32_20261015/submit_jobs.py --submit
python experimental_10_15/multiple_beads_random1000_native_64_32_20261015/collect_results.py
```

Existing job receipts prevent duplicate submission. The collection command watches,
downloads and generates `replays.html` in this folder. Use `--ready-only` to collect
finished jobs without waiting. The explicit `--random-starts 1000` solver override
is recorded in each submitted bundle manifest and final result; the embedded
original MATLAB solver settings remain unchanged.
