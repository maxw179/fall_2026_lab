64×64 repeat of the full nine-Z multiple-bead acquisition, excluding original
1-based images 002, 020, 057, 004, 112, 130, 024, 044, 116, 135. The solver
retains 134 PD images, all 17 free modes and all remaining coma/trefoil probes.
Fast CUDA settings and 1,000 random starts match the earlier full-stack run.
No SNR filtering, software alignment or Z clipping. Raw files remain unchanged.
See `excluded_images.csv` and `submission/preparation.json` for the exclusion audit
(the preparation file is under `multiple_beads_64x64/submission/`).

From the repository root in `phase_diversity_env`:

```bash
python experimental_10_15/multiple_beads_64x64_artifacts_excluded_20261015/submit_jobs.py --submit
python experimental_10_15/multiple_beads_64x64_artifacts_excluded_20261015/collect_results.py
```

Existing receipts prevent duplicate submissions. Collection watches, downloads
and renders `replays.html`. Add `--ready-only` to collect without waiting.
