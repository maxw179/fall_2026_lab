# Parallel regularization comparison

Five independent optimizations: rho = 1e47, 1e48, 1e49, 1e50, 1e51. These replace the earlier proposed three-value batch. Jobs are queued without dependencies and may run concurrently as GPUs become available. Previously prepared 3.16e51 and 1e52 folders are inactive and will not be submitted or collected by this batch.

Same 134 background-subtracted 64x64 images as the completed rho=1e46 case, X mirror, all nine Z planes, ten artifact exclusions, all 17 modes free, no intensity factors. Fast CUDA QR with fallback, 1,000 screened starts, batch sizes 8, frequency batch 4096, rho-specific object fits, 21 object planes over 4 microns, max 50 Newton iterations. Each independent source/data snapshot records the actual rho in bundle.json solver_args and result.json. run_settings.json documents each intended case locally.

Run these on the user's Mac from the repository root (agent must not connect to Bouchet):

```bash
conda activate phase_diversity_env
python experimental_10_15/multiple_beads_64x64_regularization_20261015/submit_jobs.py --submit
```

Then wait for all five, download, build the combined replay page, and open:

```bash
python experimental_10_15/multiple_beads_64x64_regularization_20261015/collect_results.py && \
open experimental_10_15/multiple_beads_64x64_regularization_20261015/replays.html
```

Submission is idempotent per saved receipt. For completed jobs only without waiting, use collect_results.py --ready-only; for already downloaded results use --local-only. The existing rho=1e46 baseline has 0.099212 waves RMS error; new recovery is not established by the fixed-candidate rho sweep.
