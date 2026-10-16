# Six original acquisitions, corrected 64x64 pipeline

One independent job each: no aberration (32 PD images), System AO (40), spherical 0.15w (40), random 0.15w (40), random aligned 0.15w (60), random aligned 0.25w (60). Total 272 PD images.

Use original acquisition.mat signed averages and metadata. Block mean to 64x64, mirror X, subtract one outer 10-pixel border median per image. No SNR or Z-range clipping. Preserve only the explicitly documented failed plane 05 exclusion in no-aberration. No software registration, per-image intensity scaling, or exclusions inherited from the many-bead dataset. Source paths and retained image indices are recorded in summary.json and each submission/preparation.json.

Standard rho=1e46. All 17 modes free, 1,000 screened random candidates, 21 signed object planes over 4 microns, max 50 Newton iterations; fast CUDA QR with fallback and batch settings 8/8/8/4096. These six jobs have no dependencies on each other or the separate many-bead regularization batch.

From repository root on the user's Mac:

```bash
conda activate phase_diversity_env
python experimental_10_15/all_stacks_64x64_flipx_background_20261015/submit_jobs.py --submit
```

Wait, download, build, and open the combined replay page:

```bash
python experimental_10_15/all_stacks_64x64_flipx_background_20261015/collect_results.py && \
open experimental_10_15/all_stacks_64x64_flipx_background_20261015/replays.html
```

To collect only completed jobs without waiting, use --ready-only; to render downloaded results without SSH use --local-only. Saved receipts prevent duplicate submission. Agent prepares locally and must not execute cluster operations.
