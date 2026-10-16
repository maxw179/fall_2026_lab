# Background-subtracted many-bead run

134 PD images, all nine Z planes, 64x64 block means, X mirror, same ten artifact exclusions. Each image has one median estimated from its outer 10-pixel border and subtracted from all pixels. Signed values retained. Original acquisitions and prior solves are unchanged. Per-image offsets are saved in input.npz and preparation.json.

All 17 modes free; 1,000 screened random starts; CUDA QR with fallback, screening/PSF/derivative batches 8, frequency batch 4096, 21 object planes spanning 4 microns, rho 1e46, max 50 Newton iterations.

From the repository root:

```bash
conda activate phase_diversity_env
python experimental_10_15/multiple_beads_64x64_flipx_background_subtracted_20261015/submit_jobs.py --submit
```

Watch, download, and render:

```bash
python experimental_10_15/multiple_beads_64x64_flipx_background_subtracted_20261015/collect_results.py
open experimental_10_15/multiple_beads_64x64_flipx_background_subtracted_20261015/replays.html
```
