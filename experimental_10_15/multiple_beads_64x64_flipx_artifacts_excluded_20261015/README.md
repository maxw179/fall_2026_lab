64×64 full-stack repeat with the original model X axis mirrored (no transpose),
excluding the same ten flagged images. 134 PD images, all nine Z planes, all
17 fit modes, 1,000 random starts and fast CUDA settings. Raw files are preserved.
No reference-derived bead model is imposed on this optimization; the object is free.

```bash
python experimental_10_15/multiple_beads_64x64_flipx_artifacts_excluded_20261015/submit_jobs.py --submit
python experimental_10_15/multiple_beads_64x64_flipx_artifacts_excluded_20261015/collect_results.py
```

Collection tracks, downloads and generates `replays.html`. Existing receipts
prevent duplicate submissions. Pixel orientation is recorded in the prepared
input, preparation report and solution outputs. Phase commands remain unchanged.
