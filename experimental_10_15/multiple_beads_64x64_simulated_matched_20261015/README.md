Matched simulated control for the X-mirrored 64² experimental run. Uses the
existing `reference_bead_orientation_scan/flip_x/comparison.npz` predictions:
three reference-fitted 1-µm spheres, fixed brightnesses/background, recorded
phases, native128 rendering then block means64. No noise/artifacts are added.

Predictions are already in optical coordinates: NO second flip is applied.
The same ten original image numbers are excluded, leaving the identical 134
source indices, nine Z planes and diversities. All 17 modes and the object remain
free; no true coefficients or bead geometry initialize the optimizer. Uses
1,000 random starts and identical fast settings. Original acquisition files and
experimental optimization are preserved. Synthetic MAT files are explicitly
marked as simulation and bundled with reference-fit/forward provenance.

```bash
python experimental_10_15/multiple_beads_64x64_simulated_matched_20261015/submit_jobs.py --submit
python experimental_10_15/multiple_beads_64x64_simulated_matched_20261015/collect_results.py
```

Collection watches, downloads and generates `replays.html`; existing receipts
prevent duplicate submission. Replay frames are labeled Simulated. This control
retains forward/inverse discretization differences (native128 finite spheres
versus the original64 inverse object grid); it is not a test of numerical identity
between the forward and inverse discretizations.
