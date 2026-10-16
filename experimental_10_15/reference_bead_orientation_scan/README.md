All eight global rotations/reflections of the original loaded x/y arrays were tested.
Each orientation receives a new reference-only fit of the three bead centers,
brightnesses and one signed background. The recorded optical modes, commands
and Z positions are unchanged. No frame-specific registration or intensity fit.

The winner is flip_x: model-array `images[:, ::-1, :]`, a left-right mirror in
plots, starting from the ORIGINAL untransposed images. All raw acquisition files
and previous optimizations are preserved. This is the best tested image-axis
hypothesis under the reference-calibrated model, not a full optical calibration.

Orientation ranking uses aggregate RMSE on all 134 unflagged aberrated images;
none of the 10 flagged artifact images affect ranking. `scores.csv` also reports
mean per-image correlation and separate errors for flagged images 112 and 130.
The same-bead geometry and optical model can still have residual mismatch.

`flip_x/measured_vs_simulated.pdf`: complete best-orientation comparison.
`orientation_examples.pdf`: original, transposed, and best mirror for images
76, 94, 112 and 130 (last two flagged); shared scaling within each probe type.
`scores.csv` / `summary.json`: rankings, metrics and transform provenance.

Reproduce (cached simulations are verified against the transformed measurements):

```bash
python -m experimental_10_15.scan_bead_orientation
```
