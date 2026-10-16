This diagnostic scans **α × imposed wavefront + cumulative correction** at 51
fractions from −0.5 to 2 in steps of 0.05. The object is re-estimated at each
point, using the original regularized objective and all 144 images/nine planes.
System AO and diversities stay fixed. The three resolutions share one GPU job.
Both the 100-start and 1,000-start final fits are independently reevaluated.

From the repository root in `phase_diversity_env`:

```bash
python experimental_10_15/multiple_beads_truth_loss_profiles_20261015/manage.py submit
python experimental_10_15/multiple_beads_truth_loss_profiles_20261015/manage.py collect
```

`collect` watches, downloads and plots 6 × 6-inch figures using the established
`bmh` style, LaTeX text, external legends and 300-DPI PNG/PDF exports, plus
`profiles.csv` and `minima.csv`. No HTML is generated. Raw losses and numerical replay checks are saved in
`solve/results/*.json`. `collect --local-only` rebuilds figures without SSH.
Saved receipts prevent duplicate submission. All original solves are preserved.

The marked minimum is the best sampled fraction, with 0.05 resolution, not a
global minimum over all modes. Fit losses appear as horizontal lines because
their 17-mode estimates generally do not lie on the true-wavefront scaling line.
