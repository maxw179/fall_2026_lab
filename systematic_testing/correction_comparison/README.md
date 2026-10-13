# Correction comparisons

`correction_test.ipynb` contains the original single-condition comparison.

`pd_booth_paired_sweeps.ipynb` compares the completed cumulative PD RMS–SNR and bead-count sweeps with single-plane three-point modal brightness correction. The notebook documents the matching rules and contains the complete run, checkpoint, validation, plotting and summary code.

Each Booth trajectory uses the paired PD case's exact initial wavefront, sample and exposure. Both methods follow their own closed-loop optical states. Budgets are 36, 72 and 108 images; Booth commits corrections after 35, 70 and 105 images. Probe bias is fixed at 0.10 coefficient waves, using the existing baseline setting without test-set tuning.

Numeric checkpoints and summary JSON are in `results/paired_sweeps/`; PNG and PDF heatmaps are in `figures/paired_sweeps/`. Both the absolute and relative RMS heatmaps share color scales across methods and sweep families. Paired-difference maps show Booth minus PD residual fraction, so positive favors PD.

Filled-contour versions are also exported, with `_contour` added to each figure filename. They use the same data, color limits and ticks as the heatmaps, with 20 shared contour intervals and gray markers for sampled points. Both plot types are embedded in the notebook.

Styles match the original PD sweep notebooks: grid maps use equal-size `imshow` cells, categorical axis ticks, three-decimal values and no contour overlays or sample dots. Contour maps use physical count or log10(SNR) coordinates, gray sample markers and labeled black residual-fraction-one boundaries. Both retain the original bmh style, base font and dedicated narrow colorbar layout; comparison figures add a second row for Booth.

Set `RUN_BOOTH=False` to reload saved results and regenerate figures. With `RUN_BOOTH=True`, already completed cases are skipped; configuration and source hashes must match. Each trajectory is saved atomically on completion. Plotting and summary cells require both sweeps to be complete.

Only three wavefront directions are represented; trial directions repeat across axes. The single-plane Booth configuration and untuned probe bias limit conclusions about sensorless AO more generally. Acquisition budgets are matched, but observed photon totals and wall-clock runtime are not matched. PD photon totals and runtime were not archived.
