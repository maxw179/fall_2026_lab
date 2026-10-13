# Bead-count sweep

The completed experiment is organized as follows:

- `pd_rms_bead_count_sweep.ipynb`: sweep setup and summary plots; saved cell outputs are retained.
- `pd_bead_count_correction_examples.ipynb`: example image grids and saved per-iteration RMS plots, styled with the same helpers as the SNR examples.
- `results/pd_rms_bead_count_sweep_snr15_results.npz`: completed 90-trajectory checkpoint.
- `results/pd_rms_bead_count_sweep_snr15_iteration_histories.npz`: 4,205 recorded optimizer states across 270 AO loops, including coefficients, loss, estimation RMS and applied corrections.
- `figures/`: summary plots and correction-example figures.
- `export_iteration_histories.py`: export retained histories from a completed sweep kernel without rerunning optimization.

Open the sweep notebook from this folder before saving further edits. Future results and plots use explicit experiment-local paths. The old `../pd_rms_bead_count_sweep_snr15_results.npz` path is a compatibility link for historical references. Shared `../correction.py` and `../progress.py` remain in place.

The sweep computation cell starts a fresh run and refuses to overwrite existing results. To inspect the saved run in a fresh kernel, load the NPZ instead of executing the sweep.

Run the correction-examples notebook to regenerate eight matched image/iteration pairs in `figures/correction_examples_acquired_planes/` (6 × 6 inches; PNG at 300 DPI and PDF). The notebook verifies the history archive matches the checkpoint and does not rerun optimization. Older `_ao_loop_rms.pdf` files are retained as endpoint-only plots.
