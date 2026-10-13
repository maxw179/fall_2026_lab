# Systematic experiments

Each experiment folder contains its notebooks and helpers, with numerical output in `results/` and exported plots in `figures/`.

| Folder | Experiment |
|---|---|
| [rms_snr_sweep](rms_snr_sweep/) | Phase-diversity RMS/SNR sweeps, archives, iteration histories and analysis |
| [correction_examples](correction_examples/) | Correction image grids and iteration-error plots |
| [noise_robustness](noise_robustness/) | Camera models, noise experiments and acquisition comparisons |
| [random_start_benchmark](random_start_benchmark/) | Random-start replay and closed-loop benchmarks |
| [bead_count_sweep](bead_count_sweep/) | Completed bead-count sweep, correction examples and optimizer histories |
| [booth](booth/) | Booth correction experiment |
| [correction_comparison](correction_comparison/) | Stack feedback versus Booth |
| [regularization](regularization/) | Rho, SNR, sampling and object-size studies |
| [diversity](diversity/) | Diversity/loss diagnostics |
| [bessel](bessel/) | Research: SLM patterns and optical layouts for Bessel generation |

Run notebooks from the repository root or their experiment folders. The import/setup cells find the repository by walking through parent directories; outputs use explicit experiment-local paths.

`correction.py` and `progress.py` remain shared modules in this directory. Small import/CLI wrappers and results-path links preserve compatibility with existing kernels, tests and historical result metadata. `analysis/pd_sweep_findings.md` also remains available as a link. See [organization_manifest.json](organization_manifest.json) for every move.

## Bead-count sweep

The completed sweep notebook is in `bead_count_sweep/`, with archives in `results/` and summary plots in `figures/`. The old parent-directory results path remains a compatibility link. Shared `correction.py` and `progress.py` stay here.

Other notebooks have moved: reopen them from their experiment folders so later IDE saves use the new locations. Old cell outputs and result configuration files may contain historical absolute paths; the numerical-result compatibility links keep those paths usable.
