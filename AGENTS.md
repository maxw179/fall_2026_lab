# Plotting conventions

Preserve the established plotting style unless the user explicitly requests a
different style or layout. Reuse existing plotting functions rather than making
a new layout for each experiment.

- AO and sweep examples use `systematic_testing/correction_examples/plot_pd_correction_examples.py`
  and `plot_pd_iteration_rms.py`: `bmh`, LaTeX text, 300 DPI exports, 6 × 6-inch
  acquired-plane and RMS figures, shared normalization, polar phase insets and
  the existing loop boundary annotations and external legend.
- Optimization replay uses `utils.plotting.animate_reconstruction`,
  `zernike_plot` and `plot_sample_3D`, as in `optimization/test_complex.ipynb`.
  Keep polar phase maps, existing colormaps and phase limits, and 3D object
  rendering. Do not substitute Cartesian phase maps or projection animations
  without a user request.
- When adding a new axis such as elapsed time, extend the shared helper while
  preserving its appearance. Disable LaTeX only when explicitly requested or
  for headless tests on a machine without LaTeX.
