# Experimental workflow and user preference

Read [experimental_10_15/HANDOFF.md](experimental_10_15/HANDOFF.md) for current experimental workflow, findings and run status.
For Bouchet requests, prepare locally and provide commands to submit, wait, download, build and open the visualization. Do not connect to or submit to Bouchet yourself.

# Plotting conventions

Preserve the established plotting style unless the user explicitly requests a
different style or layout. Reuse existing plotting functions rather than making
a new layout for each experiment.

- AO and sweep examples use `systematic_testing/correction_examples/plot_pd_correction_examples.py`
  and `plot_pd_iteration_rms.py`: `bmh`, LaTeX text, 300 DPI exports, 6 × 6-inch
  acquired-plane and RMS figures, shared normalization, polar phase insets and
  the existing loop boundary annotations and external legend.
- AO demonstrations showing simulated images and estimated wavefronts use
  `systematic_testing/ao_animation/viewer.py`: the existing three polar phase
  panels, initial/current simulated images, RMS trace and HTML playback controls.
  Use voxel/object animations only when explicitly requested. For those, reuse
  `utils.plotting.animate_reconstruction`, `zernike_plot` and `plot_sample_3D`,
  as in `optimization/test_complex.ipynb`. Preserve the existing colormaps and
  phase limits; do not invent Cartesian phase maps or projection layouts.
- When adding a new axis such as elapsed time, extend the shared helper while
  preserving its appearance. Disable LaTeX only when explicitly requested or
  for headless tests on a machine without LaTeX.
