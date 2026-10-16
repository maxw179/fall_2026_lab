Reference-constrained simulation of the many-fluorobead stack.

Open `measured_vs_simulated.pdf`: measured (left) and simulated (right), grouped
by aberration type. Both columns share one display normalization per type across
all Z planes. All 162 images are included; the ten previously flagged artifact
images are marked with an asterisk. Per-page PNGs are under `figures/`.

Three uniform 1-µm spheres are fitted using only the 18 reference images. Each
bead has one static 3D center and brightness; one signed background is shared.
Before/after repeats are averaged at each Z for the robust reference fit. No
measured data are shifted or aligned. Individual bead brightnesses, positions
and the background are then fixed for all aberrated/diversity predictions.

Recorded optical settings and Zernike commands drive vector 3P propagation.
System AO is assumed to cancel native aberration. A sphere uses 21 axial slices;
reference fitting interpolates a 0.05-µm axial template grid and native-pixel
lateral shifts. Final predictions evaluate each recorded focus directly.
Simulations are computed on native 128² pixels before 2×2 block means to 64².
No noise or acquisition artifacts are added. These are model predictions;
finite sampling, residual native phase, drift and model mismatch can remain.

`bead_centers.csv`: inferred centers (µm relative to recorded origin) and fixed
brightnesses in reference-calibrated arbitrary units. `reference_fit.json` saves
the fit. `comparison.npz` contains the complete measured and predicted stacks
in image,x,y order and original zero-based source indices. `image_metrics.csv`
and `comparison.json` record errors and provenance.

Reproduce from the repository root in phase_diversity_env:

```bash
python -m experimental_10_15.reference_bead_simulation
```

The cached finite-sphere renderer is independently tested against the public
`Microscope.compute_image_3D` forward model on odd/even lateral grids.
