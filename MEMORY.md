# Project handoff — 2026-09-29

## Implemented

- `utils/psf.py`: scalar/vector 3D image formation; per-image known aberration diversities added to a shared unknown aberration. Repeat focal positions for multiple diversities.
- `optimization/reconstruction.py`: padded, frequency-wise regularized SVD sample reconstruction; Gaussian reduced loss; analytic gradient and exact reduced Hessian; damped Newton optimization with backtracking and per-iteration printing.
- Public entry points: `estimate_sample_3D`, `evaluate_loss_3D`, `evaluate_loss_derivatives_3D`, `optimize_aberration_3D`.
- Mathematical reference: `math/math.tex` (PDF also saved).
- Derivatives follow `aberration.modes`; strengths are in waves (include 2*pi factors). Hessian includes response of the refitted sample. Vector optics sum field-component intensities before applying multiphoton order.
- Loss is the full unnormalized Fourier sum of residual squared plus rho times sample norm squared, evaluated before cropping. Default padding approximates unobserved exterior measurements as zero. No PSF normalization or axial integration weights.
- Optimizer returns `(estimated_aberration, info)` with convergence status, derivatives, and history. Starts at zero unless `initial_strengths` is supplied. Negate estimated strengths for correction.

## Experiments and findings

- Notebook: `optimization/test.ipynb`; saved outputs are retained. Three-photon vector model, image and pupil grids 128x128, 40 sample planes, eight focal positions, rho=1e48. Default reconstruction padding produces 256x256 lateral grids.
- Original single spherical aberration [0,4], strength -0.15, without diversities: fitted-sample loss favored zero because an underdetermined unconstrained sample could absorb the aberration; regularization favored the smaller sample norm at zero.
- Reproduced original loss at -0.15: 9.19727656077e56; at zero: 3.80986011242e55. Before common Fourier scaling, penalties were 1.39129481245e52 and 5.75953927366e50; residual sums were 1.20982174531e50 and 5.38459076409e48.
- Added three spherical diversities per focal position (24 acquired images), initially 0 and +/-0.15 waves. User observed a minimum near truth but also a shallow local basin near +0.02 from zero initialization. Local Newton descent cannot cross loss barriers.
- IMPORTANT latest saved notebook now uses diversities [0, -0.05, +0.05] waves. Its saved outputs show successful recovery from zero:
  - Single bead: -0.14970817 after five accepted steps; loss 5.6598448269e55.
  - Random beads: -0.14971294 after six accepted steps; loss 2.1555794610e57.
  - Both report gradient and curvature tolerances satisfied. Sweeps over [-0.20,0.20) in steps of 0.01 have lowest sampled losses at -0.15. This is not proof of global optimality.
- Notebook variables such as I and mask_3D are overwritten for the second example; rerun matching acquisition/reconstruction cells together.

## Validation and runtime

Run from repository root:

```sh
/opt/miniconda3/envs/phase_diversity_env/bin/python -m unittest discover -s tests
```

13 tests pass. Default `python` is a different environment without NumPy. Tests use tiny grids and include analytic-vs-finite-difference derivative comparisons, scalar/vector optics, padding, diversity consistency, optical recovery, negative curvature, extreme objective scaling, and iteration-limit reporting.

Tests are correctness checks, not performance benchmarks. User measured about 20 seconds per full notebook iteration. Full setup requires 960 PSFs per loss evaluation, derivative propagation, plus trial loss evaluations. Current implementation rebuilds propagation geometry, repeats an SVD during derivatives, and recomputes accepted trial PSFs.

## Possible next work (discussed, not implemented)

- Cache geometry/propagators, reuse factorizations and accepted trial calculations; profile at notebook scale.
- Bounded randomized multistart for many coefficients: short runs, select best candidates, refine. User cannot afford a Cartesian sweep in high dimension.
- Coarse-to-fine exploration and staged mode fitting; compare finalists on identical full objectives.
- Explore diversities spanning more than spherical aberration to improve multi-mode identifiability. Exclude piston (unobservable in intensity).
- No multistart/global-search implementation yet; current optimizer is local.
