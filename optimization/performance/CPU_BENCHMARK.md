# CPU reconstruction benchmark

The seven prototype checks passed, and both representative benchmarks completed
on 2026-10-07. See [CPU_RESULTS.md](CPU_RESULTS.md) for those measurements.
The workspace is now the default in `optimization.reconstruction`. Tests have
been updated for the production integration but have not been rerun, as requested.
The new files use NumPy and project utilities only; they do not import the CUDA
implementation, CuPy, or JAX.

`optimization/_reconstruction_cpu.py` snapshots the fixed optical geometry, Zernike basis, diversity
and defocus phases, FFT wrapping indices, and observed-image FFT. It retains the
exact reduced-objective Hessian and the existing damped Newton stopping rules.
The public reconstruction functions use it automatically, with their existing
signatures and return values. Direct users can import
`CPUReconstructionWorkspace` (also aliased as `ReconstructionWorkspace`) from
`optimization.reconstruction`. The old performance-module import still works.

The former solver is preserved in `reconstruction_reference.py`. Benchmarks
compare against that independent implementation rather than comparing the new
public API with itself. No production code imports the reference solver.

`optimize_aberration_3D(..., method="gradient")` selects steepest descent with
Armijo backtracking. The default `method="newton"` retains the exact Hessian.
Gradient mode evaluates the analytic envelope gradient, skips second PSF
derivatives and the Hessian response solve, and avoids retained full SVD bases
in the public optimizer. It returns `info["hessian"] = None` and stops on the
scaled gradient tolerance alone, so it does not perform the Newton method's
negative-curvature check. Both methods respect `max_step`, `max_backtracks`,
`max_iterations`, and `strength_tolerance`.

The diversity notebook exposes this choice as `OPTIMIZATION_METHOD`, initially
`"newton"`. Tests for the gradient path have been added but have not been run.

The cumulative benchmark variants are:

| Variant | Changes |
| --- | --- |
| `geometry_and_batches` | Precompute geometry; batch PSFs and mode pairs; reuse base/first fields within derivative calls |
| `shared_svd` | Also share one SVD between sample fitting and the exact Hessian response |
| `spatial_hessian` | Also contract second PSF derivatives against spatial correlations instead of Fourier-transforming each derivative |
| `all_improvements` | Also share PSFs and derivatives with identical defocus and diversity phases |

All variants cache the latest fit so an accepted line-search point can be reused
at the next derivative evaluation. Duplicate defocus coordinates are compared
exactly: nearby floating-point values are not rounded together. Known diversity
modes may be outside the correction basis. Full right singular bases preserve
null-space contributions when there are more sample planes than acquired images.
No changes to rho, physical PSF scaling, optical mode, or numeric precision are
made by the prototype.

## Run when ready

Use your project environment from the repository root to reproduce the checks
and benchmark runs.

First run the numerical checks:

```sh
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/diversity_mpl /opt/miniconda3/envs/phase_diversity_env/bin/python -m unittest discover -s tests -p 'test_reconstruction_cpu_workspace.py' -v
```

The checks compare loss, gradient and Hessian with the preserved independent
reference solver for scalar/vector optics, multiphoton orders, odd/even images,
padding aliases, K<P null spaces, K>P, diversity modes outside the correction
basis, and small rho. They cover finite differences, cache snapshots, field
recomputation, a small Newton recovery, and stationary negative-curvature escape.

Then benchmark the 64x64 one-mode setup (24 images, 40 sample planes, rho=1e48,
Poisson SNR=50, zero initial estimate, spherical diversities 0/+-0.05 waves):

```sh
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/diversity_mpl /opt/miniconda3/envs/phase_diversity_env/bin/python -m optimization.performance.benchmark_cpu --output optimization/performance/cpu_64_one_mode.json
```

A smaller case makes the 15-mode exact Hessian comparison more manageable:

```sh
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/diversity_mpl /opt/miniconda3/envs/phase_diversity_env/bin/python -m optimization.performance.benchmark_cpu --resolution 32 --pupil 32 --sample-planes 9 --focal-planes 5 --modes 15 --output optimization/performance/cpu_32_fifteen_modes.json
```

For full Newton runs, add `--optimize`. To tune propagation batches, add
`--psf-batches 1 8 16`. To limit that sweep to the final implementation, use
`--variants all_improvements`. `--profile` collects profiles separately from
the repeated timings. `--snr 0` selects clean images. Other knobs include
`--rho`, `--padding`, `--frequency-batch`, `--derivative-batch`, `--repeat`,
`--max-iterations`, `--gradient-tolerance`, and `--seed`.

Benchmark thread counts in separate processes by changing
`OPENBLAS_NUM_THREADS=1` to 2 or 4. Set it before NumPy is imported; changing a
notebook environment variable after import may not alter the active thread pool.
There is no automatic global thread-setting change in the prototype.

## Interpret the results

Before timing each variant, the script asserts agreement at zero, half-truth,
and true strengths, including derivative evaluation after a cached loss fit.
Loss, gradient and Hessian are compared after dividing by reference loss, with
rtol=2e-8 and atol=2e-9 by default. Failed checks stop the benchmark.

For every variant the JSON records:

- Workspace setup time and numerical agreement errors.
- Warm loss and exact-derivative timings, with fit caches cleared before each
  timed call so repeated points cannot produce misleading cache-hit speedups.
- Derivative time after an accepted loss fit; the preceding loss call is outside
  that particular timer, so this is a marginal cost, not a complete step time.
- Paired reference/prototype calls in random order, repeated timings, medians,
  and their speedup ratios. Variant order itself is fixed.
- When requested, full Newton time, time including workspace setup, final
  coefficients, residual wavefront RMS, final loss, success flag, stopping
  message, scaled gradient/curvature, and histories. These runs are single
  measurements, rather than repeated optimizer timings.
- NumPy/BLAS configuration, thread environment and partial array-storage
  estimates. Storage estimates are not process peak-memory measurements.

Synthesis and agreement checks are excluded from steady-state timings. Results
are saved after each completed variant, and the requested output file is
overwritten. Run without other heavy computations for meaningful comparisons.

Retaining SVD factors and derivative fields can use substantial RAM. Frequency
batching limits temporary arrays, but the complete retained bases and first OTFs
still scale with the number of frequencies and planes. Use `--recompute-fields`
to trade extra propagation for lower field storage. Construct only one workspace
per worker when parallelizing independent experiments.

The reference loss and optimizer are nonconvex. Matching derivatives does not
guarantee identical line-search decisions near floating-point thresholds. Judge
the optional full runs by loss, recovery accuracy, stopping conditions and wall
time, rather than accepted-step count alone.
