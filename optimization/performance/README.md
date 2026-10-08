# Reconstruction optimization performance investigation

For the NumPy CPU workspace now used by the public reconstruction APIs, checks, and staged
benchmark, see [CPU_BENCHMARK.md](CPU_BENCHMARK.md). Its completed checks and
measurements are in [CPU_RESULTS.md](CPU_RESULTS.md). Integration tests have
not been rerun since promotion, at the user's request. The measurements below
belong to the earlier investigation.

Measured on the local ARM macOS CPU with NumPy 2.4.6, 2026-10-07.
Production reconstruction code and notebook settings are unchanged. JAX and CuPy
are not installed locally; no GPU or JAX timings are claimed.

## Measured results

The controlled main timing used OPENBLAS_NUM_THREADS=1 and no other benchmark
running concurrently. Two warm repeat timings are stored along with a separately
profiled call. Workspace setup and image synthesis are excluded from steady-state
call timings. The workspace is the existing CUDA implementation's internal NumPy
test backend, **not GPU execution**. Loss, gradient, and Hessian are compared
against the independent CPU reference after division by its loss scale.

| Case | CPU reference exact derivatives | Workspace exact derivatives (batch 16) | Gradient-only prototype |
|---|---:|---:|---:|
| 64x64 image/pupil, 24 acquisitions, 40 sample planes, 1 mode | 4.222 s | 3.693 s | Not measured |
| 32x32 image/pupil, 15 acquisitions, 9 sample planes, 15 modes | 2.640 s | 1.981 s | 0.249 s |

On the 15-mode case the gradient-only prototype is
10.6x faster per evaluation
than the CPU exact-Hessian reference. It computes the same profiled objective and
analytic envelope gradient, refitting the object each time. It skips both second
PSF derivatives and the Hessian's object-response solve. This is not a measured
end-to-end optimizer speedup: quasi-Newton optimization may require more iterations
or reach a different stationary point. The current Newton optimizer also tests
negative curvature; a gradient-only method needs its own convergence assessment.

The default-thread preliminary loss timings were 25.39 and 27.88 seconds; with
one BLAS thread the controlled loss timings were [1.7827890003100038, 1.8218901250511408].
The default-thread full benchmark was interrupted because it was much slower.
The preliminary and controlled timings were not a randomized paired benchmark,
so treat this as evidence to tune threading, not a universal speedup claim.

## Where time goes

In reconstruction.py, _solve_reconstruction builds every PSF separately using
compute_PSF/rw_fast, reconstructing pupil geometry and phase bases. The derivative
evaluation subsequently propagates fields again and runs a second SVD per
frequency to compute the reduced Hessian. The profile shows pupil propagation,
SVD, and derivative FFTs as the principal costs. For 15 modes there are 120
upper-triangular second derivatives per PSF, in addition to 15 first derivatives.

reconstruction_cuda.py already snapshots geometry and observations, batches PSFs,
and shares one SVD between object fitting and Hessian response. Its derivative
path still regenerates fields/first derivatives in its second-derivative pass;
accepted line-search points are also regenerated at the next derivative evaluation.

## Recommended implementation order

1. **Persistent CPU workspace:** snapshot geometry, basis maps, diversity phases,
   defocus factors, FFT wrapping and observed FFT once per acquisition. Reuse
   factorization within a derivative evaluation. Keep the independent reference
   solver for numerical parity checks. The existing CUDA workspace supplies a
   starting design, but its _xp=np hook is private and is not a supported CPU API.
2. **Optional loss/gradient optimizer:** expose an analytic gradient-only path and
   benchmark L-BFGS against Newton at equal recovered-wavefront accuracy and
   convergence budgets. Retain exact Newton for small mode sets and cases where
   curvature information is useful. An exact Hessian-vector product with a
   truncated Newton method is another option without materializing all mode pairs.
3. **Reuse identical PSFs:** cache by exact (defocus, diversity phase) within an
   evaluation. In the baseline grid, 320 pairs per diversity reduce to 100 exact
   distinct defocus distances. Mathematically there are 75, with differences due
   to floating-point construction. Integer axial grid indices could obtain 75
   without silently rounding physical coordinates. The factor of 3.2 is a bound
   on removed PSF work, not a total optimizer speedup. Reuse derivatives too.
4. **Batch and fuse propagation and derivative contractions:** tune independently
   on CPU and GPU; batch 16 was only modestly faster here. Avoid allocating all
   mode-pair OTF stacks. Keep fields/factors for accepted points only when the
   memory cost is justified. Device-to-host scalar transfers in line search should
   be measured on the actual GPU.
5. **Optional alternative object solve:** for K<P, the dual identity
   F=S^H (SS^H+rho I_K)^(-1) D reduces matrix size. Batched Cholesky may help, but
   forming a Gram matrix squares conditioning. When rho is tiny relative to
   operator power, roundoff can defeat the regularization. Preserve SVD fallback,
   null-space contributions in Hessian solves, and verify small-rho derivatives.
   CuPy's fast batched SVD path requires both matrix dimensions <=32; the usual
   24x40 case misses that path. Do not reduce physical plane count just to trigger it.

## JAX assessment

JAX is technically suitable: the hot path consists of complex phase arithmetic,
separable propagation, FFTs, batched linear algebra, and real scalar objectives
with real coefficient inputs. A staged implementation should precompute geometry,
then jit a pure array function, with vmap or fixed-size batches over optical planes
and frequencies. Python microscope/aberration objects belong outside compiled
kernels. Changing array shapes, such as mode count or plane count, generally
requires a new compiled specialization. Keep the small host optimizer initially.

For gradient-only optimization, the envelope identity is
  dL/da = -2 Re sum(weights * residual.conj() * (dS/da) F).
At the fitted object, its own gradient vanishes, so the first derivative does not
require differentiating through singular vectors. A custom derivative or implicit
linear-solve rule is preferable to blindly autodifferentiating the current SVD
pipeline. Second derivatives must still include the object's response: stopping
its gradient is valid for the first derivative only, not an exact Hessian.

Start with float64/complex128 and jax_enable_x64=True. The current PSFs/data can
have magnitude around 1e26, whose squared magnitudes overflow float32. Smaller
precision needs fixed, physically equivalent nondimensionalization, with rho
transformed consistently; per-aberration PSF normalization changes the objective.
GPU double-precision throughput and SVD performance must be measured. Apple GPU
support is experimental; CPU support on ARM macOS is available. NVIDIA GPU JAX
runs on supported Linux setups. JAX is not automatically faster than the existing
CuPy backend: its strongest potential advantages here are fused kernels, batching,
and cheap gradient/Hessian-vector computation.

Measure compilation separately from warm execution, synchronize GPU operations,
and compare loss/derivatives, full optimizer trajectories, residual wavefront RMS,
peak memory, and time to equivalent accuracy. Include small rho, repeated/zero
singular values, K<P, both optical modes, and padding aliases in validation.

## FFT-based propagation prototype

The uniformly spaced source/output grids allow a separable chirp-Z implementation
of the same discrete pupil sum (not a replacement of the vector optical model).
The prototype in propagation.py checks complex-field agreement for masked random
pupils with 4 PSFs and 3 vector components. At 64 it was slower than dense matmul;
at 256 it was about 2.2x faster in the preliminary microbenchmark. Relative field
errors were 2.6e-14 through 2.1e-12. Those microbenchmarks overlapped a different
CPU benchmark and need isolated repeats before adoption. This says nothing yet
about end-to-end loss/gradient accuracy or GPU performance. SciPy notes that
ZoomFFT can improve accuracy on the unit circle; benchmark that as well.

## Reproduce

From the repository root, using the project Python environment:

```sh
OPENBLAS_NUM_THREADS=1 python -m optimization.performance.benchmark --output optimization/performance/single_thread.json
OPENBLAS_NUM_THREADS=1 python -m optimization.performance.benchmark --resolution 32 --pupil 32 --sample-planes 9 --focal-planes 5 --modes 15 --output optimization/performance/many_modes.json
OPENBLAS_NUM_THREADS=1 python -m optimization.performance.propagation
```

Set the BLAS environment before Python imports NumPy. The scripts overwrite their
specified result files. These are research benchmarks using private workspace
methods; they do not replace production APIs.

## Primary documentation

- [JAX JIT and timing](https://docs.jax.dev/en/latest/201/jit.html)
- [JAX x64](https://docs.jax.dev/en/latest/101/default_dtypes.html)
- [JAX platform support](https://docs.jax.dev/en/latest/installation.html)
- [JAX implicit linear solve](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html)
- [JAX autodiff and Hessian-vector products](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
- [CuPy SVD algorithm selection](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html)
- [SciPy chirp-Z transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.CZT.html)
