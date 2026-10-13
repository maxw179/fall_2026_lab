# CPU acceleration with the exact Hessian

The solver keeps float64/complex128 arithmetic, the same optical sampling,
regularized SVD estimator, exact reduced Hessian, and Newton/backtracking rules.
CPU FFTs now use SciPy's double-precision FFT implementation. Additional options
provide alternative propagation and parallel execution without changing the
mathematical objective. Floating-point reduction order can change, so results
agree within roundoff rather than being bitwise identical.

## Use in existing notebooks

```python
from optimization import reconstruction

estimated, info = reconstruction.optimize_aberration_3D(
    microscope, grid, images, focal_z_levels, sample_z_levels,
    modes, rho, diversities=diversities,
    backend="cpu", method="newton",
    propagation="zoom", fft_workers=2, cpu_workers=1,
)
```

These three new keyword arguments apply to all four public reconstruction APIs
and to `CPUReconstructionWorkspace` / `ReconstructionWorkspace`:

| Option | Meaning | Default |
| --- | --- | --- |
| `propagation` | `dense`: matrix pupil sums; `zoom`: separable ZoomFFT pupil sums | `auto` uses dense |
| `fft_workers` | Threads for CPU PSF/observation/inverse FFTs and ZoomFFT propagation | 1 |
| `cpu_workers` | Threads for independent frequency-batched sample SVD fits | 1 |

ZoomFFT evaluates the same finite pupil sum on the same physical coordinates.
It does not interpolate an image, truncate pupil contributions, reduce precision,
or change resolution/padding. It supports scalar/vector optics and offset,
rectangular grids. The unit-circle ZoomFFT specialization avoids the repeated
complex powers used by a general CZT. GPU propagation still uses dense matrices.

Dense remains the default because ZoomFFT's benefit depends on pupil dimensions,
batch sizes, and the CPU's BLAS library. For explicit GPU calls leave the CPU-only
options at defaults. `propagation="zoom"` and `cpu_workers>1` reject CUDA workspaces.

## Reuse geometry and parallelize scans

```python
solver = reconstruction.ReconstructionWorkspace(
    microscope, grid, images, focal_z_levels, sample_z_levels,
    modes, rho, diversities=diversities, backend="cpu",
    propagation="zoom", fft_workers=1,
)
losses = solver.losses(coefficient_vectors, workers=4)
# Full Newton runs from independent starting points; results retain input order.
results = solver.optimize_starts(initial_vectors, workers=4, max_iterations=50,
                                 num_random_starts=0)
```

Parallel evaluations share fixed geometry and data read-only. Each thread has
its own fit cache, so workers never mutate another worker's optimizer state.
Parallel scans preserve input order and repeated points. Each Newton run still
computes the exact Hessian and evaluates the existing convergence criteria.
The methods return every result; selecting a final starting point is explicit.

Workers multiply fit/derivative memory. Keep `fft_workers=1` and BLAS threads=1
when running several independent optimizations to avoid competing nested thread
pools. The scan helpers set each worker's `cpu_workers=1`. Do not mutate a workspace
or its acquisition data while running these operations. Correction feedback
updates remain sequential because each acquisition depends on the prior update.

## Tune BLAS and batch sizes

Set the applicable thread variables **before starting Python or Jupyter**, for
example on Linux/macOS:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m optimization.performance.benchmark_cpu_acceleration --size 128 --pupil 128 --images 6 --planes 8
```

Repeat with thread counts 2 and 4 to compare. Only the variable for your installed
BLAS implementation takes effect; inspect `numpy.show_config()`. For Jupyter,
start the server/kernel from the configured environment. Setting variables after
NumPy imports may have no effect. On Windows, set the corresponding environment
variables before launching the notebook process.

The benchmark compares dense/ZoomFFT propagation, FFT worker counts, SVD worker
counts, PSF batch sizes, and independent scan workers. It warms every variant,
interleaves timing order, clears fit caches before measured calls, and verifies
loss/gradient/Hessian agreement before reporting speedups. Workspace construction
is excluded. All timed derivative evaluations compute the exact Hessian.

`psf_batch_size` and `derivative_batch_size` control field batching; larger batches
can improve throughput or hurt cache/memory use. `frequency_batch_size` balances
SVD batch overhead against parallel work and memory. Retaining fields and SVD
factors avoids recomputation; keep those existing defaults unless memory demands
otherwise. No frequency bin or pupil sample is discarded.

## Local measurements and validation

`performance/cpu_acceleration_128.json` records a development-Mac/OpenBLAS run:
128x128 image/pupil grids, three images, four sample planes, three optimized modes,
three timed repeats, BLAS threads=1. The best tested full derivative setting was
ZoomFFT with two FFT workers: about 1.49x faster than the current dense workspace.
Four parallel scan workers were about 3.24x faster than the serial ZoomFFT scan.
These are workload-specific timings, not predictions for another CPU or an
end-to-end correction notebook. A separate 64-grid run showed slower ZoomFFT
Hessians, which is why it is not enabled globally.

On a separate 64x64, three-image/four-plane test, setting OpenBLAS threads to
2 reduced the dense derivative median from about 62 ms to 49 ms; four BLAS
threads did not improve it further. Thread tuning can help independently of
changing propagation. These short measurements should be repeated on the target
machine at actual acquisition dimensions.

Maximum loss-scaled component differences in the 128-grid run were approximately
3.6e-13. Additional tests compare complex fields on offset/asymmetric grids up to
128 pupil samples, sample volumes, gradients, exact Hessians, Newton results,
parallel scans, and parallel starting points. Finite differences independently
check the Hessian. The existing independent reference tests continue to pass.

```sh
python -m unittest discover -s tests -v
```

Neither lower resolution, reduced precision, approximate Hessians nor a different
optimizer is enabled by these changes.

## Random initialization

Every optimization screens 100 random aberrations by loss, alongside zero and
any supplied initial point, then optimizes only the lowest-loss candidate.
Gaussian coefficient directions use only the requested correction modes. Their
RMS is normalized using `zernike_RMS` and sampled log-uniformly from 0.005 to
0.50 waves. Screening computes no phase gradient or Hessian; the subsequent
Newton solve still uses the exact Hessian.

```python
estimated, info = solver.optimize(
    num_random_starts=100,
    random_start_rms_range=(0.005, 0.50),
    random_start_seed=0,
)
```

These options also work in `optimize_aberration_3D`, on CPU and CUDA, for Newton
and gradient methods. Set `num_random_starts=0` to use your supplied initial point
(zero by default) directly. Seed 0 repeats the pool for matching modes and optics;
use `None` for fresh random draws. `info["initialization"]` contains candidate
strengths/losses, the selected point and screening time, or `None` when disabled.
A separate progress bar counts completed screening loss evaluations, including
zero and any supplied point. It closes before optimization begins. Set
`show_initialization_progress=False` to hide it.
Selecting a low-loss start does not guarantee a better wavefront RMS or a global
minimum. The random-start benchmark disables this internal screening because it
already screens its explicitly controlled pools.
