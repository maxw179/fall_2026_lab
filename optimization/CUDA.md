# NVIDIA GPU optimization

All four public functions in `optimization.reconstruction` now default to
`backend="auto"`: sample reconstruction, loss evaluation, analytic gradient/exact
Hessian evaluation, and aberration optimization. Existing notebook imports and
calls automatically use a working NVIDIA GPU through CuPy. Without CuPy or a
usable CUDA device they use NumPy on the CPU.

```python
from optimization import reconstruction
print(reconstruction.get_backend())  # 'cuda' or 'cpu'

estimated, info = reconstruction.optimize_aberration_3D(
    microscope, grid, images, focal_z_levels, sample_z_levels,
    modes, rho, diversities=diversities, backend="auto",
)
print(info["backend"])
```

Use `backend="cpu"` to force CPU or `backend="cuda"` to require GPU. Explicit CUDA
requests raise an error when unavailable. Errors during a GPU solve (including
out-of-memory errors) propagate; they do not silently restart on the CPU.
`optimization.reconstruction_cuda` exposes the same four functions for explicit
GPU use. Both Newton and `method="gradient"` run their numerical work on the GPU.

## Install on the NVIDIA machine

Use the project environment, then add CuPy:

```sh
conda env create -f environment.yml
conda activate phase_diversity_env
conda install -c conda-forge cupy
```

Install a CuPy build compatible with the GPU machine's NVIDIA driver. The portable
project environment keeps CuPy optional so CPU installations also work on macOS.
See the [official installation guide](https://docs.cupy.dev/en/stable/install.html)
for supported CUDA versions and alternative wheel installations. Do not install
multiple CuPy distributions in one environment.

## Persistent workspace

Reuse an acquisition for optimization, scans, and final sample estimation:

```python
solver = reconstruction.ReconstructionWorkspace(
    microscope, grid, images, focal_z_levels, sample_z_levels,
    modes, rho, diversities=diversities, backend="auto",
    psf_batch_size=8, derivative_batch_size=8, frequency_batch_size=4096,
)
estimated, info = solver.optimize(method="newton")
sample, diagnostics = solver.estimate_sample(estimated.strengths, return_info=True)
```

A workspace snapshots data and optical geometry. Construct a new one after
changing them. On CUDA, construct and use it within the same current CuPy device
context. Respect scheduler allocations through `CUDA_VISIBLE_DEVICES`. One
workspace uses one GPU and is not safe for concurrent calls.

`CUDAReconstruction` (also named `CUDAReconstructionWorkspace`) supports
`solver.sample(strengths, return_device=True)` to return a cropped CuPy volume.
The CUDA public `estimate_sample_3D` also accepts `return_device=True`. Otherwise
sample outputs retain the existing `Image_Mask_3D` interface and NumPy arrays.
Losses are host floats; returned gradients, Hessians, coefficients and optimizer
history are small host arrays. Input image stacks may be NumPy or CuPy arrays.

## Acceleration and memory

GPU execution covers batched scalar/vector pupil propagation, multiphoton PSFs,
real FFTs, regularized batched SVD fits, analytic first/second derivatives,
reduced Hessian contractions, inverse FFTs, and residual diagnostics. Host setup
constructs geometry and Zernike bases once. The small optimizer eigensolve and
line-search decisions run on the host.

Residuals and reduced-Hessian response terms are evaluated in singular-value
coordinates. For K<=P this avoids subtracting nearly equal observed/predicted
arrays; expanding the curvature contraction also avoids cancellation between
large terms and artificial right-null-space response. This matters when rho is
very small relative to operator power. Both CPU and CUDA use these algebraically
equivalent formulas, with unchanged objective and double precision. Retained
left bases add 16*Q*K*min(K,P) bytes when reuse_svd=True.

The GPU shares the optimized CPU workspace algorithm: exact repeated PSFs are
merged, second derivatives contract in the spatial domain without M-squared OTF
storage, SVD factors are reused for the object response, and accepted line-search
fits are cached. Gradient optimization skips second derivatives and retained SVD
factors. Float64/complex128 preserve the existing optical scale and mathematics.

For K images, P sample planes, M modes and Q stored real-FFT frequencies, base
OTFs use `16*K*P*Q` bytes and first derivatives add `16*M*K*P*Q` bytes. Retained
fields, fit factors, FFT workspaces and the CuPy pool consume additional memory.
Use `solver.memory_summary()` for array estimates. Lower `psf_batch_size`,
`derivative_batch_size`, or `frequency_batch_size` to reduce temporary memory.
Use `retain_fields=False` to trade field storage for recomputation and
`reuse_svd=False` to trade factor storage for an additional Hessian factorization.
These settings do not reduce the full OTF arrays.

## Experimental QR reduction for rectangular SVDs

The 128x128 / 24-image / 40-plane benchmark measured on Bouchet's RTX 5000
Ada took about 55 seconds for each GPU operation, versus 5.04 seconds for CPU
loss and 16.14 seconds for CPU loss/gradient/Hessian. Numerical parity passed.
Those nearly equal GPU times suggest shared fit work dominates; stage profiling
on the device is still needed to attribute the measured time.

CuPy 14.2 uses a separate cuSOLVER SVD for each frequency when either matrix
dimension exceeds 32. The padded benchmark has 33,024 stored frequencies, each
with a 24x40 operator. Merely increasing frequency_batch_size does not eliminate
that per-frequency loop.

An opt-in CUDA workspace option avoids that dispatch for eligible operators:

```python
solver = reconstruction.ReconstructionWorkspace(
    microscope, grid, images, focal_z_levels, sample_z_levels,
    modes, rho, diversities=diversities, backend="cuda",
    svd_method="qr",
)
```

For S shaped KxP with K<P, this computes a batched Householder QR of S^H, then
the SVD of the KxK triangular factor's adjoint. A 24x40 SVD becomes a 24x24
SVD, eligible for CuPy's fast batched Jacobi implementation. The QR uses array
operations across all frequencies, rather than a single-matrix QR in a Python
loop. Tall operators use QR of S instead. Full right bases retain null-space
directions for the exact Hessian. There is no Gram matrix, singular-value
truncation, precision reduction, or change to physical planes, rho or padding.

Reduction applies only when min(K,P)<=32<max(K,P). Other shapes use the existing
SVD. Frequencies with sum(abs(S)**2)/rho > 1e6 also use the existing SVD: this
is a conservative numerical guard motivated by derivative sensitivity under
very weak regularization, not a universal error bound. The benchmark reports
QR/direct frequency counts when profiling. Always verify loss, sample, gradient,
Hessian and optimizer agreement on the real acquisition.

The default remains svd_method="direct". Portable NumPy checks validate the
factorization, extreme scales, rank deficiency, full null-space response, cached
fits, optical derivatives and finite differences, Newton and random screening.
On the first Bouchet CUDA test run, factorization, scale/rank checks, null-space
response and optimizer checks passed. Optical parity failed in the very weak
regularization test: the shared residual/curvature subtractions amplified
roundoff, even with the direct-SVD fallback. The singular-coordinate formulas
above resolve that issue in portable checks without relaxing comparison
tolerances. Added checks compare nearly interpolated loss/gradient/Hessian
against a closed-form rational objective down to rho=1e-18, and optical finite
differences now include the weak-regularization case. On the subsequent Bouchet
CUDA run, all numerical QR/CUDA checks passed, including the previously failing
optical parity and independent closed-form test. The full suite had two errors
in older synthetic-kernel fixtures that mixed NumPy arrays into the auto-selected
CUDA workspace; those fixtures now create arrays on the selected backend and
need a GPU rerun.

The subsequent user-reported B200 benchmark (128x128 image/pupil, 24 images,
40 planes, two modes, QR, PSF batch 8, frequency batch 4096, BLAS threads=1)
passed numerical parity. Median CPU/CUDA times were 3.951/0.245 seconds for loss,
6.682/0.279 for loss/gradient, and 11.226/0.370 for loss/gradient/exact Hessian.
A single full optimization including 100 random starts took 400.197 seconds on
CPU and 25.586 on B200; both converged to matching coefficients/loss. History
had two states (one accepted Newton update). Screening alone took 24.828 GPU
seconds, approximately 97% of optimization time. These are warmed synthetic
two-mode timings, not measurements of a real many-mode acquisition. Those runs
precede the loss-only candidate screening implementation below.

On Bouchet, after pushing/pulling these changes and within a GPU allocation:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
REQUIRE_CUDA=1 python -m unittest discover -s tests -p test_batched_svd.py -v
python -u -m optimization.performance.benchmark_cuda --size 128 --pupil-size 128 --images 24 --planes 40 --cuda-svd qr --profile --repeats 3 > benchmark_qr.json
```

The new benchmark checks three coefficient points, including a cached loss fit
and reconstructed samples, before timing. It interleaves warmed backends in
random order, clears fit caches and synchronizes GPU completion. Profiles are
separate calls with synchronization at stage boundaries; inclusive/exclusive
stage times have profiling overhead and must not be substituted for the normal
operation timings. GPU model, runtime/driver versions, raw repeats and listed
workspace memory estimates are saved in the JSON. Progress goes to stderr.

Use --cuda-svd compare to time CPU, original CUDA and QR CUDA together. Increase
--modes to match the real fit (default 2), and try --psf-batch-size 32 and
--derivative-batch-size 16 after profiling. Larger batches increase memory.
Add --optimize --max-iterations 20 --random-starts 100 to measure a full Newton
run and seeded initialization; optimizer strengths and loss are checked against
CPU. These synthetic random-image runs measure timing/parity, not successful
recovery of a physical aberration. Benchmark actual acquisitions and time to
equivalent wavefront accuracy before choosing an optimizer or GPU.

## Verify and measure

```sh
REQUIRE_CUDA=1 python -m unittest discover -s tests -v
python -m optimization.performance.benchmark_cuda --size 64 --pupil-size 64 --images 6 --planes 8
```

Tests include scalar/vector optics, padding and boundary aliases, underdetermined
sample fits, numerical gradient/Hessian checks, CPU parity, optimizer behavior,
and automatic selection. Without CUDA, portable NumPy execution tests the shared
GPU algorithm; device tests skip. `REQUIRE_CUDA=1` makes missing CUDA an error.

The benchmark warms both implementations, clears caches, synchronizes the GPU,
checks numerical parity, and reports median wall times and speedups for sample,
loss, gradient, and Hessian operations. It excludes workspace setup and includes
host output transfers. Use dimensions matching your acquisition for useful
measurements. Follow the [CuPy performance guide](https://docs.cupy.dev/en/stable/user_guide/performance.html)
when timing asynchronous device work.

Actual CUDA execution and speedup have not been validated on the development
Mac. Small problems may run faster on CPU; double-precision throughput and memory
capacity depend strongly on GPU model. No multi-GPU distribution is implemented.

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

## Batched loss-only candidate screening

CUDA random initialization now uses a separate scoring path. The default
screening_method="linear" batches four candidates, while CPU screening remains
serial. The Gaussian directions, RMS range, seed, zero/supplied candidates and
number of random points are unchanged. Newton/backtracking still use the exact
SVD-based fit and Hessian.

For K<=P, write T=S/sqrt(rho). Solve the smaller KxK system
(I+T T^H)y=D, then evaluate the minimized objective as Re(D^H y). The benchmark's
24x40 object fit becomes a 24x24 regularized linear solve during screening.
This computes the same minimized loss without materializing the fitted object,
right singular basis or Hessian factors. For K>P use the smaller primal system
(I+T^H T)x=T^H D and score ||D-Tx||^2+||x||^2. CuPy's batched linear solve avoids
the iterative SVD work for eligible small matrices.

Forming a Gram matrix can amplify rounding errors when regularization is weak.
The linear route is used only when sum(abs(S)**2)/rho <= 1e6 and the smaller
system is within CuPy's batched solve size limit. Other frequencies use a stable
SVD loss in singular coordinates, including left-null-space residuals. Candidates
within 4e-8 relative of the best fast score are rechecked with the canonical loss
before choosing the starting point. No point is dropped and no physical
parameter, Fourier frequency or precision is changed.

```python
solver = reconstruction.ReconstructionWorkspace(
    microscope, grid, images, focal_z_levels, sample_z_levels,
    modes, rho, diversities=diversities, backend="cuda", svd_method="qr",
    screening_method="linear", screening_batch_size=4,
)
estimated, info = solver.optimize(num_random_starts=100, random_start_seed=0)
print(info["initialization"])
# Optional ordered scoring of an externally supplied coefficient pool:
scores = solver.screen_losses(candidate_strengths)
```

CUDA-only workspace options screening_method="serial" and "svd" provide the
original full-fit baseline and batched SVD-only scores. Batch size 1 isolates
the loss-only algorithm from candidate batching. Larger batches multiply base
OTF storage (about 0.507 GB per candidate in the 128/24/40 example). A conservative
estimate of free GPU/pool memory caps the requested batch size; it is not a
peak-memory guarantee. OOM/device errors propagate. The workspace is still
single-device and cannot be called concurrently from multiple threads.

info['initialization'] additionally reports method, effective batch_size,
refined_candidates and linear_frequencies/svd_fallback_frequencies. Progress
counts completed candidates in chunks. screen_losses clears fit caches and
restores reuse_svd even when scoring or progress callbacks fail.

Portable checks compare linear/SVD scores against full fits, ill-conditioned
operators at the guard boundary, small rho and left null spaces, zero-data and
near-tied candidates, seeded selection/optimizer results, callback progress and
failure cleanup. Actual new-path CUDA speedup has not been measured yet.

The isolated benchmark compares identical 101-point pools using original serial
GPU fitting and loss-only batches 1, 4, and 8. It checks every score and selected
index against serial GPU fitting, checks three points on CPU, warms each path,
interleaves repeats and synchronizes GPU work. It avoids repeating the roughly
400-second CPU optimization just to test GPU screening.

```sh
REQUIRE_CUDA=1 python -m unittest discover -s tests -p test_candidate_screening.py -v
python -u -m optimization.performance.benchmark_screening --size 128 --pupil-size 128 --images 24 --planes 40 --random-starts 100 --candidate-batches 1 4 8 --repeats 3 > screening.json
```

For an unattended Bouchet run, submit from the updated repository root:

```sh
mkdir -p logs
sbatch --qos=normal optimization/performance/bouchet_screening.sbatch
```

The script uses account pi_cr876 and any one H100/H200/B200, runs the new GPU
checks first, and saves JSON/GPU/software/revision records in
results/bouchet/screening/JOBID/. For full optimizer comparisons, the existing
benchmark_cuda accepts --screening-method and --screening-batch-size; old
behavior is reproducible with --screening-method serial.
