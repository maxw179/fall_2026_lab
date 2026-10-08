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
