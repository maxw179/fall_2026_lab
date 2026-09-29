# CUDA reconstruction

`reconstruction_cuda.py` implements the existing unconstrained, regularized
3D solver on one NVIDIA GPU using CuPy. The CPU implementation remains the
reference. Local NumPy-backend checks cover the new algorithms; real CUDA
execution and speedups still need validation on a GPU node.

## Installation and validation

Use a Linux GPU environment with NumPy, SciPy and a CuPy wheel matching the
cluster's CUDA installation. For a CUDA 12 environment:

```sh
python -m pip install numpy scipy cupy-cuda12x
REQUIRE_CUDA=1 python -m unittest discover -s tests -p 'test_reconstruction_cuda.py' -v
```

Choose the wheel using the [official CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
and your cluster's driver/toolkit configuration. Install only one CuPy package.
The ordinary test suite skips CUDA tests without a GPU; `REQUIRE_CUDA=1` makes
missing CUDA an error. The tests exercise the same workspace with NumPy and
CuPy and compare against the independent CPU solver, including finite differences.

## Notebook use

The four function names and existing positional arguments match the CPU module:

```python
from optimization import reconstruction_cuda as reconstruction

estimated, info = reconstruction.optimize_aberration_3D(
    microscope, grid, I_noisy, focal_z_levels, sample_z_levels,
    modes, rho, diversities=diversities,
    mode='vector', psf_batch_size=16, frequency_batch_size=1024,
)
```

For repeated evaluations and final reconstruction, reuse a workspace:

```python
from optimization.reconstruction_cuda import CUDAReconstruction

solver = CUDAReconstruction(
    microscope, grid, I_noisy, focal_z_levels, sample_z_levels,
    modes, rho, diversities=diversities,
    psf_batch_size=16, frequency_batch_size=1024,
)
estimated, info = solver.optimize(initial_strengths=None)
sample, diagnostics = solver.sample(estimated.strengths, return_info=True)
# Optional: return a CuPy array rather than copying into Image_Mask_3D.
volume_gpu = solver.sample(estimated.strengths, return_device=True)
```

Inputs may be NumPy or CuPy image arrays. Coefficients and returned optimizer
statistics are small host arrays; the default sample is an `Image_Mask_3D`.
A workspace snapshots geometry, acquisition data, modes and diversities. Create
a new one after changing those inputs. Create and use it on the same current
CUDA device; it is not thread-safe.

## What runs on the GPU

- Batched scalar/vector pupil propagation and multiphoton PSFs.
- Batched periodic wrapping and real FFTs, including even-axis boundary aliases.
- Regularized frequency-batched SVD sample fits.
- Analytic first/second PSF derivatives and exact reduced Hessian contractions.
- Reconstruction inverse FFTs and residual diagnostics.

Observations, propagators, diversity phases and Zernike bases remain on device
across Newton steps and backtracking. Geometry/basis construction happens once
on the CPU. Each derivative evaluation shares one SVD per frequency between the
sample estimate and Hessian response, including null-space directions when P>K.
Second derivative OTFs are streamed by PSF batch rather than stored as M² stacks.
The most recent fit is retained for repeated loss/sample requests. Accepted
line-search trials still regenerate fields and factors for the next derivative
evaluation; no large factorization cache is retained across iterations.

The small Newton eigensolve, stopping rules and backtracking control run on the
CPU. Float64/complex128 preserve the raw multiphoton intensity scale and small
regularization behavior. Float32, mixed precision, multi-GPU sharding and custom
CUDA kernels are not implemented. Double-precision throughput depends on GPU
model; GPU speedups should be measured rather than assumed.

## Memory and timing

For K acquisitions, P sample planes, M modes and padded shape (Nx, Ny), let
Q = Nx * (Ny//2+1). OTF storage is `16*K*P*Q` bytes; first derivatives require
an additional `16*M*K*P*Q` bytes. This excludes pupil fields, observations,
reconstruction, FFT workspaces, SVD workspaces and CuPy's memory pool.

For example, K=24, P=40, M=2, padded shape 256×256 requires about 1.42 GiB for
S and dS alone. Peak memory is higher. Lower `psf_batch_size` to reduce pupil,
field and FFT temporary storage; lower `frequency_batch_size` to reduce SVD and
Hessian workspace. Neither setting shrinks the full S/dS arrays. If those arrays
do not fit, reduce problem size or use a larger-memory GPU. No silent CPU fallback
or automatic out-of-core mode is provided.

[CuPy's SVD](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.svd.html)
selects algorithms according to matrix dimensions; tune frequency batches on the
target GPU. Timing must include synchronization and warmup, as described in the
[CuPy performance guide](https://docs.cupy.dev/en/stable/user_guide/performance.html).
The runner below does both and reports derivative wall time including host returns.

## Remote batch jobs

Export arrays from the notebook (units remain mm and waves):

```python
np.savez_compressed('acquisition.npz', images=I_noisy,
                    focal_z_levels=focal_z_levels,
                    sample_z_levels=sample_z_levels)
```

Create `config.json` using the actual experiment values. This is the schema;
the numbers below are illustrative, especially `rho`:

```json
{
  "microscope": {
    "N_order": 3, "lambd": 0.0013, "n": 1.333, "num_apt": 1.05,
    "f": 7.2, "mag": 4, "w_0": 3.5, "L_bfp": 15.12, "grid_bfp": 64
  },
  "grid": {
    "L_ffp_x": 0.008, "L_ffp_y": 0.008,
    "grid_ffp_x": 32, "grid_ffp_y": 32,
    "x_offset": 0, "y_offset": 0, "z_level": 0
  },
  "modes": [[0, 4], [-2, 2]],
  "rho": 1e48,
  "initial_strengths": [0, 0],
  "solver": {
    "mode": "vector", "padding": null,
    "psf_batch_size": 16, "frequency_batch_size": 1024
  },
  "optimizer": {"max_iterations": 50, "verbose": true}
}
```

An optional `diversities` array contains one
`{"modes": [[0,4]], "strengths": [-0.05]}` object per acquired image, in acquisition
order. Omitting it means zero diversity, not the notebook's experimental diversity.

Copy the repository, NPZ and JSON to the cluster. From the repository root,
inside an allocated GPU environment:

```sh
python -m optimization.run_cuda acquisition.npz config.json result.npz --benchmark 5
```

The runner prints CUDA/GPU configuration, optionally benchmarks warmed derivative
evaluations, fits coefficients and saves the sample, derivatives, optimization
history, timings and configuration metadata. Outputs contain no pickled objects.
Existing outputs are refused. Exit code 2 means the optimizer stopped without
convergence; its result and status are still saved. This is a local optimizer.

A Slurm template (adapt partition/account/environment activation to your site):

```sh
#!/bin/bash
#SBATCH --job-name=reconstruct
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=reconstruct-%j.log
set -euo pipefail
# Activate your prepared Python/CUDA environment here.
# Submit from the repository root.
REQUIRE_CUDA=1 python -m unittest discover -s tests -p 'test_reconstruction_cuda.py' -v
python -m optimization.run_cuda acquisition.npz config.json "result-${SLURM_JOB_ID}.npz" --benchmark 5
```

Use the scheduler's GPU allocation/`CUDA_VISIBLE_DEVICES`; device 0 is the first
visible GPU. Independent acquisitions or starting points can run as separate jobs.
This runner does not distribute a single solve across GPUs or submit jobs itself.
