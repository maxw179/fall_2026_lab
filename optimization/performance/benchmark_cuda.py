"""Compare warmed CPU/CUDA operations: python -m optimization.performance.benchmark_cuda.

Reports full host-visible wall time. Clears fit caches to measure actual work.
"""
import argparse
import json
from time import perf_counter
import numpy as np
from optimization._reconstruction_cpu import CPUReconstructionWorkspace
from optimization.reconstruction_cuda import CUDAReconstruction, _cupy
from utils.psf import Microscope, Arbitrary_Grid, _convolution_grid
from utils.zernike import Aberration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--pupil-size', type=int, default=64)
    parser.add_argument('--images', type=int, default=6)
    parser.add_argument('--planes', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    options = parser.parse_args()
    if any(v < 1 for v in vars(options).values()):
        parser.error('all sizes and repeats must be positive')
    cp = _cupy()
    m = Microscope(3, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, options.pupil_size)
    g = Arbitrary_Grid(.006, .006, options.size, options.size, 0, 0, 0)
    a = Aberration([[0, 4], [-2, 2]], [-.07, .03])
    data = np.random.default_rng(3).random((options.images, options.size, options.size))
    rho = float(m.compute_PSF(_convolution_grid(g), a, 'vector')[2].sum()**2 * .03)
    args = (m, g, data, np.linspace(0, .002, options.images),
            np.linspace(0, .002, options.planes), a.modes, rho)
    workspaces = {'cpu': CPUReconstructionWorkspace(*args), 'cuda': CUDAReconstruction(*args)}
    results = {}
    for operation in ('loss', 'loss_gradient', 'derivatives', 'estimate_sample'):
        timings = {}
        for backend, workspace in workspaces.items():
            fn = getattr(workspace, operation)
            workspace.clear_cache()
            fn(a.strengths)  # Warm FFT, cuSOLVER, allocation and kernel compilation.
            cp.cuda.get_current_stream().synchronize()
            elapsed = []
            for _ in range(options.repeats):
                workspace.clear_cache()
                cp.cuda.get_current_stream().synchronize()
                start = perf_counter()
                fn(a.strengths)
                cp.cuda.get_current_stream().synchronize()
                elapsed.append(perf_counter() - start)
            timings[backend] = float(np.median(elapsed))
        results[operation] = dict(**timings, speedup=timings['cpu']/timings['cuda'])
    expected = workspaces['cpu'].derivatives(a.strengths)
    actual = workspaces['cuda'].derivatives(a.strengths)
    for target, value in zip(expected, actual):
        np.testing.assert_allclose(np.asarray(value)/expected[0], np.asarray(target)/expected[0],
                                   rtol=2e-8, atol=2e-9)
    print(json.dumps(dict(problem=vars(options), timings=results, parity='passed'), indent=2))


if __name__ == '__main__':
    main()
