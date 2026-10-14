"""Compare identical random pools with serial and batched CUDA loss scoring.

Example: python -m optimization.performance.benchmark_screening --size 128
--pupil-size 128 --images 24 --planes 40 --random-starts 100.
"""
import argparse
import json
import sys
from time import perf_counter
import numpy as np

from optimization._random_starts import random_aberration_starts
from optimization._reconstruction_cpu import CPUReconstructionWorkspace
from optimization.reconstruction_cuda import CUDAReconstruction, _cupy
from optimization.performance.benchmark_cuda import _modes
from utils.psf import Microscope, Arbitrary_Grid, _convolution_grid
from utils.zernike import Aberration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--pupil-size', type=int, default=128)
    parser.add_argument('--images', type=int, default=24)
    parser.add_argument('--planes', type=int, default=40)
    parser.add_argument('--modes', type=int, default=2)
    parser.add_argument('--random-starts', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--candidate-batches', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--frequency-batch-size', type=int, default=4096)
    parser.add_argument('--psf-batch-size', type=int, default=8)
    parser.add_argument('--include-svd', action='store_true', help='Also time loss-only SVD with candidate batching.')
    options = parser.parse_args()
    sizes = ('size', 'pupil_size', 'images', 'planes', 'modes', 'repeats',
             'frequency_batch_size', 'psf_batch_size')
    if any(getattr(options, key) < 1 for key in sizes) or any(n < 1 for n in options.candidate_batches):
        parser.error('sizes, repeats and candidate batches must be positive')
    if options.random_starts < 0:
        parser.error('random-starts must be nonnegative')
    cp = _cupy()
    synchronize = cp.cuda.get_current_stream().synchronize
    m = Microscope(3, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, options.pupil_size)
    g = Arbitrary_Grid(.006, .006, options.size, options.size, 0, 0, 0)
    strengths = np.zeros(options.modes)
    strengths[:min(2, options.modes)] = [-.07, .03][:min(2, options.modes)]
    a = Aberration(_modes(options.modes), strengths)
    data = np.random.default_rng(3).random((options.images, options.size, options.size))
    rho = float(m.compute_PSF(_convolution_grid(g), a, 'vector')[2].sum()**2 * .03)
    args = (m, g, data, np.linspace(0, .002, options.images),
            np.linspace(0, .002, options.planes), a.modes, rho)
    batches = dict(frequency_batch_size=options.frequency_batch_size,
                   psf_batch_size=options.psf_batch_size, reuse_svd=False)
    candidates = np.vstack((np.zeros((1, options.modes)), random_aberration_starts(
        a.modes, m.alpha, options.random_starts, seed=options.seed)))
    workspaces = dict(serial=CUDAReconstruction(*args, **batches, svd_method='qr', screening_method='serial'))
    for size in dict.fromkeys(options.candidate_batches):
        workspaces[f'linear_batch{size}'] = CUDAReconstruction(*args, **batches,
            svd_method='qr', screening_method='linear', screening_batch_size=size)
    if options.include_svd:
        workspaces['svd_batch4'] = CUDAReconstruction(*args, **batches,
            svd_method='qr', screening_method='svd', screening_batch_size=4)
    print('Warming serial CUDA and computing reference scores...', file=sys.stderr, flush=True)
    reference = workspaces['serial'].screen_losses(candidates)
    synchronize()
    cpu = CPUReconstructionWorkspace(*args, **batches)
    ids = np.unique(np.linspace(0, len(candidates)-1, min(3, len(candidates)), dtype=int))
    expected = cpu.screen_losses(candidates[ids])
    np.testing.assert_allclose(reference[ids], expected, rtol=2e-8, atol=2e-9*np.max(abs(expected)))
    selected = int(np.argmin(reference))
    for name, workspace in workspaces.items():
        if name == 'serial':
            continue
        print(f'Warming and checking {name}...', file=sys.stderr, flush=True)
        scores = workspace.screen_losses(candidates)
        synchronize()
        np.testing.assert_allclose(scores, reference, rtol=2e-8, atol=2e-9*np.max(abs(reference)))
        if int(np.argmin(scores)) != selected:
            raise AssertionError(f'{name} selected a different candidate')
    raw = {name: [] for name in workspaces}
    order = np.random.default_rng(0)
    for repeat in range(options.repeats):
        for name in order.permutation(list(workspaces)):
            print(f'Timing {name} ({repeat+1}/{options.repeats})...', file=sys.stderr, flush=True)
            synchronize()
            started = perf_counter()
            scores = workspaces[name].screen_losses(candidates)
            synchronize()
            raw[name].append(perf_counter()-started)
            np.testing.assert_allclose(scores, reference, rtol=2e-8, atol=2e-9*np.max(abs(reference)))
            if int(np.argmin(scores)) != selected:
                raise AssertionError(f'{name} selected a different candidate')
    serial = float(np.median(raw['serial']))
    timings = {name: dict(seconds=float(np.median(values)),
                         speedup=serial/float(np.median(values)), samples=values,
                         summary=workspaces[name]._last_screening_info.copy())
               for name, values in raw.items()}
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(json.dumps(dict(problem=vars(options), candidate_count=len(candidates),
        candidate_strengths=candidates.tolist(), reference_losses=reference.tolist(),
        selected_index=selected, timings=timings, parity='passed',
        hardware=dict(gpu=name, cupy=cp.__version__, numpy=np.__version__,
                      vram_bytes=int(props['totalGlobalMem']),
                      cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                      driver=cp.cuda.runtime.driverGetVersion())), indent=2))


if __name__ == '__main__':
    main()
