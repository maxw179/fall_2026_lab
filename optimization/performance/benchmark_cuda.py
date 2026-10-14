"""Compare warmed CPU/CUDA operations: python -m optimization.performance.benchmark_cuda.

Reports full host-visible wall time. Clears fit caches to measure actual work.
"""
import argparse
import json
import sys
from contextlib import contextmanager
from time import perf_counter
import numpy as np
from optimization._reconstruction_cpu import CPUReconstructionWorkspace
from optimization.reconstruction_cuda import CUDAReconstruction, _cupy
from utils.psf import Microscope, Arbitrary_Grid, _convolution_grid
from utils.zernike import Aberration


@contextmanager
def _stage_profile(workspace, synchronize):
    """Diagnostic timing with forced stage synchronization, outside normal timings."""
    methods = ('_get_fit', '_build_otfs', '_propagate', '_transform', '_solve',
               '_svd', '_second_psf_hessian')
    originals, records, stack = {}, {}, []

    def wrap(name, fn):
        def measured(*args, **kwargs):
            synchronize()
            entry = [perf_counter(), 0.]
            stack.append(entry)
            try:
                return fn(*args, **kwargs)
            finally:
                synchronize()
                duration = perf_counter() - entry[0]
                stack.pop()
                record = records.setdefault(name, dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))
                record['calls'] += 1
                record['inclusive_seconds'] += duration
                record['exclusive_seconds'] += duration - entry[1]
                if stack:
                    stack[-1][1] += duration
        return measured

    for name in methods:
        originals[name] = (name in workspace.__dict__, getattr(workspace, name))
        setattr(workspace, name, wrap(name, originals[name][1]))
    try:
        yield records
    finally:
        for name, (was_local, fn) in originals.items():
            if was_local:
                setattr(workspace, name, fn)
            else:
                delattr(workspace, name)


def _modes(count):
    modes = [[0, 4], [-2, 2]]
    radial = 2
    while len(modes) < count:
        for azimuth in range(-radial, radial + 1, 2):
            candidate = [azimuth, radial]
            if candidate not in modes:
                modes.append(candidate)
        radial += 1
    return modes[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--pupil-size', type=int, default=64)
    parser.add_argument('--images', type=int, default=6)
    parser.add_argument('--planes', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--modes', type=int, default=2)
    parser.add_argument('--cuda-svd', choices=('direct', 'qr', 'compare'), default='direct',
                        help='QR reduces eligible rectangular operators to a <=32 square SVD; compare times both.')
    parser.add_argument('--psf-batch-size', type=int, default=8)
    parser.add_argument('--derivative-batch-size', type=int, default=8)
    parser.add_argument('--frequency-batch-size', type=int, default=4096)
    parser.add_argument('--profile', action='store_true', help='Separately time synchronized derivative stages.')
    parser.add_argument('--optimize', action='store_true', help='Time and check a full Newton run per backend.')
    parser.add_argument('--max-iterations', type=int, default=5)
    parser.add_argument('--random-starts', type=int, default=0,
                        help='Screen this many starts during the optional full optimizer measurement.')
    parser.add_argument('--screening-method', choices=('linear', 'svd', 'serial'), default='linear')
    parser.add_argument('--screening-batch-size', type=int, default=4)
    options = parser.parse_args()
    sizes = ('size', 'pupil_size', 'images', 'planes', 'repeats', 'modes',
             'psf_batch_size', 'derivative_batch_size', 'frequency_batch_size', 'max_iterations',
             'screening_batch_size')
    if any(getattr(options, key) < 1 for key in sizes):
        parser.error('all sizes and repeats must be positive')
    if options.random_starts < 0:
        parser.error('random-starts must be nonnegative')
    cp = _cupy()
    m = Microscope(3, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, options.pupil_size)
    g = Arbitrary_Grid(.006, .006, options.size, options.size, 0, 0, 0)
    strengths = np.zeros(options.modes)
    strengths[:min(2, options.modes)] = [-.07, .03][:min(2, options.modes)]
    a = Aberration(_modes(options.modes), strengths)
    data = np.random.default_rng(3).random((options.images, options.size, options.size))
    rho = float(m.compute_PSF(_convolution_grid(g), a, 'vector')[2].sum()**2 * .03)
    args = (m, g, data, np.linspace(0, .002, options.images),
            np.linspace(0, .002, options.planes), a.modes, rho)
    batches = dict(psf_batch_size=options.psf_batch_size,
                   derivative_batch_size=options.derivative_batch_size,
                   frequency_batch_size=options.frequency_batch_size)
    workspaces = {'cpu': CPUReconstructionWorkspace(*args, **batches)}
    screening_options = dict(screening_method=options.screening_method,
                             screening_batch_size=options.screening_batch_size)
    if options.cuda_svd == 'compare':
        workspaces.update({f'cuda_{method}': CUDAReconstruction(*args, svd_method=method,
                           **batches, **screening_options)
                           for method in ('direct', 'qr')})
    else:
        workspaces['cuda'] = CUDAReconstruction(*args, svd_method=options.cuda_svd,
                                               **batches, **screening_options)
    synchronize = cp.cuda.get_current_stream().synchronize
    # Check multiple phases before timing, including derivatives from a cached loss fit.
    print('Checking CPU/CUDA loss, sample, gradient and Hessian agreement...', file=sys.stderr, flush=True)
    for point in (np.zeros(options.modes), a.strengths / 2, a.strengths):
        cpu = workspaces['cpu']
        cpu.clear_cache()
        expected = cpu.derivatives(point)
        ref_sample = cpu.estimate_sample(point).image_mask
        for backend, workspace in workspaces.items():
            if backend == 'cpu':
                continue
            workspace.clear_cache()
            workspace.loss(point)
            actual = workspace.derivatives(point)
            for target, value in zip(expected, actual):
                np.testing.assert_allclose(np.asarray(value)/expected[0], np.asarray(target)/expected[0],
                                           rtol=2e-8, atol=2e-9)
            np.testing.assert_allclose(workspace.estimate_sample(point).image_mask, ref_sample,
                                       rtol=2e-8, atol=2e-9*np.max(abs(ref_sample)))
    results, raw_timings, profiles, optimizations = {}, {}, {}, {}
    timing_order = np.random.default_rng(0)
    for operation in ('loss', 'loss_gradient', 'derivatives', 'estimate_sample'):
        elapsed = {backend: [] for backend in workspaces}
        for backend, workspace in workspaces.items():
            print(f'Warming {backend} {operation}...', file=sys.stderr, flush=True)
            fn = getattr(workspace, operation)
            workspace.clear_cache()
            fn(a.strengths)
            synchronize()
        for repeat in range(options.repeats):
            for backend in timing_order.permutation(list(workspaces)):
                print(f'Timing {backend} {operation} ({repeat+1}/{options.repeats})...',
                      file=sys.stderr, flush=True)
                workspace = workspaces[backend]
                workspace.clear_cache()
                synchronize()
                start = perf_counter()
                getattr(workspace, operation)(a.strengths)
                synchronize()
                elapsed[backend].append(perf_counter() - start)
        timings = {backend: float(np.median(samples)) for backend, samples in elapsed.items()}
        raw_timings[operation] = elapsed
        speedups = {key: timings['cpu']/value for key, value in timings.items() if key != 'cpu'}
        results[operation] = dict(**timings)
        if 'cuda' in timings:
            results[operation]['speedup'] = speedups['cuda']
        else:
            results[operation]['speedups'] = speedups
    for backend, workspace in workspaces.items():
        if options.profile:
            workspace.clear_cache()
            if backend != 'cpu':
                workspace.svd_counts = dict(qr=0, direct=0)
            sync = (lambda: None) if backend == 'cpu' else synchronize
            with _stage_profile(workspace, sync) as stages:
                sync()
                start = perf_counter()
                workspace.derivatives(a.strengths)
                sync()
                profiles[backend] = dict(total_seconds=perf_counter()-start, stages=stages,
                                         memory=workspace.memory_summary())
            if backend != 'cpu':
                profiles[backend]['svd_frequencies'] = workspace.svd_counts.copy()
        if options.optimize:
            workspace.clear_cache()
            synchronize()
            start = perf_counter()
            estimated, info = workspace.optimize(max_iterations=options.max_iterations,
                num_random_starts=options.random_starts, random_start_seed=0,
                show_initialization_progress=False)
            synchronize()
            optimizations[backend] = dict(seconds=perf_counter()-start, strengths=estimated.strengths.tolist(),
                loss=info['loss'], success=bool(info['success']), message=info['message'],
                history_steps=len(info['history']),
                initialization_seconds=info['initialization']['seconds'] if info['initialization'] else 0.,
                initialization_method=info['initialization']['method'] if info['initialization'] else None,
                initialization_batch_size=info['initialization']['batch_size'] if info['initialization'] else None,
                refined_candidates=info['initialization']['refined_candidates'] if info['initialization'] else 0)
    if optimizations:
        ref = optimizations['cpu']
        for backend, result in optimizations.items():
            if backend != 'cpu':
                np.testing.assert_allclose(result['strengths'], ref['strengths'], rtol=2e-7, atol=2e-8)
                np.testing.assert_allclose(result['loss'], ref['loss'], rtol=2e-8)
                result['speedup'] = ref['seconds']/result['seconds']
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    print(json.dumps(dict(problem=vars(options), timings=results, raw_timings=raw_timings,
        profiles=profiles, optimizations=optimizations, parity='passed',
        hardware=dict(gpu=name, vram_bytes=int(props['totalGlobalMem']), cupy=cp.__version__,
                      numpy=np.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                      driver=cp.cuda.runtime.driverGetVersion())), indent=2))


if __name__ == '__main__':
    main()
