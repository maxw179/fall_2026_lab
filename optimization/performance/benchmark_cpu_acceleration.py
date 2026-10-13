"""Benchmark CPU propagation, FFT workers, SVD workers, and parallel scans.

Run with BLAS thread settings selected before Python starts. Timings compare
against the current dense workspace, not the older reference implementation.
"""
import argparse
import json
from time import perf_counter
from pathlib import Path
import numpy as np
from optimization.reconstruction import CPUReconstructionWorkspace
from utils.psf import Microscope, Arbitrary_Grid, _convolution_grid
from utils.zernike import Aberration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--pupil', type=int, default=64)
    parser.add_argument('--images', type=int, default=6)
    parser.add_argument('--planes', type=int, default=8)
    parser.add_argument('--modes', type=int, choices=[1, 3, 6], default=3)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.size, args.pupil) < 2 or min(args.images, args.planes, args.repeat) < 1:
        parser.error('grid sizes must be >=2 and counts positive')
    m = Microscope(3, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, args.pupil)
    g = Arbitrary_Grid(.006, .006, args.size, args.size, 0, 0, 0)
    modes = [[0, 4], [-2, 2], [2, 2], [-3, 3], [3, 3], [0, 6]][:args.modes]
    a = Aberration(modes, [-.07]+[.03]*(args.modes-1))
    data = np.random.default_rng(3).random((args.images, args.size, args.size))
    rho = float(m.compute_PSF(_convolution_grid(g), a, 'vector')[2].sum()**2*.03)
    common = (m, g, data, np.linspace(0, .002, args.images),
              np.linspace(0, .002, args.planes), modes, rho)
    variants = dict(dense=dict(propagation='dense'),
                    dense_svd2=dict(propagation='dense', cpu_workers=2),
                    dense_svd4=dict(propagation='dense', cpu_workers=4),
                    dense_fft2=dict(propagation='dense', fft_workers=2),
                    dense_batch4=dict(propagation='dense', psf_batch_size=4),
                    dense_batch16=dict(propagation='dense', psf_batch_size=16),
                    zoom=dict(propagation='zoom'),
                    zoom_fft2=dict(propagation='zoom', fft_workers=2),
                    zoom_fft4=dict(propagation='zoom', fft_workers=4),
                    zoom_fft2_svd2=dict(propagation='zoom', fft_workers=2, cpu_workers=2),
                    zoom_svd2=dict(propagation='zoom', cpu_workers=2),
                    zoom_svd4=dict(propagation='zoom', cpu_workers=4))
    workspaces = {name: CPUReconstructionWorkspace(*common, **options)
                  for name, options in variants.items()}
    expected = workspaces['dense'].derivatives(a.strengths)
    scale = expected[0]
    results = {}
    for name, workspace in workspaces.items():
        actual = workspace.derivatives(a.strengths)
        error = 0.
        for target, value in zip(expected, actual):
            np.testing.assert_allclose(np.asarray(value)/scale, np.asarray(target)/scale,
                                       rtol=2e-8, atol=2e-9)
            error = max(error, float(np.max(abs(np.asarray(value)-target))/scale))
        results[name] = dict(options=variants[name], max_scaled_error=error, times={})
    # Interleave variants to limit timing bias from CPU frequency/thermal changes.
    rng = np.random.default_rng(5)
    for operation in ('loss', 'derivatives'):
        timings = {name: [] for name in workspaces}
        for _ in range(args.repeat):
            for name in rng.permutation(list(workspaces)):
                workspace = workspaces[name]
                workspace.clear_cache()
                start = perf_counter()
                getattr(workspace, operation)(a.strengths)
                timings[name].append(perf_counter()-start)
        baseline = float(np.median(timings['dense']))
        for name, values in timings.items():
            median = float(np.median(values))
            results[name]['times'][operation] = dict(seconds=median, speedup=baseline/median)
    points = [a.strengths*v for v in np.linspace(-1, 1, 8)]
    scan = workspaces['zoom']
    expected_scan = scan.losses(points)
    scans = {}
    for workers in (1, 2, 4):
        samples = []
        for _ in range(args.repeat):
            scan.clear_cache()
            start = perf_counter()
            actual = scan.losses(points, workers=workers)
            samples.append(perf_counter()-start)
            np.testing.assert_allclose(actual, expected_scan, rtol=2e-9)
        scans[workers] = float(np.median(samples))
    import os
    import platform
    import scipy
    payload = dict(platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__, problem={k: v for k, v in vars(args).items() if k != 'output'},
                   threads={k: os.environ.get(k) for k in
                            ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS']},
                   variants=results, scan_seconds=scans)
    report = json.dumps(payload, indent=2)+'\n'
    if args.output:
        args.output.write_text(report)
    print(report)


if __name__ == '__main__':
    main()
