"""Headless cluster entry point: python -m optimization.run_cuda --help."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np

from optimization.reconstruction_cuda import CUDAReconstruction, _cupy
from utils.psf import Microscope, Arbitrary_Grid
from utils.zernike import Aberration


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fit a 3D acquisition on one NVIDIA GPU.')
    parser.add_argument('data', type=Path, help='NPZ: images, focal_z_levels, sample_z_levels')
    parser.add_argument('config', type=Path, help='JSON optical/solver configuration; see CUDA.md')
    parser.add_argument('output', type=Path, help='Destination NPZ (must not already exist)')
    parser.add_argument('--device', type=int, default=0, help='CUDA-visible device index (default 0)')
    parser.add_argument('--benchmark', type=int, default=0, metavar='REPEATS',
                        help='Time warmed derivative evaluations before fitting')
    args = parser.parse_args(argv)
    if args.benchmark < 0:
        parser.error('--benchmark must be nonnegative')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    config = json.loads(args.config.read_text())
    microscope = Microscope(**config['microscope'])
    grid = Arbitrary_Grid(**config['grid'])
    modes = config['modes']
    strengths = config.get('initial_strengths', np.zeros(len(modes)))
    diversities = config.get('diversities')
    if diversities is not None:
        diversities = [Aberration(a['modes'], a['strengths']) for a in diversities]
    cp = _cupy()
    with cp.cuda.Device(args.device):
        cp.show_config()
        properties = cp.cuda.runtime.getDeviceProperties(args.device)
        name = properties['name']
        if isinstance(name, bytes):
            name = name.decode()
        print(f'GPU: {name}', flush=True)
        start = time.perf_counter()
        with np.load(args.data, allow_pickle=False) as data:
            workspace = CUDAReconstruction(
                microscope, grid, data['images'], data['focal_z_levels'],
                data['sample_z_levels'], modes, config['rho'], diversities,
                **config.get('solver', {}))
        cp.cuda.get_current_stream().synchronize()
        setup_seconds = time.perf_counter()-start
        timings = []
        if args.benchmark:
            workspace.derivatives(strengths)  # Warm cuBLAS/cuFFT/cuSOLVER and kernels.
            cp.cuda.get_current_stream().synchronize()
            for _ in range(args.benchmark):
                start = time.perf_counter()
                workspace.derivatives(strengths)
                cp.cuda.get_current_stream().synchronize()
                timings.append(time.perf_counter()-start)
            print(f'Warmed derivative median: {np.median(timings):.6f} s', flush=True)
        start = time.perf_counter()
        estimated, info = workspace.optimize(strengths, **config.get('optimizer', {}))
        sample = workspace.sample(estimated.strengths)
        cp.cuda.get_current_stream().synchronize()
        solve_seconds = time.perf_counter()-start
        metadata = dict(gpu=name, cupy=cp.__version__, numpy=np.__version__,
                        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                        cuda_driver=cp.cuda.runtime.driverGetVersion(),
                        setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                        success=info['success'], message=info['message'],
                        iterations=info['iterations'], config=config)
        # Exclusive creation also guards against another job choosing this path.
        with args.output.open('xb') as output:
            np.savez_compressed(
                output, sample=sample.image_mask, sample_z_levels=workspace.sample_z,
                modes=estimated.modes, strengths=estimated.strengths,
                loss=info['loss'], gradient=info['gradient'], hessian=info['hessian'],
                history_loss=[h['loss'] for h in info['history']],
                history_strengths=[h['strengths'] for h in info['history']],
                derivative_seconds=timings, metadata=json.dumps(metadata))
        print(f'Saved {args.output}; success={info["success"]}', flush=True)
    return 0 if info['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
