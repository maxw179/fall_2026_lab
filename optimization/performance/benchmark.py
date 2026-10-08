"""Profile CPU reference versus the CUDA workspace's NumPy numerical-test backend.

Run from the repository root: python -m optimization.performance.benchmark
This measures CPU execution only; _xp=np is not a CUDA performance estimate.
"""
import argparse
import cProfile
import io
import json
import platform
import pstats
from pathlib import Path
from time import perf_counter

import numpy as np
from optimization.performance import reconstruction_reference as cpu
from optimization.reconstruction_cuda import CUDAReconstruction
from utils.psf import Microscope, Centered_Square_Grid, bead_img_3D
from utils.zernike import Aberration

MODES = [[0, 4], [-2, 2], [2, 2], [-3, 3], [3, 3], [-4, 4], [-2, 4],
         [2, 4], [4, 4], [-5, 5], [-3, 5], [-1, 5], [1, 5], [3, 5], [5, 5]]


def profile_call(fn):
    profile = cProfile.Profile()
    start = perf_counter()
    result = profile.runcall(fn)
    elapsed = perf_counter() - start
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(20)
    return result, elapsed, stream.getvalue()


def gradient_only(workspace, strengths):
    """Prototype envelope gradient: refit the sample, omit all Hessian work."""
    S, dS = workspace._otfs(strengths, True)
    F, residual, loss, _, _ = workspace._fit(S)
    q = np.einsum('mkpf,pf->mkf', dS, F)
    gradient = -2 * np.real(np.einsum('f,kf,mkf->m', workspace.weights,
                                    residual.conj(), q))
    return loss, gradient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--pupil', type=int, default=64)
    parser.add_argument('--sample-planes', type=int, default=40)
    parser.add_argument('--focal-planes', type=int, default=8)
    parser.add_argument('--modes', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('optimization/performance/baseline.json'))
    args = parser.parse_args()
    m = Microscope(3, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, args.pupil)
    g = Centered_Square_Grid(.0065, args.resolution, 0)
    sample_z = -.002 + np.arange(args.sample_planes) * (.004 / args.sample_planes)
    focal = -.002 + np.arange(args.focal_planes) * (.004 / args.focal_planes)
    focus = np.tile(focal, 3)
    div = [Aberration([[0, 4]], [v]) for v in [0, -.05, .05] for _ in focal]
    modes = MODES[:args.modes]
    truth = Aberration(modes, [-.15] + [.02] * (len(modes) - 1))
    sample = bead_img_3D(g, sample_z, [0], [0], [0], [.001])
    _, _, _, images = m.compute_image_3D(sample, truth, focus, diversities=div)
    strengths = np.zeros(len(modes))
    point = Aberration(modes, strengths)
    common = dict(microscope=m, grid=g, images=images, focal_z_levels=focus,
                  sample_z_levels=sample_z, rho=1e48, diversities=div)
    output = dict(platform=platform.platform(), numpy=np.__version__, config=vars(args).copy())
    output['config']['output'] = str(args.output)
    for label, fn in [
        ('reference_loss', lambda: cpu.evaluate_loss_3D(aberration=point, **common)),
        ('reference_derivatives', lambda: cpu.evaluate_loss_derivatives_3D(aberration=point, **common)),
    ]:
        result, seconds, profile = profile_call(fn)
        if label.endswith('derivatives'):
            reference = result
        samples = []
        for _ in range(args.repeat):
            start = perf_counter(); fn(); samples.append(perf_counter() - start)
        output[label] = dict(profile_seconds=seconds, seconds=samples, profile=profile)
        print(label, samples, flush=True)
    for batch in [1, 16]:
        start = perf_counter()
        workspace = CUDAReconstruction(modes=modes, psf_batch_size=batch, _xp=np, **common)
        setup = perf_counter() - start
        result, seconds, profile = profile_call(lambda: workspace.derivatives(strengths))
        for ref, actual in zip(reference, result):
            np.testing.assert_allclose(np.asarray(actual) / reference[0], np.asarray(ref) / reference[0],
                                       rtol=2e-8, atol=2e-9)
        samples = []
        for _ in range(args.repeat):
            start = perf_counter(); workspace.derivatives(strengths); samples.append(perf_counter() - start)
        output[f'workspace_batch_{batch}'] = dict(setup_seconds=setup, profile_seconds=seconds,
                                                 seconds=samples, profile=profile, parity=True)
        print('workspace', batch, samples, flush=True)
    result, seconds, profile = profile_call(lambda: gradient_only(workspace, strengths))
    for ref, actual in zip(reference[:2], result):
        np.testing.assert_allclose(np.asarray(actual) / reference[0], np.asarray(ref) / reference[0],
                                   rtol=2e-8, atol=2e-9)
    samples = []
    for _ in range(args.repeat):
        start = perf_counter(); gradient_only(workspace, strengths)
        samples.append(perf_counter() - start)
    output['gradient_only'] = dict(seconds=samples, profile_seconds=seconds, profile=profile,
                                   parity=True, note='Per-evaluation timing, not optimizer convergence')
    print('gradient_only', samples, flush=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
