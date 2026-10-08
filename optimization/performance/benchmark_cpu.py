"""Benchmark exact-Hessian CPU improvements against the preserved CPU reference.

Run deliberately from the repository root; importing this module runs nothing.
OPENBLAS_NUM_THREADS=1 python -m optimization.performance.benchmark_cpu
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
from pathlib import Path
from time import perf_counter

import numpy as np

from optimization.performance import reconstruction_reference as reference
from optimization.reconstruction import CPUReconstructionWorkspace
from utils.psf import Microscope, Centered_Square_Grid, bead_img_3D
from utils.zernike import Aberration, zernike_RMS_difference


MODES = [[0, 4], [-2, 2], [2, 2], [-3, 3], [3, 3], [-4, 4], [-2, 4],
         [2, 4], [4, 4], [-5, 5], [-3, 5], [-1, 5], [1, 5], [3, 5], [5, 5]]
VARIANTS = {
    "geometry_and_batches": dict(reuse_svd=False, spatial_hessian=False,
                                  deduplicate_psfs=False),
    "shared_svd": dict(reuse_svd=True, spatial_hessian=False,
                       deduplicate_psfs=False),
    "spatial_hessian": dict(reuse_svd=True, spatial_hessian=True,
                            deduplicate_psfs=False),
    "all_improvements": dict(reuse_svd=True, spatial_hessian=True,
                             deduplicate_psfs=True),
}


def poisson_images(images, snr, rng):
    peak = float(np.max(images))
    if not np.all(np.isfinite(images)) or peak <= 0:
        raise ValueError("Synthetic images must have finite, positive signal; increase resolution or sample planes.")
    if np.min(images) < -1e-12*peak:
        raise ValueError("Synthetic images contain negative signal beyond convolution roundoff.")
    if snr is None:
        return images
    unit = np.clip(images / peak, 0, None)
    counts = snr**2 * unit.sum() / np.sum(unit**2)
    return rng.poisson(counts * unit) / counts * peak


def parity(expected, actual, rtol, atol):
    """Normalize every component by reference loss, matching optimizer scaling."""
    scale = max(abs(expected[0]), np.finfo(float).tiny)
    errors = {}
    for label, a, b in zip(["loss", "gradient", "hessian"], expected, actual):
        a, b = np.asarray(a)/scale, np.asarray(b)/scale
        np.testing.assert_allclose(b, a, rtol=rtol, atol=atol,
                                   err_msg=f"CPU workspace {label} mismatch")
        errors[label] = float(np.max(abs(b-a)))
    return errors


def timed(fn, prepare=None):
    if prepare is not None:
        prepare()
    start = perf_counter()
    fn()
    return perf_counter() - start


def paired_times(reference_fn, prototype_fn, repeat, rng,
                 reference_prepare=None, prototype_prepare=None):
    """Interleave reference/prototype calls in random order for each repeat."""
    values = {"reference": [], "prototype": []}
    calls = {
        "reference": (reference_fn, reference_prepare),
        "prototype": (prototype_fn, prototype_prepare),
    }
    for _ in range(repeat):
        for label in rng.permutation(["reference", "prototype"]):
            fn, prepare = calls[label]
            values[label].append(timed(fn, prepare))
    medians = {label: float(np.median(times)) for label, times in values.items()}
    return dict(seconds=values, median_seconds=medians,
                speedup=medians["reference"]/medians["prototype"])


def profile(fn):
    profiler = cProfile.Profile()
    profiler.runcall(fn)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(25)
    return stream.getvalue()


def optimization_summary(result, truth, alpha):
    estimated, info = result
    scale = max(abs(info["history"][0]["loss"]), np.finfo(float).tiny)
    return dict(success=bool(info["success"]), message=info["message"],
                iterations=int(info["iterations"]), loss=float(info["loss"]),
                strengths=estimated.strengths.tolist(),
                residual_wavefront_rms=float(zernike_RMS_difference(estimated, truth, alpha)),
                scaled_gradient_inf=float(np.linalg.norm(info["gradient"]/scale, ord=np.inf)),
                scaled_min_hessian_eigenvalue=float(np.linalg.eigvalsh(info["hessian"]/scale)[0]),
                history=[dict(iteration=int(h["iteration"]), loss=float(h["loss"]),
                              strengths=h["strengths"].tolist()) for h in info["history"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--pupil", type=int, default=64)
    parser.add_argument("--sample-planes", type=int, default=40)
    parser.add_argument("--focal-planes", type=int, default=8)
    parser.add_argument("--modes", type=int, choices=range(1, len(MODES)+1), default=1)
    parser.add_argument("--optical-mode", choices=["scalar", "vector"], default="vector")
    parser.add_argument("--rho", type=float, default=1e48)
    parser.add_argument("--snr", type=float, default=50,
                        help="Global Poisson SNR; zero requests clean images.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--frequency-batch", type=int, default=4096)
    parser.add_argument("--psf-batches", type=int, nargs="+", default=[8])
    parser.add_argument("--derivative-batch", type=int, default=8)
    parser.add_argument("--padding", type=int, default=None)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--recompute-fields", action="store_true",
                        help="Reduce memory by recomputing fields in the second-derivative pass.")
    parser.add_argument("--optimize", action="store_true", help="Also compare full Newton runs.")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-6)
    parser.add_argument("--profile", action="store_true", help="Profile separately from timed repeats.")
    parser.add_argument("--rtol", type=float, default=2e-8)
    parser.add_argument("--atol", type=float, default=2e-9)
    parser.add_argument("--output", type=Path,
                        default=Path("optimization/performance/cpu_benchmark.json"))
    args = parser.parse_args()
    for name in ["resolution", "pupil", "sample_planes", "focal_planes", "repeat",
                 "frequency_batch", "derivative_batch", "max_iterations"]:
        minimum = 2 if name == "resolution" else 1
        if getattr(args, name) < minimum:
            parser.error(f"--{name.replace('_', '-')} must be >= {minimum}")
    if any(batch < 1 for batch in args.psf_batches):
        parser.error("--psf-batches must be positive")
    if (not np.isfinite(args.rho) or args.rho <= 0
            or not np.isfinite(args.snr) or args.snr < 0):
        parser.error("rho must be finite and positive; snr must be finite and nonnegative")
    for name in ["rtol", "atol", "gradient_tolerance"]:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")

    microscope = Microscope(3, .0013, 1.333, 1.05, 7.2, 4, 3.5, 15.12, args.pupil)
    grid = Centered_Square_Grid(.0065, args.resolution, 0)
    sample_z = -.002 + np.arange(args.sample_planes)*(.004/args.sample_planes)
    focal = -.002 + np.arange(args.focal_planes)*(.004/args.focal_planes)
    focus = np.tile(focal, 3)
    diversities = [Aberration([[0, 4]], [v]) for v in [0, -.05, .05] for _ in focal]
    modes = MODES[:args.modes]
    truth = Aberration(modes, [-.15] + [.02]*(len(modes)-1))
    sample = bead_img_3D(grid, sample_z, [0], [0], [0], [.001])
    start = perf_counter()
    clean = microscope.compute_image_3D(sample, truth, focus, args.optical_mode, diversities)[3]
    images = poisson_images(clean, args.snr or None, np.random.default_rng(args.seed))
    synthesis_seconds = perf_counter() - start
    common = dict(microscope=microscope, grid=grid, images=images,
                  focal_z_levels=focus, sample_z_levels=sample_z,
                  rho=args.rho, diversities=diversities, mode=args.optical_mode,
                  frequency_batch_size=args.frequency_batch, padding=args.padding)
    probes = [np.zeros(len(modes)), .5*truth.strengths, truth.strengths.copy()]
    reference_probes = []
    for i, strengths in enumerate(probes):
        print(f"Reference agreement probe {i+1}/{len(probes)}", flush=True)
        reference_probes.append(reference.evaluate_loss_derivatives_3D(
            aberration=Aberration(modes, strengths), **common))
    point = Aberration(modes, probes[0])
    reference_loss = lambda: reference.evaluate_loss_3D(aberration=point, **common)
    reference_derivatives = lambda: reference.evaluate_loss_derivatives_3D(aberration=point, **common)
    config = vars(args).copy()
    config["output"] = str(args.output)
    numpy_config = io.StringIO()
    from contextlib import redirect_stdout
    with redirect_stdout(numpy_config):
        np.show_config()
    output = dict(platform=platform.platform(), numpy=np.__version__, config=config,
                  blas_configuration=numpy_config.getvalue(),
                  thread_environment={key: os.environ.get(key) for key in
                    ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS"]},
                  synthesis_seconds=synthesis_seconds,
                  note="Warm CPU timings; setup and synthesis excluded from call timings. "
                       "Agreement checks precede speed measurements. No GPU backend.",
                  variants={})
    rng = np.random.default_rng(args.seed + 1)
    optimizer_options = dict(max_iterations=args.max_iterations,
                             gradient_tolerance=args.gradient_tolerance, verbose=False)
    if args.optimize:
        print("Reference full Newton optimization", flush=True)
        start = perf_counter()
        result = reference.optimize_aberration_3D(modes=modes, **common, **optimizer_options)
        seconds = perf_counter() - start
        output["reference_optimization"] = optimization_summary(result, truth, microscope.alpha)
        output["reference_optimization"]["seconds"] = seconds
    if args.profile:
        output["reference_derivative_profile"] = profile(reference_derivatives)

    for batch in args.psf_batches:
        for name in args.variants:
            label = f"{name}_batch_{batch}"
            print(f"Preparing {label}", flush=True)
            start = perf_counter()
            workspace = CPUReconstructionWorkspace(modes=modes, psf_batch_size=batch,
                derivative_batch_size=args.derivative_batch,
                retain_fields=not args.recompute_fields, **common, **VARIANTS[name])
            setup_seconds = perf_counter() - start
            agreement = []
            for expected, strengths in zip(reference_probes, probes):
                workspace.clear_cache()
                actual = workspace.derivatives(strengths)
                errors = parity(expected, actual, args.rtol, args.atol)
                workspace.clear_cache()
                np.testing.assert_allclose(workspace.loss(strengths)/expected[0], 1,
                                           rtol=args.rtol, atol=args.atol)
                errors["after_loss_fit"] = parity(expected, workspace.derivatives(strengths),
                                                  args.rtol, args.atol)
                agreement.append(errors)
            memory = workspace.memory_summary()
            # Clear outside every measured call: repeated evaluations at the
            # same strengths must not turn into misleading constant-time hits.
            loss_times = paired_times(reference_loss, lambda: workspace.loss(probes[0]),
                args.repeat, rng, prototype_prepare=workspace.clear_cache)
            derivative_times = paired_times(reference_derivatives,
                lambda: workspace.derivatives(probes[0]), args.repeat, rng,
                prototype_prepare=workspace.clear_cache)

            def prepare_accepted_fit():
                workspace.clear_cache()
                workspace.loss(probes[0])

            accepted_times = paired_times(reference_derivatives,
                lambda: workspace.derivatives(probes[0]), args.repeat, rng,
                reference_prepare=reference_loss, prototype_prepare=prepare_accepted_fit)
            record = dict(setup_seconds=setup_seconds, options=VARIANTS[name],
                          agreement=agreement, memory=memory,
                          loss=loss_times, derivatives=derivative_times,
                          derivatives_after_accepted_loss=accepted_times)
            if args.profile:
                workspace.clear_cache()
                record["derivative_profile"] = profile(lambda: workspace.derivatives(probes[0]))
            if args.optimize:
                print(f"Full Newton optimization: {label}", flush=True)
                start = perf_counter()
                result = workspace.optimize(**optimizer_options)
                seconds = perf_counter() - start
                summary = optimization_summary(result, truth, microscope.alpha)
                summary.update(seconds=seconds, seconds_including_setup=seconds + setup_seconds,
                    speedup=output["reference_optimization"]["seconds"]/seconds,
                    speedup_including_setup=output["reference_optimization"]["seconds"]
                        / (seconds + setup_seconds),
                    max_coefficient_difference_from_reference=float(np.max(abs(
                        np.asarray(summary["strengths"])
                        - np.asarray(output["reference_optimization"]["strengths"])))),
                    success_matches_reference=summary["success"]
                        == output["reference_optimization"]["success"])
                record["optimization"] = summary
            output["variants"][label] = record
            print(f"{label}: derivatives {derivative_times['median_seconds']['prototype']:.4f}s; "
                  f"speedup {derivative_times['speedup']:.2f}x", flush=True)
            # Write completed stages incrementally so long runs retain results.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2) + "\n")
            del workspace
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
