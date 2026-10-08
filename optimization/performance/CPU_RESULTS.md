# CPU prototype results

Measured 2026-10-07 on local ARM macOS, NumPy 2.4.6 / OpenBLAS 0.3.31,
using the project Python environment and OPENBLAS_NUM_THREADS=1. Benchmark
processes ran sequentially. Both implementations used the same synthetic data,
vector multiphoton optics (order 3), rho=1e48, Poisson SNR=50, and zero initial
coefficients. These measurements precede promotion of the workspace to the
production default. The new API integration and its updated tests have not been
executed, at the user's request; the numerical results below describe the tested
workspace before promotion.

## Numerical agreement

All seven prototype tests in `tests/test_reconstruction_cpu_workspace.py` passed
before integration. That file now also checks the public APIs against the
preserved reference solver; its updated suite has not been rerun. The original checks
include reference parity for the improvement switches, finite-difference gradient
and Hessian checks, padded/unpadded edge wrapping, K<P null-space response, K>P,
repeated PSFs, field recomputation, snapshot/caching behavior, small rho, Newton
recovery and negative-curvature escape.

All four benchmark variants also passed loss/gradient/Hessian agreement checks
at zero, half-truth and true strengths, including evaluation after a cached loss
fit. For the complete prototype, the maximum componentwise differences after
division by reference loss were:

| Case | Loss | Gradient | Hessian |
| --- | ---: | ---: | ---: |
| 64x64, 1 mode | 4.44e-16 | 6.91e-12 | 1.12e-9 |
| 32x32, 15 modes | 2.22e-16 | 3.55e-14 | 5.51e-12 |

These are absolute differences in loss-scaled quantities, not relative errors
of individual gradient/Hessian entries. All comparisons satisfied rtol=2e-8,
atol=2e-9. The exact Hessian and current Newton stopping rules were retained.

The one-mode solvers both converged in 5 accepted steps, recovering spherical
strength -0.149658350543136 waves (truth -0.15). The maximum final coefficient
difference was 4.44e-16 waves, and both had residual wavefront RMS
0.000339117383771 waves.

The 15-mode solvers both converged in 15 accepted steps, with maximum final
coefficient difference 1.98e-16 waves. Both had residual wavefront RMS
0.065008743669273 waves. Their success flag indicates convergence to the same
local stationary solution; it does not mean that the true aberration was fully
recovered. The faster code preserves the existing objective's result.

## Complete prototype timings

The 64x64 case used 24 acquired images and 40 reconstructed planes. The 32x32
case used 15 images and 9 reconstructed planes. Pupil resolutions matched image
resolutions. Three warm, paired, randomly ordered repeats were used for each
call measurement; the table reports medians. Fit caches were cleared before
uncached loss and derivative measurements.

| Case and operation | Original | CPU prototype | Speedup |
| --- | ---: | ---: | ---: |
| 64x64, 1 mode: loss | 1.361 s | 0.740 s | 1.84x |
| 64x64, 1 mode: loss + gradient + exact Hessian | 3.787 s | 1.600 s | 2.37x |
| 64x64, 1 mode: derivatives after accepted loss fit | 3.318 s | 0.492 s | 6.74x |
| 64x64, 1 mode: full Newton run, including prototype setup | 24.949 s | 7.264 s | 3.43x |
| 32x32, 15 modes: loss | 0.067 s | 0.027 s | 2.51x |
| 32x32, 15 modes: loss + gradient + exact Hessian | 1.790 s | 1.292 s | 1.39x |
| 32x32, 15 modes: derivatives after accepted loss fit | 2.079 s | 1.244 s | 1.67x |
| 32x32, 15 modes: full Newton run, including prototype setup | 36.804 s | 20.844 s | 1.77x |

Prototype setup took 0.0250 s and 0.00269 s respectively. Full Newton runs were
single measurements per variant, rather than three repeats. Synthesis and
agreement checks are excluded. The accepted-fit row excludes the preceding loss
evaluation and measures only its subsequent derivative cost; it is not a full
optimization-step speedup.

## Contributions of the improvements

| Cumulative variant | 64x64 full-run speedup | 32x32 full-run speedup |
| --- | ---: | ---: |
| Geometry, batching and within-call field reuse | 1.53x | 1.46x |
| Also shared SVD | 1.85x | 1.46x |
| Also spatial second-derivative contractions | 1.88x | 1.77x |
| Also identical-PSF reuse | 3.43x | 1.77x |

Each full-run speedup includes prototype setup. In the 64x64 case, duplicate-PSF
reuse reduced 960 requested PSFs to 300 exact distinct combinations. The 32x32
case had 135 requests and 135 distinct combinations, so that switch removes no
PSF work. This explains why its last two full-run times are nearly identical.

The one-mode case has only one second derivative per PSF; spatial contractions
have little effect there. With 15 modes and 120 second-derivative pairs, they
provided a useful additional reduction. Reusing the SVD has more impact in the
case with 40 sample planes than the case with 9.

Timings varied between stages: for example, the 15-mode reference derivative
median was 2.229 s during the spatial-only stage and 1.790 s during the final
stage. Corresponding prototype medians were 1.251 s and 1.292 s. Consequently,
differences between speedup ratios from different stages are not isolated causal
measurements. Use the paired final-stage timings and full-run results as local
estimates, rather than universal speedup guarantees.

The prototype exchanges RAM for speed by retaining SVD factors and derivative
fields. The final 64x64 workspace reported approximately 351 MB of cached fit
arrays, 23 MB of listed fixed arrays, 128 MB of first-derivative OTFs and 201 MB
of temporarily retained derivative fields. These partial array estimates are
not process peak-memory measurements. `--recompute-fields` can lower field
storage at the cost of extra propagation; it was checked numerically but not
timed in these runs.

## Artifacts

- [64x64 one-mode results](cpu_64_one_mode.json)
- [32x32 fifteen-mode results](cpu_32_fifteen_modes.json)
- [Benchmark and reproduction instructions](CPU_BENCHMARK.md)

The runs used the documented commands with `--optimize` and the corresponding
output filenames. No GPU or CUDA implementation was involved.
