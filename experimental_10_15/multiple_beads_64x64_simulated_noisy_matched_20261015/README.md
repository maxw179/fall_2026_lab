Empirical Poisson–Gaussian repeat of the noiseless simulated 64² comparison.
Same 134 original image indices, ten artifact exclusions, nine Z planes, 17 free
modes/free object, 1,000 random starts and fast CUDA settings. No second flip.
The clean images and earlier optimization runs are preserved.

Calibration uses 16 usable before/after identical-command pairs, excluding the
flagged artifacts. Pair differences remove static image structure; robust smooth
brightness and first/second spatial derivative regressions reduce drift/bleaching
contamination only for variance estimation (no images are warped). A nonnegative
variance-versus-signal fit estimates effective Poisson gain. Per-image outer-band
MADs set Gaussian floors. Repeated-pair bootstrapping reports gain uncertainty.

Noise is added ONCE at saved/averaged/binned64 resolution: effective shot noise
plus independent Gaussian noise. No further division by 20 exposures or 4 bin
pixels. Expected image means/backgrounds are unchanged; signed values are kept.
One realization is frozen with seed 20261015. Sources, coefficients and hashes
are bundled in simulated_acquisition/noise_{calibration,provenance}.json.
This is an effective moment match, not an absolute photon calibration; residual
motion and structured noise can affect it, and stripes/artifacts are not recreated.

```bash
python experimental_10_15/multiple_beads_64x64_simulated_noisy_matched_20261015/submit_jobs.py --submit
python experimental_10_15/multiple_beads_64x64_simulated_noisy_matched_20261015/collect_results.py
```

Collection watches, downloads and builds replays.html. Existing receipts prevent
duplicate submissions; the noisy realization is cached and checked on retries.
