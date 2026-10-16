18 objective scans: native 128², 64² and 32² images; all nine Z planes,
|Z| ≤ 1.5 µm or |Z| ≤ 1 µm; all probes or excluding third-order coma/trefoil
probe images (n=3, m=±1/±3). The latter removes 72/56/40 images, leaving
72/56/40 respectively instead of 144/112/80. No SNR filtering or alignment.

All 17 imposed/candidate coefficients, fixed known diversities on retained images,
object grid, rho and optics remain unchanged. At each α from −0.5 to 2 in 0.05
steps, the unconstrained regularized object is refitted. Each curve is normalized
by its own α=1 loss; raw losses are not comparable across image counts/resolutions.

The saved 100/1,000-start coefficient vectors are **reevaluated**, not reoptimized,
on each subset. Their ratios cannot establish where new subset optimizers would
converge. Saved original losses are checked only for the unchanged full-stack
objective. Canonical loss checks are performed for every scan.

```bash
python experimental_10_15/multiple_beads_z_probe_loss_profiles_20261015/manage.py submit
python experimental_10_15/multiple_beads_z_probe_loss_profiles_20261015/manage.py collect
```

`figures/` contains 6×6-inch bmh/LaTeX/300-DPI PNG/PDF figures: six Z-window
comparisons, nine probe comparisons, and individual curves with saved-estimate
loss ratios. `profiles.csv` contains all 918 points; `minima.csv` summarizes
sampled minima and saved-estimate ratios. No HTML is generated.
