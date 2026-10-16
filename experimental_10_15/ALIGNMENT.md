# Align a bead acquisition using its reference images

From the repository root in `phase_diversity_env`:

```sh
python -m experimental_10_15.align_stack RUN_FOLDER --output ALIGNED_FOLDER --preview
```

Choose a new/empty output directory. The script preserves the raw MAT/TIFF files
and writes a separate acquisition folder, including signed floating-point data
in `acquisition.mat`, copied phase/optics metadata, and aligned TIFF previews.
The derived folder works with the existing `compare`, `submit`, and replay tools.
Use **ALIGNED_FOLDER** as the acquisition path for a new fit; existing cluster
jobs and downloaded solutions are unaffected.

The default method estimates **lateral translation only**:

1. Within each Z plane, register `reference_after` to `reference_before`. These
   images have the same baseline/system-AO phase, without the imposed aberration
   or diversities, so their displacement estimates temporal drift.
2. Interpolate that applied shift through the intervening phase acquisitions.
   Recorded completion times are used if all are strictly increasing; otherwise
   acquisition order provides equal spacing. Averaging centers are approximated
   by these per-image times; individual exposure timestamps were not recorded.
3. Apply each shift once to the original floating image. Diversity images do
   **not** determine their own registration, preserving phase-dependent shifts
   and PSF structure that the aberration solver needs to explain.

Each plane keeps its own `reference_before` position by default. This avoids
assuming different focal planes show identical bead shapes. To also register
planes to a common reference, explicitly add:

```sh
python -m experimental_10_15.align_stack RUN_FOLDER --output ALIGNED_FOLDER \
  --align-planes --anchor-plane 3 --preview
```

The anchor is a **1-based plane index**, not a Z offset; without an explicit
anchor, the acquired plane nearest focus is selected. Cross-plane registration
can confuse changing bead visibility/defocus with motion. Inspect the results
carefully before using that option for a fit. No Z motion is estimated or changed.

Useful options:

- `--max-shift-pixels 8`: allowed shift per registration component in original
  acquisition pixels. Reaching the search boundary rejects registration.
- `--smoothing-sigma 1`: reference-image Gaussian smoothing for estimating shifts;
  no smoothing is applied to the images that are saved.
- `--min-correlation 0.4`: reject poor reference matches rather than silently
  shifting the stack. Registration optimizes normalized correlation to tolerate
  a constant gain or background difference.
- `--exclude-plane 5`: omit a known bad/incomplete plane. Repeat for more planes.
  This is required for plane 05 in the original no-aberration run. In the aligned
  output, excluded images are marked unacquired so later tools retain the exclusion.
- `--no-tex`: explicitly disable LaTeX for preview figures if unavailable locally.

Inspect `alignment_shifts.csv` and `alignment.json` for applied X/Y shifts,
reference correlations before/after alignment, timing sources, and source hashes.
The sign is the translation **applied to the image**: +X moves toward increasing
columns, +Y toward increasing rows. `figures/references_original_*.pdf` and
`references_aligned_*.pdf` show the before/after reference pairs in the existing
acquired-plane style with shared intensity scaling.

The original image dimensions, physical FOV, intensity units, phase coefficients
and Z coordinates are preserved. Linear interpolation can change pixel noise and
slightly blur fine features. Pixels exposed at image borders are filled with that
image's median border intensity, without circular wrapping. `alignment.npz`
records per-image and common valid masks; the current solver does not use those
masks to omit padded pixels. Inspect substantial border losses before fitting.
Two references support a linear drift estimate; they cannot detect nonlinear
motion or undo blur caused by movement during an averaged exposure. Run alternate
settings from the original acquisition, not an already resampled output.

To compare and submit the aligned stack:

```sh
python -m experimental_10_15.bead_pd compare ALIGNED_FOLDER --output ALIGNED_FOLDER/analysis
python -m experimental_10_15.submit_bead_pd submit ALIGNED_FOLDER \
  --output ALIGNED_FOLDER/submission --host mcw77@bouchet.ycrc.yale.edu \
  --fetch-output ALIGNED_FOLDER/solve
```

Submission bundles include the alignment report, applied shifts, validity masks
and original source metadata so the preprocessing remains reviewable.
