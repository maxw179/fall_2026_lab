For the short operator checklist, see [WORKFLOW.md](../WORKFLOW.md).

Run `bead_phase_diversity_acquire.m` from the ScanImage MATLAB workspace with
`hSI` available. The old `phase_diversity_acquire.m` entry point runs this script.
Edit the settings at the top of the new script before acquisition.

Use a centered 1 µm fluorobead at the focal plane, 1300 nm excitation, NA 1.05,
water index 1.333 and a calibrated 6.5 µm field of view. Configure ScanImage for
one plane, one ROI, the selected saved/displayed channel and 512 × 512 pixels;
stop acquisition before running. Keep XY position, zoom, laser power and
PMT gain fixed; move Z only when prompted. Optical values in the script are recorded metadata, not commands
to configure the microscope. Verify the pupil center/radius, LUT and relay phase
orientation on the actual SLM. The provided pupil coordinates require a large
SLM and deliberately fail if the pupil extends outside its queried dimensions.

Set `cfg.dryRun = true` to generate the commands and metadata without touching
hardware; change `outputRoot` to a local writable folder for this check.
Dry runs mark every image as unacquired and store NaNs, never synthetic images.

The default is a manually defined two-mode aberration. Choose
`cfg.aberrationSource = 'random'` for a seeded combination of 17 normalized real
Zernikes, scaled to 0.15 waves of piston-removed RMS on the sampled pupil.
Choose `cfg.aberrationSource = 'manual'` to apply `cfg.manualAberration` instead.
Each table row is `[m n coefficient_in_waves]`, for example `[2 2 0.10]` for
0.10 waves of cosine astigmatism. Unlisted modes are zero; all listed modes must
appear in `cfg.modes`, with no duplicates. Manual coefficients are applied
exactly without RMS rescaling; `imposedRMSWaves` and `randomSeed` only control
random generation. The actual sampled RMS and complete coefficient vector are
saved in `metadata.imposedPupilRMSWaves` and `metadata.imposedWaves`.
All mode pairs use Python's `[m n]` order and `utils.zernike` normalization.
This differs from the original MATLAB starter's unnormalized `[n m]` convention:
do not paste old coefficients into the new settings without converting them.
The diversities are ±0.10 waves of each astigmatism and spherical aberration,
with a shared zero-diversity image. All PD commands add the same imposed phase.
Set `cfg.zPlanesUm` at the beginning of the script (default: four planes from
−2 to +2 µm, matching the simulation). These are offsets from the manually
established bead-focus reference, not incremental moves from the previous plane.
Their sign follows your microscope Z readout; calibrate its relation to the
forward-model focus sign before fitting. Before each plane the script restores
the baseline phase, prints the requested offset, and waits for Enter after you
move manually and stop ScanImage. Enter `q` to cancel and preserve partial data.
Each plane acquires the full ten-image reference/diversity sequence with the same
imposed aberration. Dry runs skip movement prompts. There is no automatic stage
motion or automatic correction/estimation loop; corrections are loaded manually
between acquisitions using `cfg.correctionFile`. The stage remains at the last manual Z
position when the run completes or is cancelled.

Each run creates its own directory with:

- `metadata.mat`: optics, pupil, mode ordering, imposed coefficients, baseline,
  diversity coefficients, total commands, image names, PD selection, plane indices,
  requested Z offsets in µm (and focal positions in mm), and status.
- `acquisition.mat`: floating-point `images` in `[row,column,image]` order and a
  per-image log including Z offsets and manual confirmation, saved after each image so completed data survive a later failure.
- TIFF previews: 16-bit clipped/rounded images. Use the floating-point MAT data
  for fitting; negative PMT values are preserved there. Clipped pixel counts are
  logged, but detector saturation must be assessed for your PMT/digitizer range.

Images are ordered by plane, then by phase command. TIFF names include the plane
index and Z offset. Recorded Z values are requested, manually confirmed offsets;
the script does not read or verify stage coordinates.

Fit images selected by `metadata.isPDImage` using known `diversityWaves` and a
3P forward model with the recorded pixel sampling (6.5 µm / 512). The repeated
zero-diversity image can be excluded from fitting and used as a drift check.
Reference images exclude the imposed aberration and are not zero-diversity PD
images. Compare fitted coefficients with `imposedWaves`, remembering that the
estimate also includes residual native microscope aberration. To isolate the
imposed phase quantitatively, use an independently calibrated baseline or a
separate native-aberration estimate. `baselineWaves` is an optional fixed system
correction included in every command. The SLM is restored to baseline + imposed
+ current cumulative correction (no diversity) at completion/error. Imported
`correction.mat` files carry the complete cumulative vector, parent run/loop and
focus reference; MATLAB checks convergence and matching aberration/optics before
applying them. Keep the imposed aberration fixed across correction loops.

Acquisition averages frames per phase condition and refuses missing,
nonfinite or wrong-sized display data. `grabStrict` reads ScanImage's read-only
`hDisplay.lastFrame` after waiting for acquisition to return to idle and allowing
display callbacks to settle. It never assigns to that property. This follows the
existing controller's display-read approach; it does not independently verify
frame freshness with a frame counter or completion event.
Hardware execution has not been validated in this repository environment.

Use the local Python tools in [Python workflow README](../README.md)
to replay each acquired image, compare simulated/measured Z-stacks and explicitly
submit a fit to Bouchet. New runs default to `cfg.matFileVersion = '-v7'`, which
SciPy can read directly. Use `'-v7.3'` for files exceeding MATLAB v7's 2 GB limit;
reading those files in Python requires `h5py`. Copy the complete run folder after
acquisition so both `metadata.mat` and `acquisition.mat` are available.

The outside-pupil grating now uses the same method as `grab_data.m`:
`generateGratings(period, 3, false, WFC)`, selecting the first returned grating
and converting its phase bytes with `exp(1j * double(bytes) * pi/128)`. Keep the
microscope's `generateGratings` helper on the MATLAB path. The grating is generated
once per run and applied only outside the pupil; its applied phase bytes are
saved as `metadata.backgroundPhase8`. Dry runs omit the hardware grating and
record `metadata.backgroundGratingGenerated = false`.

With `cfg.averageInSingleGrab = true` (default), one ScanImage grab acquires
`cfg.framesPerImage` frames for each phase condition. The rolling display average
uses the same count, and the saved image is read from `lastAveragedFrame`.
This reduces shutter cycling to once per condition rather than once per frame.
The shutter can still close between phase commands and during manual Z movement.
The mode requires `hDisplay.lastAveragedFrame` and the legacy
`displayRollingAverageFactor` API; it fails before SLM initialization if the
averaged-frame property is unavailable. Set `averageInSingleGrab = false` to
restore separate single-frame grabs and software averaging. As before, configure
ScanImage for single-plane acquisition, without an automatic volume/stack.

Set `cfg.systemAOFigure = 'C:\data\sysAO_slm.fig'` to apply a fixed system AO
correction on every reference/test image and the restored SLM phase. Leave it
empty to disable. The loader reads CData from exactly one full-SLM-size image or
surface, requiring finite real values in [0,255]. It preserves orientation and
size, ignores colormap/CLim, and applies the map only inside the pupil so the
outside grating is retained. Raw units convert as `waves = CData/256`, not /255.
The map is added in floating point before phase wrapping, avoiding uint8 addition
saturation. It is saved in `metadata.systemAOPhaseSLM` and carried into the
exported `correction.mat`; changing it starts a new correction experiment.

System AO is treated as a calibration of native microscope aberration, not as
an additional Zernike probe. Simulation comparison assumes the native phase is
cancelled by that calibration; the solver estimates whatever remains after it.
The arbitrary pixel map is not approximated by the fitted Zernike modes or
added again as a known probe.
