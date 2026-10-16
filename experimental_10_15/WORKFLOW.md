# Bead experiment: acquire → compare → solve → correct

**Once:** use the laptop's `phase_diversity_env` Python environment and configure
an SSH login alias for Bouchet. Run terminal commands from this repository root.
Replace `RUN`, `SUBMISSION`, `FETCHED` and `YOUR_LOGIN_ALIAS` below with your paths
and SSH alias; use new folders for each loop.

1. **Microscope: align and configure.** In ScanImage, center the bead approximately
   and focus it. Stop acquisition. In `experimental_10_15/matlab_oct15/bead_phase_diversity_acquire.m`,
   check the output folder, optics/FOV, image size, Z offsets and frame averaging.
   Set `cfg.systemAOFigure` to your system correction `.fig` (or `''` to disable).
   Its CData must match the full SLM image size with raw phase values 0–255.
   Set `cfg.aberrationSource = 'manual'`, edit `cfg.manualAberration` (rows are
   `[m n waves]`), and leave `cfg.correctionFile = ''` for loop 1. Comparison and
   solver settings are also defined here under `cfg.analysis` and `cfg.solver`;
   you do not need to duplicate them on the laptop. Optical settings are recorded
   values: set the actual microscope to match them in ScanImage.

2. **Microscope: acquire.** Click MATLAB **Run**. At the first prompt, enter the
   focused bead's microscope Z readout in µm. At each plane prompt, move manually
   to the displayed **target Z readout**, stop ScanImage acquisition, then press
   **Enter** in the MATLAB Command Window. The script averages the frames and
   captures all diversities automatically. Enter **q** to cancel. Note the saved
   run folder; the original focus readout is now in its metadata.

3. **Laptop: copy and inspect.** Copy the entire saved run folder to the laptop.
   Open `experimental_10_15/bead_phase_diversity.ipynb`, select **phase_diversity_env**,
   edit only `run_dir`, then click **Run All**. Inspect the measured/simulated
   columns at every Z and diversity. An approximate lateral bead center is
   estimated from the reference image; all other settings come from MATLAB
   metadata. If comparison is poor, resolve alignment/settings before submitting.
   For terminal-only comparison:

   ```sh
   python -m experimental_10_15.bead_pd compare RUN --output RUN/analysis
   ```

4. **Laptop: submit after inspection.** When satisfied, run:

   ```sh
   python -m experimental_10_15.submit_bead_pd submit RUN \
     --output SUBMISSION --host mcw77@YOUR_LOGIN_ALIAS --fetch-output FETCHED
   ```

   This uploads, queues the GPU solve, prints progress and downloads the result.
   Future submissions automatically use the optimized CUDA QR path, batched
   initialization and cached Newton calculations; no extra flags are needed.
   `results/runtime.json` records the execution settings and factorization counts.
   If watching is interrupted, reconnect with `watch SUBMISSION/job_JOBID.json`;
   fetch separately with `fetch SUBMISSION/job_JOBID.json --output FETCHED`, using
   the same `python -m experimental_10_15.submit_bead_pd` prefix.

5. **Inspect the estimate.** Open `FETCHED/results/result.json` and
   `coefficients.csv`. Check `optimizer_success` is true. `correction.mat`
   contains the **complete cumulative correction** for the next loop. The solve
   estimates the remaining aberration, which may include native microscope phase.

6. **Microscope: repeat with correction.** Copy `FETCHED/results/correction.mat`
   onto the microscope computer. In MATLAB, set `cfg.correctionFile` to its full
   path, e.g. `'C:\data\loop1\correction.mat'`. Keep the original
   `manualAberration`, optical settings and bead fixed; return the stage to the
   original focus readout without choosing a new focus reference. Click **Run**.
   MATLAB loads the cumulative correction, checks the parent experiment, reuses
   the focus reference, and prompts for the same Z planes. Repeat steps 3–6 with
   fresh folders. Each solve subtracts its residual estimate from the previous
   cumulative correction; you do not add coefficient vectors by hand.

At acquisition completion the SLM is written with baseline + original imposed
aberration + current cumulative correction, with no diversity. The stage remains
at the last requested Z; move it back to the original focus readout to inspect
the corrected bead. System AO is included inside the pupil on **every image**, including references
and the final restored phase. Reference images add only the calibrated baseline.
Keep the same system AO map between correction loops. For a
**new** bead, aberration or optical configuration, clear `cfg.correctionFile` and
`cfg.zReferenceReadoutUm` to start a fresh experiment.
