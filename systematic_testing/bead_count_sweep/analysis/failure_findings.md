# Bead-count sweep: failure analysis

Failure means final applied residual RMS after loop 3 is at least the initial RMS. All 90 trajectories are complete; 7 failed and 83 improved. RMS values below are in waves. Harmful individual loops increase RMS versus the preceding applied state (38/270).

## Failed trajectories

| Initial RMS | Beads | Trial | After loop 1 | After loop 2 | After loop 3 |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 1 | 3 | 0.1686 | 0.1928 | 0.3097 |
| 0.1 | 2 | 3 | 0.1714 | 0.2794 | 0.2363 |
| 0.25 | 2 | 3 | 0.3136 | 0.3380 | 0.2939 |
| 0.3 | 1 | 1 | 0.4092 | 0.3819 | 0.3410 |
| 0.3 | 2 | 3 | 0.2830 | 0.2627 | 0.3315 |
| 0.3 | 4 | 2 | 0.3342 | 0.2233 | 0.4137 |
| 0.3 | 5 | 1 | 0.3836 | 0.3202 | 0.3885 |

## Observed commonalities

- **Initial amplitude:** failures by RMS are 0/15 at 0.05, 2/15 at 0.10, 0/15 at 0.15, 0/15 at 0.20, 1/15 at 0.25, and 4/15 at 0.30. Large amplitude is a risk in this sample, but the isolated 0.10 failures and successful 0.15–0.20 cases show a nonmonotonic outcome consistent with basin selection/noise effects.
- **Bead count:** failures are 2/18, 3/18, 0/18, 1/18, and 1/18 for 1–5 beads. Five of seven failures involve one or two beads. Three beads succeeded in every trial here, but four/five beads did not eliminate failure.
- **Aberration direction:** trial 3 accounts for 4/7 failed trajectories and 22/38 harmful loops; trial 1 accounts for 2 failures and 9 harmful loops, trial 2 for 1 failure and 7 harmful loops. These are only three distinct wavefront directions, reused over all amplitudes/counts. Trial identity also selects a different geometry at every bead count, so direction and placement cannot be isolated causally.
- **Repeated geometry:** two-bead trial 3 fails at RMS 0.10, 0.25, and 0.30. Its centers are (-1.06, 1.20, -0.79) and (-1.11, -0.87, -0.89) µm: nearly the same x/z, axial spread 0.098 µm, mean z -0.842 µm, separation 2.071 µm (no overlap for 1 µm radius). The two successful two-bead geometries have axial spreads 0.405 and 0.929 µm, and overlap. Limited axial structure is a plausible contributor to poorer identification in this specific geometry. It still succeeds at 0.05, 0.15, and 0.20.
- **No universal placement signature:** one failed single-bead geometry is near the center (lateral radius 1.022 µm), another is farther out (2.389 µm). Successful four-/five-bead geometries reach similar or larger lateral radii than failed ones. Both overlapping and nonoverlapping geometries fail, and many overlapping geometries succeed. Neither overlap nor proximity to the edge alone explains the failures.
- **Aberration content:** trial 3's largest initial coefficient is (m,n)=(-4,4), followed by (1,3) and (5,5). These correspond to a fourth-order fourfold mode, primary coma, and a fifth-order fivefold mode. Trial 1 instead has a large (-3,5) coefficient and 56.8% of its squared coefficient magnitude in radial order 5, versus 31.5% in trial 2 and 27.2% in trial 3. Thus failures do not share one dominant aberration family or simply the highest fifth-order content. Squared-coefficient shares describe content; they are not an exact attribution of sampled-pupil RMS.

## What the loop records suggest

- **Convergence does not certify the correction:** 20/21 optimizer calls in failed trajectories report convergence, compared with 225/249 in successful trajectories. One failed call stops below strength tolerance. The main issue is inaccurate estimates despite solver convergence, rather than widespread exhaustion of iterations.
- **Wrong-direction estimates:** five of seven failed trajectories worsen immediately. Four of those first-loop estimates have negative coefficient cosine with the true residual (the fifth is nearly orthogonal at 0.052). Subtracting a wrong-direction estimate increases wavefront error. Cosine is computed in coefficient space, as a diagnostic, rather than the sampled-pupil RMS metric.
- **Late reversals:** RMS 0.30 / two beads / trial 3 improves to 0.2627 by loop 2 but finishes at 0.3315. RMS 0.30 / four beads / trial 2 reaches 0.2233 then ends at 0.4137; its last estimate has coefficient cosine -0.438. These are not simply cases that never made progress.
- **Clipping is not sufficient as an explanation:** neither RMS 0.10 failure clips any coefficient in any loop. The higher-amplitude failures do clip some estimates, but clipping also limits harmful updates. The records indicate estimation errors before application, rather than clipping alone.
- **Noise is not a universal explanation:** first-loop median acquisition SNR is 14.1 for failed trajectories and 12.6 for successful ones. Some final harmful loops have diagnostic acquisition SNR 41.3 or 87.2. Later SNR is affected by the preceding correction and is not an independent predictor; calibration SNR 15 is not a constant acquired-stack SNR.
- **Iteration counts:** failed trajectories have median steps [22,20,27], versus [22,10,3] for successful trajectories. The continuing long solves in loops 2/3 reflect failure to reach the easy small-residual regime, but do not establish why the initial fit went wrong.

## Follow-up that would discriminate explanations

Cross all three aberration directions with all saved geometries at selected RMS levels, rather than coupling both to trial index. Repeat photon-noise and random-start seeds separately. For the troublesome two-bead geometry, vary axial separation while preserving lateral coordinates and control/calibrate brightness. Compare more complete mode diversity and repeated/more thorough initialization against the current schedule. These would test placement, wavefront direction, and objective-basin effects separately; no new optimization was run for this report.

Only 15 geometries and three aberration directions were sampled. Repeated RMS levels are not independent new geometries/wavefronts, and the descriptive counts do not establish statistical significance or causation.

## Artifacts

![Final outcome](failure_outcomes.png)

![Geometry and aberration diagnostics](aberration_geometry_diagnostics.png)

[All trial and loop diagnostics](trial_diagnostics.csv)
