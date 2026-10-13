"""Compare saved bead-sweep failures without rerunning optimization."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'analysis'


def main():
    OUT.mkdir(exist_ok=True)
    with np.load(BASE / 'results/pd_rms_bead_count_sweep_snr15_results.npz') as archive:
        d = {key: archive[key].copy() for key in archive.files}
    with np.load(BASE / 'results/pd_rms_bead_count_sweep_snr15_iteration_histories.npz') as archive:
        h = {key: archive[key].copy() for key in archive.files}
    for key in ('initial_rms', 'bead_counts', 'modes', 'optimizer_steps', 'slm_strengths'):
        if not np.array_equal(d[key], h[key]):
            raise ValueError(f'Archive mismatch: {key}')
    r = d['residual_rms']
    if not d['completed'].all() or not np.isfinite(r).all():
        raise ValueError('Analysis requires a completed finite sweep.')
    initial = d['initial_rms'][:, None, None]
    failed = r[..., -1] >= initial
    before = np.concatenate([np.broadcast_to(initial[..., None], r.shape[:-1] + (1,)), r[..., :-1]], axis=-1)
    harmful = r > before
    rows = []
    for ri, ci, trial in np.ndindex(failed.shape):
        centers = d['bead_positions_mm'][ci, trial, :int(d['bead_counts'][ci])] * 1000
        distances = d['pairwise_distances_3d_mm'][ci, trial, :len(centers), :len(centers)] * 1000
        pairs = distances[np.triu_indices(len(centers), 1)]
        rows.append(dict(initial_rms_waves=float(d['initial_rms'][ri]), bead_count=int(d['bead_counts'][ci]),
            trial=trial+1, failed=bool(failed[ri, ci, trial]), final_ratio=float(r[ri, ci, trial, -1]/initial[ri, 0, 0]),
            axial_spread_um=float(np.ptp(centers[:, 2])), mean_z_um=float(centers[:, 2].mean()),
            max_lateral_radius_um=float(np.linalg.norm(centers[:, :2], axis=1).max()),
            min_separation_um=float(pairs.min()) if len(pairs) else '',
            overlapping_pairs=int((pairs < 2 * float(d['bead_radius_mm']) * 1000).sum())))
        for loop in range(3):
            index = np.flatnonzero(np.all(h['loop_run_indices'] == [ri, ci, trial], axis=1) & (h['loop_record_indices'] == loop))
            if len(index) != 1:
                raise ValueError('Missing or duplicate loop record.')
            index = index[0]
            truth = d['truth_strengths'][ri, trial] + (d['slm_strengths'][ri, ci, trial, loop-1] if loop else 0)
            estimate = h['estimated_strengths'][index]
            rows[-1].update({f'loop_{loop+1}_residual_rms': float(r[ri, ci, trial, loop]),
                f'loop_{loop+1}_harmful': bool(harmful[ri, ci, trial, loop]),
                f'loop_{loop+1}_converged': bool(d['optimizer_success'][ri, ci, trial, loop]),
                f'loop_{loop+1}_clipped_modes': int((np.abs(estimate) > float(d['max_modal_update'])).sum()),
                f'loop_{loop+1}_coefficient_cosine': float(truth @ estimate / (np.linalg.norm(truth) * np.linalg.norm(estimate))),
                f'loop_{loop+1}_acquisition_snr': float(d['acquisition_snr'][ri, ci, trial, loop])})
    with (OUT/'trial_diagnostics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with plt.style.context('bmh'):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for t, ax in enumerate(axes):
            ratio = r[:, :, t, -1] / d['initial_rms'][:, None]
            im = ax.imshow(np.log10(ratio), vmin=-3, vmax=.5, cmap='coolwarm', aspect='auto')
            ax.set(title=f'Trial {t+1}', xlabel='Bead count', xticks=np.arange(5), xticklabels=d['bead_counts'],
                   yticks=np.arange(6), yticklabels=[f'{x:g}' for x in d['initial_rms']])
            for ri, ci in np.argwhere(failed[:, :, t]):
                ax.text(ci, ri, f'{ratio[ri,ci]:.2f}×', ha='center', va='center', color='black', fontsize=9)
        axes[0].set_ylabel('Initial RMS [waves]')
        fig.colorbar(im, ax=axes, label='log10(final RMS / initial RMS)', shrink=.8)
        fig.suptitle('Final correction outcome; labels mark failures', y=1.02)
        for ext in ('png', 'pdf'):
            fig.savefig(OUT/f'failure_outcomes.{ext}', dpi=200, bbox_inches='tight')
        plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        modes = d['modes']
        for t in range(3):
            axes[0].plot(np.arange(len(modes)), d['truth_strengths'][-1, t], 'o-', label=f'Trial {t+1}')
        axes[0].set(xticks=np.arange(len(modes)), xticklabels=[f'({m},{n})' for m,n in modes],
                    ylabel='Coefficient [waves]', title='Aberration directions scaled to initial RMS 0.30; modes (m,n)')
        axes[0].legend()
        for ci in range(5):
            for t in range(3):
                centers = d['bead_positions_mm'][ci,t,:ci+1]*1000
                count = int(failed[:,ci,t].sum())
                axes[1].scatter(np.ptp(centers[:,2]), np.linalg.norm(centers[:,:2],axis=1).max(),
                    c='tab:red' if count else 'tab:blue', s=40+35*count)
                axes[1].annotate(f'{ci+1}b/T{t+1}', (np.ptp(centers[:,2]), np.linalg.norm(centers[:,:2],axis=1).max()),
                                 xytext=(-40, 12) if (ci, t) == (3, 2) else (8, -12) if (ci, t) == (4, 0) else (4, 4),
                                 textcoords='offset points', fontsize=8)
        axes[1].set(xlabel='Bead-center axial spread [µm]', ylabel='Maximum lateral radius [µm]',
                    title='15 reused geometries; red = at least one failure; size increases with failure count')
        axes[1].margins(y=.12)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(OUT/f'aberration_geometry_diagnostics.{ext}', dpi=200)
        plt.close(fig)
    table = ['| Initial RMS | Beads | Trial | After loop 1 | After loop 2 | After loop 3 |', '|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['failed']:
            table.append('| ' + ' | '.join([f"{row['initial_rms_waves']:g}", str(row['bead_count']), str(row['trial']),
                *[f"{row[f'loop_{l}_residual_rms']:.4f}" for l in range(1,4)]]) + ' |')
    report = '''# Bead-count sweep: failure analysis

Failure means final applied residual RMS after loop 3 is at least the initial RMS. All 90 trajectories are complete; 7 failed and 83 improved. RMS values below are in waves. Harmful individual loops increase RMS versus the preceding applied state (38/270).

## Failed trajectories

''' + '\n'.join(table) + '''

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
'''
    (OUT/'failure_findings.md').write_text(report)
    print('Saved report, diagnostics for', len(rows), 'trials, and two PNG/PDF figure pairs.')


if __name__ == '__main__':
    main()
