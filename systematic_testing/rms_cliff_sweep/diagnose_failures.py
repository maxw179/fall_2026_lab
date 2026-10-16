"""Replay saved objectives at truth/fit and inspect basins without rerunning fits.

All losses profile out the unconstrained regularized object, using the original
images, axial grid, rho, padding and historical SLM phases. CPU direct SVD is an
independent check of the recorded GPU loss. Truth is diagnostic only.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np

from optimization import reconstruction
from systematic_testing.rms_cliff_sweep.run_sweep import SweepConfig, make_scene
from utils import psf, zernike


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def replay_workspace(case, loop, *, images_override=None):
    """Reconstruct the exact cumulative acquisition in its current-SLM frame."""
    case = Path(case)
    meta = json.loads((case/'run.json').read_text())
    config = SweepConfig(**meta['config'])
    microscope, grid, sample, sample_z, _, truth, _ = make_scene(
        config, meta['rms_index'], meta['trial'])
    images, focal_z, phases, slms = [], [], [], []
    for index in range(1, loop+1):
        with np.load(case/f'acquisition_{index}.npz') as acquired:
            images.append(acquired['images'].copy())
            focal_z.append(acquired['focal_z'].copy())
            slms.append(acquired['acquisition_slm_strengths'].copy())
            phases.append(acquired['diversity_strengths']+slms[-1])
    current_slm = slms[-1]
    diversities = [zernike.Aberration(truth.modes, strengths-current_slm)
                   for strengths in np.concatenate(phases)]
    with np.load(case/'results.npz') as saved:
        if not saved['completed'][loop-1]:
            raise ValueError(f'Loop {loop} is not complete in {case}.')
        estimated = saved['estimated_strengths'][loop-1].copy()
    inputs = dict(microscope=microscope, grid=grid,
        images=np.concatenate(images) if images_override is None else images_override,
        focal_z_levels=np.concatenate(focal_z), sample_z_levels=sample_z,
        modes=truth.modes, rho=config.rho, diversities=diversities, mode='vector',
        backend='cpu', reuse_svd=False, retain_fields=False)
    workspace = reconstruction.ReconstructionWorkspace(**inputs)
    return workspace, truth.strengths+current_slm, estimated, meta, sample, inputs


def diagnose(case, loop, *, profile=True, controls=False):
    workspace, truth, fitted, meta, sample, inputs = replay_workspace(case, loop)
    modes, alpha = inputs['modes'], inputs['microscope'].alpha
    rms = lambda x: float(zernike.zernike_RMS(zernike.Aberration(modes, x), alpha))
    true_loss = workspace.loss(truth)
    fit_loss = workspace.loss(fitted)
    zero_loss = workspace.loss(np.zeros_like(truth))
    recorded = meta['loops'][loop-1]['loss']
    replay_error = abs(fit_loss-recorded)/recorded
    if replay_error > 1e-7:
        raise ValueError(f'CPU replay disagrees with saved loss by {replay_error:.3g}: {case}, loop {loop}')
    identity = dict(run_id=Path(case).parent.name, case_index=meta['case_index'],
        trial=meta['trial'], initial_rms=meta['initial_rms'],
        diversity_rms=meta['config'].get('diversity_rms', .1), ao_loop=loop)
    row = dict(identity, truth_loss=true_loss, fit_loss=fit_loss, zero_loss=zero_loss,
        fit_over_truth=fit_loss/true_loss, saved_loss_relative_replay_error=replay_error,
        fit_error_rms=rms(fitted-truth), optimizer_success=meta['loops'][loop-1]['success'],
        optimizer_message=meta['loops'][loop-1]['message'],
        accepted_steps=meta['loops'][loop-1]['iterations'])
    with np.load(Path(case)/f'loop_{loop}_screening.npz') as screening:
        selected = int(screening['selected_index'])
        distances = np.array([rms(x-truth) for x in screening['strengths']])
        row.update(screened_candidates=len(distances),
            selected_start_error_rms=distances[selected], nearest_start_error_rms=distances.min(),
            selected_start_over_truth=screening['losses'][selected]/true_loss,
            candidates_within_010_rms=int((distances < .10).sum()))
    trace = []
    with np.load(Path(case)/'history.npz') as history:
        selected = history['loop'] == loop
        losses = history['loss'][selected]
        row['history_loss_increases'] = int((np.diff(losses) > 1e-10*true_loss).sum())
        for iteration, loss, strengths in zip(history['iteration'][selected], losses,
                                             history['strengths'][selected]):
            trace.append(dict(identity, iteration=int(iteration), loss=float(loss),
                              loss_over_truth=float(loss/true_loss), error_rms=rms(strengths-truth)))
    profiles = []
    if profile:
        ts = np.unique(np.r_[np.linspace(0, 1, 21), -.02, -.005, .005, .02, .98, .995, 1.005, 1.02])
        for t in ts:
            loss = (true_loss if t == 0 else fit_loss if t == 1
                    else workspace.loss(truth+t*(fitted-truth)))
            profiles.append(dict(identity, fraction_truth_to_fit=float(t), loss=float(loss),
                                 loss_over_truth=float(loss/true_loss)))
        row['best_sampled_profile_over_truth'] = min(p['loss_over_truth'] for p in profiles)
        row['best_sampled_profile_fraction'] = min(profiles, key=lambda p: p['loss'])['fraction_truth_to_fit']
    control_rows = []
    if controls:
        if loop != 1:
            raise ValueError('Noiseless controls currently require loop one.')
        for label, control_sample in (
                ('noiseless_original_41_plane_object', sample),
                ('noiseless_matched_21_plane_object', psf.bead_img_3D(
                    inputs['grid'], inputs['sample_z_levels'], [0.], [0.], [0.], [.001]))):
            clean = inputs['microscope'].compute_image_3D(control_sample,
                zernike.Aberration(modes, truth), inputs['focal_z_levels'],
                mode='vector', diversities=inputs['diversities'])[3]
            control = reconstruction.ReconstructionWorkspace(**dict(inputs, images=clean))
            reference = control.loss(truth)
            for t in (-.02, -.005, 0., .005, .02, 1.):
                loss = reference if t == 0 else control.loss(truth+t*(fitted-truth))
                control_rows.append(dict(identity, control=label,
                    fraction_truth_to_fit=t, loss=loss, loss_over_truth=loss/reference))
            del control
    del workspace
    gc.collect()
    print(f"{identity}: fit/truth loss={row['fit_over_truth']:.6f}; "
          f"CPU/GPU replay relative error={replay_error:.2g}", flush=True)
    return row, profiles, trace, control_rows


def plot_scaled_truth_profiles(curves, *, levels=None, title='', use_tex=True):
    """Plot experimental truth-scaling scans using the existing profile style."""
    import matplotlib.pyplot as plt
    with plt.style.context('bmh'), plt.rc_context({'text.usetex': use_tex, 'font.size': 9}):
        fig, ax = plt.subplots(figsize=(6, 6))
        for label, fractions, losses in curves:
            ax.plot(fractions, losses, 'o-', markersize=3, label=label)
            best = int(np.argmin(losses))
            ax.plot(fractions[best], losses[best], 'D', color='black', markersize=5)
        ax.axvline(1., color='black', linestyle=':', linewidth=1, label='True imposed wavefront')
        ax.axhline(1., color='black', linestyle='--', linewidth=1, label='Loss at truth')
        for index, (label, value) in enumerate((levels or {}).items()):
            ax.axhline(value, linestyle=['-.', ':'][index % 2], linewidth=1, label=label)
        ax.set(xlabel=r'Imposed wavefront scale $\alpha$ (1 = truth)',
               ylabel='Profiled loss / loss at truth', title=title)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.32), fontsize=8, ncol=2)
        fig.subplots_adjust(left=.15, right=.95, bottom=.14, top=.73)
    return fig, ax


def plot_profiles(rows, output, *, use_tex=True):
    import matplotlib.pyplot as plt
    grouped = {}
    for row in rows:
        grouped.setdefault((row['case_index'], row['ao_loop']), []).append(row)
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for (case, loop), group in grouped.items():
        with plt.style.context('bmh'), plt.rc_context({'text.usetex': use_tex, 'font.size': 9}):
            fig, ax = plt.subplots(figsize=(6, 6))
            for index, magnitude in enumerate(sorted({r['diversity_rms'] for r in group})):
                subset = [r for r in group if r['diversity_rms'] == magnitude]
                ax.plot([r['fraction_truth_to_fit'] for r in subset],
                        [r['loss_over_truth'] for r in subset], 'o-', markersize=3,
                        color=palette[index % len(palette)],
                        label=fr'Diversity $\pm {magnitude:.2f}$ waves')
            ax.axhline(1., color='black', linestyle='--', linewidth=1, label='Loss at truth')
            ax.set(xlabel='Interpolation: 0 = true aberration; 1 = saved fit',
                   ylabel='Profiled loss / loss at truth',
                   title=f'Case {case}, initial RMS {group[0]["initial_rms"]:g}, AO loop {loop}')
            ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.25), fontsize=8)
            fig.subplots_adjust(left=.15, right=.95, bottom=.14, top=.76)
            for ext in ('png', 'pdf'):
                fig.savefig(output/f'loss_profile_case_{case:04d}_loop_{loop}.{ext}', dpi=300)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--cases', nargs='+', type=int, default=[111, 167])
    parser.add_argument('--loops', nargs='+', type=int, choices=[1, 2, 3], default=[1, 3])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--controls', action='store_true', help='Noiseless controls for case 167, diversity .10, loop 1.')
    parser.add_argument('--no-tex', action='store_true')
    args = parser.parse_args()
    output = args.output or args.root/'analysis/failure_diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    summaries, profiles, traces, controls = [], [], [], []
    for case_index in args.cases:
        for run in sorted(args.root.glob('[0-9]*')):
            case = run/f'case_{case_index:04d}'
            if not (case/'results.npz').exists():
                continue
            for loop in args.loops:
                with np.load(case/'results.npz') as saved:
                    if not saved['completed'][loop-1]:
                        continue
                meta = json.loads((case/'run.json').read_text())
                result = diagnose(case, loop, profile=loop == 1,
                    controls=args.controls and case_index == 167 and loop == 1
                    and meta['config'].get('diversity_rms', .1) == .1)
                summaries.append(result[0]); profiles.extend(result[1])
                traces.extend(result[2]); controls.extend(result[3])
                # Incremental outputs survive interruption of a longer replay.
                for name, rows in (('endpoints', summaries), ('profiles', profiles),
                                   ('histories', traces), ('noiseless_controls', controls)):
                    write_csv(output/f'{name}.csv', rows)
    plot_profiles(profiles, output, use_tex=not args.no_tex)
    print(f'Saved diagnostics to {output}', flush=True)


if __name__ == '__main__':
    main()
