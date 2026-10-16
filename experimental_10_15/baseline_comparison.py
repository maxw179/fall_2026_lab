"""Compare downloaded bead solves after subtracting either zero-control estimate.

Local post-processing only: no uploads, solver runs or correction-file changes.
Run: python -m experimental_10_15.baseline_comparison
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from experimental_10_15.bead_pd_io import write_json
from experimental_10_15.replay import build_replay
from experimental_10_15.results_site import LABELS
from systematic_testing.ao_animation.viewer import _phase, render_replay
from systematic_testing.correction_examples.plot_pd_iteration_rms import plot_error_histories
from utils import psf, zernike

ROOT = Path(__file__).resolve().parent
BASELINES = {'no_aberration': 'Minus no-aberration control',
             'system_AO': 'Minus System AO control'}
COLORS = ['tab:blue', 'tab:orange', 'tab:green']
NOTICE = ('Post-processing comparison: subtract a zero-control final signed coefficient '
          'vector from every saved estimate, then recompute phase and RMS against the '
          'known imposed wavefront. Controls use the same filtering and image resolution '
          'as each target; SNR selection can retain different individual images. The '
          'System AO control uses the same AO map as the aberrated acquisitions. The '
          'no-aberration control has a different system-AO calibration. Playback switches '
          'between final baseline choices; the loss and RMS panels show full saved '
          'optimizer histories. Measured images and optimizer losses are unchanged. '
          'No microscope correction is applied or exported.')


def load_fit(input_path, results):
    input_path, results = Path(input_path), Path(results)
    with np.load(input_path, allow_pickle=False) as data:
        fit = {key: data[key].copy() for key in (
            'modes', 'cumulative_correction', 'imposed_for_validation',
            'microscope_json', 'system_ao_phase_slm', 'field_of_view_um')}
        fit['shape'] = data['images'].shape[-2:]
        fit['z_um'] = np.unique(data['focal_z'])*1000
    with np.load(results/'solution.npz', allow_pickle=False) as data:
        if not np.array_equal(data['modes'], fit['modes']):
            raise ValueError('Solution and input mode bases differ.')
        if not np.array_equal(data['cumulative_correction'], fit['cumulative_correction']):
            raise ValueError('Solution and input cumulative corrections differ.')
        final = data['estimated_strengths'].copy()
    optimizer = json.loads((results/'optimizer.json').read_text())
    history = optimizer['history']
    if not history or not np.allclose(final, history[-1]['strengths'], rtol=1e-10, atol=1e-12):
        raise ValueError('The solution must match the final saved optimizer iterate.')
    fit.update(input_path=input_path, results=results, history=history,
               estimate=final-fit['cumulative_correction'])
    fit['alpha'] = psf.Microscope(**json.loads(str(fit['microscope_json']))).alpha
    return fit


def baseline_vector(target, control):
    """Align a zero-control estimate by (m,n); remove its known applied correction."""
    if not np.allclose(control['imposed_for_validation'], 0, rtol=0, atol=1e-12):
        raise ValueError('A baseline acquisition must have zero imposed aberration.')
    target_modes = [tuple(mode) for mode in target['modes']]
    control_modes = [tuple(mode) for mode in control['modes']]
    if (len(set(target_modes)) != len(target_modes) or
            len(set(control_modes)) != len(control_modes) or set(target_modes) != set(control_modes)):
        raise ValueError('Baseline and target must use the same distinct mode basis.')
    if (json.loads(str(target['microscope_json'])) != json.loads(str(control['microscope_json']))
            or target['shape'] != control['shape']
            or not np.isclose(target['field_of_view_um'], control['field_of_view_um'])):
        raise ValueError('Baseline and target microscope, resolution and FOV must match.')
    return control['estimate'][[control_modes.index(mode) for mode in target_modes]]


def error_histories(target, controls):
    """Subtract signed vectors, never scalar RMS values or estimated magnitudes."""
    estimates = np.asarray([row['strengths'] for row in target['history']])-target['cumulative_correction']
    vectors = {'Original': np.zeros(len(target['modes']))}
    vectors.update({BASELINES[key]: baseline_vector(target, value) for key, value in controls.items()})
    adjusted = {label: estimates-vector for label, vector in vectors.items()}
    errors = {label: np.array([float(zernike.zernike_RMS(
        zernike.Aberration(target['modes'], estimate-target['imposed_for_validation']), target['alpha']))
        for estimate in values]) for label, values in adjusted.items()}
    return adjusted, errors, vectors


def discover_groups(root):
    """Keep original, filtered and binned control fits in separate groups."""
    groups = []
    original = {}
    for solution in sorted(root.glob('3d_fluorobeads_*/tp*/solve_20261015_*/results/solution.npz')):
        acquisition = solution.parents[2]
        tag = acquisition.parent.name.removeprefix('3d_fluorobeads_')
        submission = acquisition/solution.parents[1].name.replace('solve_', 'submission_', 1)/'input.npz'
        if submission.is_file():
            original[tag] = load_fit(submission, solution.parent)
    if original:
        groups.append(('Original 128x128 · all valid acquired Z', original))
    for batch in sorted(root.glob('review_middle3_snr*')):
        if not (batch/'jobs.json').is_file():
            continue
        fits = {}
        for job in json.loads((batch/'jobs.json').read_text()):
            folder = batch/job['name']
            if (folder/'solve/results/solution.npz').is_file():
                fits[job['name'].removeprefix('3d_fluorobeads_')] = load_fit(
                    folder/'submission/input.npz', folder/'solve/results')
        threshold = json.loads((batch/'summary.json').read_text())['min_snr']
        groups.append((f'SNR >= {threshold:g} · Z = -1, 0, +1 um · 128x128', fits))
    for batch in sorted(root.glob('binning_all_images_*')):
        if not (batch/'jobs.json').is_file():
            continue
        for size in (64, 32):
            fits = {}
            for job in json.loads((batch/'jobs.json').read_text()):
                if job['size'] != size:
                    continue
                folder = batch/job['name']
                if (folder/'solve/results/solution.npz').is_file():
                    tag = job['name'].removeprefix('3d_fluorobeads_').removesuffix(f'_{size}x{size}')
                    fits[tag] = load_fit(folder/'submission/input.npz', folder/'solve/results')
            groups.append((f'{size}x{size} · all valid Z · no SNR filter', fits))
    return groups


def export_case(target, controls, output, title):
    output.mkdir(parents=True, exist_ok=True)
    adjusted, errors, vectors = error_histories(target, controls)
    iterations = [row['iteration'] for row in target['history']]
    trace = [dict(label=label, x=iterations, y=values.tolist(), color=COLORS[i])
             for i, (label, values) in enumerate(errors.items())]
    source_frames, points, _ = build_replay(target['results'], target['input_path'])
    frames = []
    for label, estimates in adjusted.items():
        estimated = zernike.Aberration(target['modes'], estimates[-1])
        error = zernike.Aberration(target['modes'], estimates[-1]-target['imposed_for_validation'])
        frame = dict(source_frames[-1], label=label, rms=float(errors[label][-1]),
                     optimization_rms=float(errors[label][-1]), trace_series=trace,
                     loss_history=(iterations, [row['loss'] for row in target['history']]),
                     trace_title='Error after baseline subtraction')
        for key, aberration in [('estimate', estimated), ('command', error)]:
            frame[key] = _phase(aberration, target['alpha'], 101)
            frame[key+'_function'] = aberration.construct_map(target['alpha'])
        frames.append(frame)
    document = render_replay(frames, points, title=title)
    document = document.replace('<body>', '<body><p>'+html.escape(NOTICE)+'</p>', 1)
    (output/'comparison.html').write_text(document)
    fig, _ = plot_error_histories(iterations, errors)
    fig.savefig(output/'rms.png', dpi=300)
    fig.savefig(output/'rms.pdf', dpi=300)
    plt.close(fig)
    with (output/'coefficients.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['m', 'n', 'imposed_waves', *[f'{label}_waves' for label in adjusted]])
        writer.writerows([int(m), int(n), truth, *[values[-1, i] for values in adjusted.values()]]
                         for i, ((m, n), truth) in enumerate(zip(target['modes'], target['imposed_for_validation'])))
    data = dict(title=title, input_path=str(target['input_path'].resolve()),
                results=str(target['results'].resolve()), iterations=iterations,
                final_errors={label: float(values[-1]) for label, values in errors.items()},
                error_histories=errors, final_estimates={label: values[-1] for label, values in adjusted.items()},
                baseline_vectors=vectors, modes=target['modes'], imposed_waves=target['imposed_for_validation'],
                controls={key: dict(input_path=str(control['input_path'].resolve()),
                    results=str(control['results'].resolve()),
                    same_system_ao_map=bool(np.array_equal(target['system_ao_phase_slm'], control['system_ao_phase_slm'])))
                    for key, control in controls.items()})
    write_json(output/'comparison.json', data)
    return data


def build_report(root=ROOT, output=None):
    root = Path(root)
    output = Path(output) if output else root/'baseline_comparison'
    output.mkdir(parents=True, exist_ok=True)
    cases, groups = [], []
    for gi, (group_label, fits) in enumerate(discover_groups(root)):
        controls = {key: fits[key] for key in BASELINES if key in fits}
        if not controls:
            continue
        group = dict(label=group_label, missing_controls=[key for key in BASELINES if key not in controls],
                     controls={key: dict(rms=float(zernike.zernike_RMS(
                         zernike.Aberration(fit['modes'], fit['estimate']), fit['alpha'])),
                         input_path=str(fit['input_path'].resolve())) for key, fit in controls.items()})
        if len(controls) == 2:
            first = controls['system_AO']
            difference = first['estimate']-baseline_vector(first, controls['no_aberration'])
            group['control_difference_rms'] = float(zernike.zernike_RMS(
                zernike.Aberration(first['modes'], difference), first['alpha']))
        groups.append(group)
        for tag, fit in fits.items():
            if tag in BASELINES:
                continue
            label = LABELS.get(tag, tag)
            folder = f'group{gi}/{tag}'
            print(f'Rendering {group_label}: {label}', flush=True)
            data = export_case(fit, controls, output/folder, label)
            cases.append(dict(group=group_label, label=label, folder=folder, **data))
    if not cases:
        raise ValueError('No downloaded target/control pairs found.')
    metadata = dict(notice=NOTICE, generated_at=datetime.now(timezone.utc).isoformat(), groups=groups, cases=cases)
    write_json(output/'summary.json', metadata)
    with (output/'summary.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['batch', 'acquisition', 'original_error_waves', 'minus_no_aberration_error_waves',
                         'minus_system_AO_error_waves'])
        for case in cases:
            writer.writerow([case['group'], case['label'], *[case['final_errors'].get(label, '')
                             for label in ['Original', *BASELINES.values()]]])
    rows = []
    for i, case in enumerate(cases):
        cells = ''.join(f'<td>{case["final_errors"][label]:.4f}</td>' if label in case['final_errors']
                        else '<td>Not downloaded</td>' for label in ['Original', *BASELINES.values()])
        rows.append(f'<tr><td><a href="#viewer" data-case="{i}">{html.escape(case["label"])}</a></td>'
                    f'<td>{html.escape(case["group"])}</td>{cells}</tr>')
    group_rows = []
    for group in groups:
        values = [f'{group["controls"][key]["rms"]:.4f}' if key in group['controls'] else 'Not downloaded'
                  for key in BASELINES]
        difference = group.get('control_difference_rms')
        group_rows.append('<tr><td>'+html.escape(group['label'])+'</td><td>'+values[0]+'</td><td>'+values[1]+
                          '</td><td>'+(f'{difference:.4f}' if difference is not None else 'Unavailable')+'</td></tr>')
    options = ''.join(f'<option value="{i}">{html.escape(case["group"]+" / "+case["label"])}</option>'
                      for i, case in enumerate(cases))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Zero-control subtraction</title>
<style>body{font:15px/1.5 system-ui;background:#f5f7f9;color:#182431;margin:0}main{max-width:1450px;margin:auto;padding:28px}
section{background:white;border:1px solid #dce3e9;border-radius:8px;padding:20px;margin:20px 0}h1{margin:0}p{max-width:1150px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;text-align:right;border-bottom:1px solid #dce3e9}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}a{color:#164e72}select{max-width:100%;padding:10px}
iframe{width:100%;height:1020px;border:0}img{width:100%;max-width:720px}.scroll{overflow:auto}.links{display:flex;gap:20px;margin:12px 0}
small{color:#566575}@media(max-width:700px){main{padding:10px}section{padding:12px}iframe{height:800px}}</style>
<main><h1>Subtracting the zero-aberration estimates</h1><p>__NOTICE__</p>
<p>Similar RMS magnitudes do not imply the same wavefront. Compare the control-to-control phase difference below.
Subtraction often increases error against the known imposed aberration; it does not consistently remove a shared bias.</p>
<section><h2>Final estimation error (waves RMS)</h2><div class="scroll"><table><thead><tr>
<th>Acquisition</th><th>Batch</th><th>Original</th><th>Minus no aberration</th><th>Minus System AO</th>
</tr></thead><tbody>__ROWS__</tbody></table></div></section>
<section id="viewer"><h2>Wavefront and error comparison</h2><label for="case">Acquisition and preprocessing</label><br>
<select id="case">__OPTIONS__</select><p>Use <b>Next / Previous</b> inside the viewer to switch the final phase maps:
original, minus no-aberration control, minus System AO control. All available error curves appear together.</p>
<div class="links"><a id="open">Open comparison</a><a id="pdf">RMS PDF</a><a id="coefficients">Coefficients CSV</a><a id="json">Data JSON</a></div>
<iframe id="replay" title="Final wavefront comparison"></iframe>
<h3>Error through saved iterations</h3><p>These curves use actual saved iterates, with a fixed final control estimate subtracted at each step.
The optimizer was not rerun. No synthetic zero-estimate prior is included.</p><img id="rms" alt="RMS estimation error with both baseline subtractions"></section>
<section><h2>Do the zero controls agree?</h2><div class="scroll"><table><thead><tr><th>Batch</th>
<th>No-aberration RMS</th><th>System AO RMS</th><th>RMS of their wavefront difference</th></tr></thead><tbody>__GROUPS__</tbody></table></div>
<p><small>All values are waves RMS on the same microscope pupil. Different acquisitions can retain different SNR-selected images.
The original no-aberration acquisition also lacks its invalid +2 um plane. Missing control solves are not borrowed from another batch.</small></p></section>
<p><a href="summary.csv">All final errors (CSV)</a> · <a href="summary.json">All data and baseline provenance (JSON)</a></p></main>
<script type="application/json" id="data">__DATA__</script><script>
const cases=JSON.parse(document.getElementById('data').textContent),select=document.getElementById('case');
function choose(i){select.value=String(i);const folder=cases[i].folder;
document.getElementById('replay').src=folder+'/comparison.html';document.getElementById('rms').src=folder+'/rms.png';
for(const [id,file] of Object.entries({open:'comparison.html',pdf:'rms.pdf',coefficients:'coefficients.csv',json:'comparison.json'}))document.getElementById(id).href=folder+'/'+file;}
select.addEventListener('change',()=>choose(Number(select.value)));
document.querySelectorAll('[data-case]').forEach(a=>a.addEventListener('click',()=>choose(Number(a.dataset.case))));
choose(__FIRST__);</script></html>'''
    # Open on the most recent full-resolution SNR-10 spherical comparison.
    first = next((i for i, c in enumerate(cases) if c['group'].startswith('SNR >= 10') and 'Spherical' in c['label']), 0)
    for token, value in {'__NOTICE__': html.escape(NOTICE), '__ROWS__': ''.join(rows),
                         '__GROUPS__': ''.join(group_rows), '__OPTIONS__': options,
                         '__DATA__': json.dumps([dict(folder=c['folder']) for c in cases]).replace('<', '\\u003c'),
                         '__FIRST__': str(first)}.items():
        page = page.replace(token, value)
    (output/'index.html').write_text(page)
    print(output/'index.html', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    build_report(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
