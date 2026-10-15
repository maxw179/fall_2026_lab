"""Plot saved RMS-cliff cases and export complete/partial sweep tables; no fits."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from systematic_testing.correction_examples.plot_pd_iteration_rms import plot_residual_vs_initial_rms
from systematic_testing.gpu_demo.run_demo import save_npz
from systematic_testing.rms_cliff_sweep.run_sweep import SweepConfig


def load_sweep(directory):
    directory = Path(directory)
    paths = sorted(directory.glob('case_*/run.json'))
    if not paths:
        raise ValueError(f'No case_*/run.json found in {directory}.')
    config = json.loads(paths[0].read_text())['config']
    SweepConfig(**config).validate()
    initial = np.asarray(config['rms_values'], dtype=float)
    shape = (len(initial), config['trials'], 3)
    data = dict(initial_rms=initial, residual_rms=np.full(shape, np.nan),
                estimation_rms=np.full(shape, np.nan), completed=np.zeros(shape, dtype=bool),
                optimizer_success=np.zeros(shape, dtype=bool),
                status=np.full(shape[:2], 'missing', dtype='<U16'), config=config)
    seen = set()
    for path in paths:
        meta = json.loads(path.read_text())
        if meta['config'] != config:
            raise ValueError(f'Mixed sweep configurations: {path}. Use one result root per sweep.')
        case = meta['case_index']
        if not 0 <= case < len(initial)*config['trials'] or case in seen:
            raise ValueError(f'Invalid or duplicate case index: {path}.')
        seen.add(case)
        ri, trial = divmod(case, config['trials'])
        if (meta['rms_index'], meta['trial']) != (ri, trial) or not np.isclose(meta['initial_rms'], initial[ri]):
            raise ValueError(f'Case coordinates do not match the sweep: {path}.')
        data['status'][ri, trial] = meta['status']
        results = path.parent/'results.npz'
        if not results.exists():
            continue
        with np.load(results, allow_pickle=False) as saved:
            if (int(saved['case_index']) != case or int(saved['rms_index']) != ri
                    or int(saved['trial']) != trial or not np.isclose(saved['initial_rms'], initial[ri])):
                raise ValueError(f'Result coordinates disagree with metadata: {results}.')
            for name in ('residual_rms', 'estimation_rms', 'completed', 'optimizer_success'):
                if saved[name].shape != (3,):
                    raise ValueError(f'Expected three loop entries in {results}: {name}.')
                data[name][ri, trial] = saved[name]
            for name in ('residual_rms', 'estimation_rms'):
                if not np.array_equal(np.isfinite(saved[name]), saved['completed']):
                    raise ValueError(f'Completion mask disagrees with {name}: {results}.')
    return data


def write_tables(data, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    for ri, initial in enumerate(data['initial_rms']):
        for trial in range(data['config']['trials']):
            for loop in range(3):
                idx = (ri, trial, loop)
                done = bool(data['completed'][idx])
                rows.append(dict(initial_rms_waves=initial, trial=trial, ao_loop=loop+1,
                    status=data['status'][ri, trial], completed=done,
                    residual_rms_waves=data['residual_rms'][idx],
                    estimation_rms_waves=data['estimation_rms'][idx],
                    residual_ratio=data['residual_rms'][idx]/initial,
                    optimizer_success=bool(data['optimizer_success'][idx]) if done else ''))
        for loop in range(3):
            for metric in ('residual_rms', 'estimation_rms'):
                v = data[metric][ri, :, loop]
                v = v[np.isfinite(v)]
                summary.append(dict(initial_rms_waves=initial, ao_loop=loop+1, metric=metric,
                    expected_trials=data['config']['trials'], completed_trials=len(v),
                    missing_trials=data['config']['trials']-len(v),
                    mean_rms_waves=float(v.mean()) if len(v) else np.nan,
                    std_rms_waves=float(v.std(ddof=1)) if len(v) > 1 else np.nan,
                    median_rms_waves=float(np.median(v)) if len(v) else np.nan,
                    fraction_at_least_initial=float(np.mean(v >= initial)) if len(v) else np.nan,
                    converged_trials=int(np.sum(data['optimizer_success'][ri, :, loop]
                                                & data['completed'][ri, :, loop]))))
    for name, records in (('trials.csv', rows), ('summary.csv', summary)):
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    save_npz(output/'sweep.npz', **{k: v for k, v in data.items() if k != 'config'},
             result_axes=['initial_rms', 'trial', 'ao_loop'],
             config_json=json.dumps(data['config'], sort_keys=True))


def discover_diversity_runs(directory):
    """Find downloaded job directories by probe magnitude, including legacy runs."""
    found = {}
    for run in sorted(Path(directory).glob('*')):
        if not run.is_dir():
            continue
        metadata = next(run.glob('case_*/run.json'), None)
        if metadata is None:
            continue
        config = json.loads(metadata.read_text())['config']
        magnitude = SweepConfig(**config).diversity_rms
        found.setdefault(magnitude, []).append(run)
    return found


def plot_diversity_comparison(sweeps, *, metric='residual_rms', use_tex=True, sigma_bands=False):
    """Overlay probe strengths in the established shared-axis AO-loop panels."""
    if metric not in ('residual_rms', 'estimation_rms'):
        raise ValueError('Choose residual_rms or estimation_rms.')
    if not sweeps:
        raise ValueError('No downloaded sweeps to compare.')
    series, reference = {}, None
    for magnitude, data in sorted(sweeps.items()):
        config = asdict(SweepConfig(**data['config']))
        actual = config.pop('diversity_rms')
        # JSON metadata stores the RMS grid as a list, while defaults use tuples.
        config['rms_values'] = tuple(config['rms_values'])
        if not np.isclose(actual, magnitude):
            raise ValueError(f'Diversity label {magnitude} disagrees with saved configuration {actual}.')
        if reference is None:
            reference = config
        elif config != reference:
            raise ValueError('Comparison sweeps differ in settings other than diversity RMS.')
        series[fr'Diversity $\pm {magnitude:.2f}$ waves'] = data[metric]
    first = next(iter(sweeps.values()))
    populated = np.any([np.any(np.isfinite(array), axis=(1, 2))
                        for array in series.values()], axis=0)
    if not populated.any():
        raise ValueError('No completed AO fits are available yet.')
    ylabel = ('Applied residual RMS [waves]' if metric == 'residual_rms'
              else 'Estimation RMS error before clipping [waves]')
    return plot_residual_vs_initial_rms(first['initial_rms'][populated],
        {label: array[populated] for label, array in series.items()},
        loop_indices=[0, 1, 2], ylabel=ylabel, separate_loops=True,
        reference_rms=None, show_uncorrected=True, use_tex=use_tex, sigma_bands=sigma_bands)


def analyze(directory, output=None, *, allow_partial=False, all_loops=False, use_tex=True,
            boxplots=False, log_rms=False, separate_loops=False):
    if separate_loops and boxplots:
        raise ValueError('Separate-loop panels require mean/error-bar plots.')
    data = load_sweep(directory)
    complete = data['completed'].sum()
    expected = data['completed'].size
    if not allow_partial and (complete != expected or not np.all(data['status'] == 'complete')):
        raise ValueError(f'Only {complete}/{expected} completed loops. '
                         'Use --allow-partial to include available results and report missing trials.')
    if not complete:
        raise ValueError('No completed fits to plot; prepared inputs contain no RMS results.')
    output = Path(output) if output is not None else Path(directory)/'analysis'
    write_tables(data, output)
    loops = [0, 1, 2] if all_loops or separate_loops else [2]
    for metric, ylabel, stem in (
            ('residual_rms', 'Applied residual RMS [waves]', 'residual_vs_initial_rms'),
            ('estimation_rms', 'Estimation RMS error before clipping [waves]', 'estimation_vs_initial_rms')):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        suffix = ('_panels' if separate_loops else '')+('_log' if log_rms else '')
        if boxplots:
            available = data[metric][..., loops]
            finite = available[np.isfinite(available)]
            if not len(finite):
                continue
            upper = max(float(finite.max()), float(data['initial_rms'].max()))*1.18
            lower = min(float(finite.min()), float(data['initial_rms'].min()))*.7 if log_rms else 0.
            with PdfPages(output/f'{stem}_boxplots{suffix}.pdf') as pages:
                for loop in loops:
                    if not np.any(np.isfinite(data[metric][..., loop])):
                        continue
                    fig, ax = plot_residual_vs_initial_rms(data['initial_rms'], data[metric],
                        loop_indices=[loop], ylabel=ylabel, use_tex=use_tex,
                        plot_kind='box', log_rms=log_rms, expected_trials=data['config']['trials'])
                    ax.set_ylim(lower, upper)
                    for ext in ('png', 'pdf'):
                        fig.savefig(output/f'{stem}_boxplot_loop_{loop+1}{suffix}.{ext}', dpi=300)
                    pages.savefig(fig, dpi=300)
                    plt.close(fig)
        else:
            populated = np.any(np.isfinite(data[metric][..., loops]), axis=(1, 2))
            if not populated.any():
                continue
            fig, _ = plot_residual_vs_initial_rms(data['initial_rms'][populated], data[metric][populated],
                loop_indices=loops, ylabel=ylabel, use_tex=use_tex, log_rms=log_rms,
                reference_rms=None, show_uncorrected=True, separate_loops=separate_loops)
            for ext in ('png', 'pdf'):
                fig.savefig(output/f'{stem}{suffix}.{ext}', dpi=300)
            plt.close(fig)
    missing = np.flatnonzero(~np.all(data['completed'], axis=-1).ravel())
    print(f'Completed loops: {complete}/{expected}; completed cases: '
          f'{np.sum(data["status"] == "complete")}/{data["status"].size}')
    if len(missing):
        print('Incomplete array task indices:', ','.join(map(str, missing)))
    print(f'Saved figures, trials.csv, summary.csv and sweep.npz to {output.resolve()}')
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Root containing case_0000/, case_0001/, ...')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--all-loops', action='store_true', help='Show all three loops; default is loop 3.')
    parser.add_argument('--separate-loops', action='store_true', help='Show all three AO loops side by side with shared axis scaling.')
    parser.add_argument('--boxplots', action='store_true', help='One box-and-whisker figure per loop, plus a combined PDF.')
    parser.add_argument('--log-rms', action='store_true', help='Also support an explicitly logarithmic residual axis.')
    parser.add_argument('--no-tex', action='store_true', help='Explicit rendering fallback without LaTeX.')
    args = parser.parse_args()
    analyze(args.directory, args.output, allow_partial=args.allow_partial,
            all_loops=args.all_loops, use_tex=not args.no_tex,
            boxplots=args.boxplots, log_rms=args.log_rms, separate_loops=args.separate_loops)


if __name__ == '__main__':
    main()
