"""Plot recovered per-iteration phase-estimation RMS error and applied corrections."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils import zernike
HISTORY_PATH = ROOT / "systematic_testing/rms_snr_sweep/results/pd_rms_snr_sweep_random100_iteration_histories.npz"


def plot_error_histories(iterations, series, *, title='', use_tex=True,
                         xlabel='Saved optimizer iteration', ylabel='Estimation error RMS [waves]',
                         nonnegative=True, integer_x=True):
    """Compare precomputed wavefront-error traces in the shared RMS style.

    Series maps labels to signed-wavefront RMS errors, evaluated at the supplied
    saved optimizer iterations. No averaging or synthetic prior is introduced.
    """
    iterations = np.asarray(iterations)
    if iterations.ndim != 1 or not len(iterations) or not series:
        raise ValueError('Supply saved iterations and at least one error history.')
    if any(np.shape(y) != iterations.shape or not np.all(np.isfinite(y))
           or (nonnegative and np.any(np.asarray(y) < 0)) for y in series.values()):
        raise ValueError('Histories must be finite, match iterations, and obey nonnegative=True.')
    with plt.style.context('bmh'), plt.rc_context({'text.usetex': use_tex, 'font.size': 9, 'figure.dpi': 300}):
        fig, ax = plt.subplots(figsize=(6, 6))
        for label, values in series.items():
            ax.plot(iterations, values, '.-', label=label, markersize=4)
        lo=min(min(y) for y in series.values());hi=max(max(y) for y in series.values())
        limits=(0,hi*1.15+1e-12) if nonnegative else (lo-.1*(hi-lo+1e-12),hi+.15*(hi-lo+1e-12))
        ax.set(xlabel=xlabel,ylabel=ylabel,title=title,ylim=limits)
        if integer_x:ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.23), fontsize=8)
        fig.subplots_adjust(left=.15, right=.95, bottom=.13, top=.79)
    return fig, ax


def plot_residual_vs_initial_rms(initial_rms, residual_rms, *, loop_indices=None,
                                 ylabel='Residual wavefront RMS [waves]',
                                 reference_rms=.30, use_tex=True, plot_kind='mean',
                                 log_rms=False, expected_trials=None, show_uncorrected=True,
                                 separate_loops=False, sigma_bands=False):
    """Plot paired RMS sweeps using the shared 6-inch RMS figure conventions.

    residual_rms has axes (initial RMS, trial, AO loop). Each mean point averages
    the available aberration cases; error bars show +/- one sample SD across those
    cases, including every finite result regardless of optimizer convergence.
    A single available case has a mean point but no estimable SD error bar.
    plot_kind='box' shows quartiles, median, 1.5-IQR whiskers and all outliers,
    with finite sample counts. log_rms adds an explicitly logarithmic Y axis.
    separate_loops puts each mean curve on a separate panel with shared axes.
    For comparisons, residual_rms may map series labels to arrays of that same
    shape. Series use consistent colors across AO panels and separate error bars.
    sigma_bands replaces error bars with darker +/-1 SD and lighter +/-2 SD bands.
    """
    initial = np.asarray(initial_rms, dtype=float)
    comparison = isinstance(residual_rms, dict)
    series = [(label, np.asarray(array, dtype=float)) for label, array in residual_rms.items()] \
        if comparison else [(None, np.asarray(residual_rms, dtype=float))]
    if not series:
        raise ValueError('Supply at least one residual series.')
    values = series[0][1]
    if any(array.shape != values.shape for _, array in series):
        raise ValueError('Comparison series must have the same RMS, trial and AO loop axes.')
    if comparison and plot_kind != 'mean':
        raise ValueError('Comparison series require mean/error-bar plots.')
    if values.ndim != 3 or values.shape[0] != len(initial):
        raise ValueError('Expected residual axes (initial RMS, trial, AO loop).')
    if plot_kind not in ('mean', 'box'):
        raise ValueError('plot_kind must be mean or box.')
    if sigma_bands and plot_kind != 'mean':
        raise ValueError('Sigma bands require mean plots.')
    if separate_loops and plot_kind != 'mean':
        raise ValueError('Separate-loop panels require mean/error-bar plots.')
    loops = list(range(values.shape[-1])) if loop_indices is None else list(loop_indices)
    if not loops or any(i < 0 or i >= values.shape[-1] for i in loops):
        raise ValueError('Select saved zero-based AO loop indices.')
    # Resolve colors before the style context so patches and lines retain the
    # same sweep colors when the figure is saved after leaving that context.
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    with plt.style.context('bmh'), plt.rc_context({'text.usetex': use_tex, 'font.size': 9, 'figure.dpi': 300}):
        if separate_loops:
            fig, axes = plt.subplots(1, len(loops), figsize=(6*len(loops), 6),
                                     sharex=True, sharey=True, squeeze=False)
            axes = axes[0]
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            axes = np.array([ax])
        for li, loop in enumerate(loops):
            ax = axes[li] if separate_loops else axes[0]
            if separate_loops:
                ax.set_title(f'After {loop+1} AO '+('loop' if loop == 0 else 'loops'), fontsize=10)
            for si, (series_label, series_values) in enumerate(series):
                v = series_values[..., loop]
                color = palette[(si if comparison else loop) % len(palette)]
                if plot_kind == 'box':
                    available = np.flatnonzero(np.any(np.isfinite(v), axis=1))
                    if not len(available):
                        continue
                    spacing = float(np.min(np.diff(initial))) if len(initial) > 1 else float(initial[0])
                    width = .65*spacing/len(loops)
                    offset = (li-(len(loops)-1)/2)*width
                    groups = [row[np.isfinite(row)] for row in v[available]]
                    artists = ax.boxplot(groups, positions=initial[available]+offset,
                        widths=.8*width, manage_ticks=False, whis=1.5, patch_artist=True,
                        showfliers=True, boxprops=dict(facecolor=color, edgecolor=color, alpha=.35),
                        medianprops=dict(color='black', linewidth=1.2),
                        whiskerprops=dict(color=color), capprops=dict(color=color),
                        flierprops=dict(marker='o', markersize=3, markerfacecolor='none',
                                        markeredgecolor=color, alpha=.9))
                    artists['boxes'][0].set_label(f'Loop {loop+1} residual distribution')
                    for x, group in zip(initial[available]+offset, groups):
                        label = (f'n={len(group)}' if expected_trials is None else
                                 f'{len(group)}/{expected_trials}')
                        ax.text(x, 1.005, label, transform=ax.get_xaxis_transform(),
                                ha='center', va='bottom', rotation=90, fontsize=7, color=color)
                    continue
                mean = np.array([row[np.isfinite(row)].mean() if np.isfinite(row).any() else np.nan
                                 for row in v])
                std = np.array([row[np.isfinite(row)].std(ddof=1) if np.isfinite(row).sum() > 1 else np.nan
                                for row in v])
                label = series_label if comparison else fr'Loop {loop+1} mean $\pm 1\sigma$'
                if sigma_bands:
                    ax.fill_between(initial, mean-2*std, mean+2*std, color=color,
                                    alpha=.10, linewidth=0)
                    ax.fill_between(initial, mean-std, mean+std, color=color,
                                    alpha=.25, linewidth=0)
                    ax.plot(initial, mean, 'o-', color=color, markersize=4, label=label)
                else:
                    ax.errorbar(initial, mean, yerr=std, fmt='o-', color=color,
                                markersize=4, capsize=3, elinewidth=1, label=label)
        diagonal = np.linspace(initial.min()*.01 if log_rms else 0., initial.max(), 512)
        if log_rms:
            all_values = np.concatenate([array[..., loops].ravel() for _, array in series])
            finite = all_values[np.isfinite(all_values)]
            if not len(finite) or np.any(finite <= 0):
                plt.close(fig)
                raise ValueError('A logarithmic RMS axis requires positive finite residuals.')
        for ai, ax in enumerate(axes):
            if show_uncorrected:
                ax.plot(diagonal, diagonal, '--', color='black',
                        linewidth=1, label='Uncorrected RMS')
            if reference_rms is not None:
                ax.axvline(reference_rms, color='black', linestyle=':', linewidth=1,
                           label=f'{reference_rms:g}-wave reference')
            if log_rms:
                ax.set_yscale('log')
                ax.set_ylim(bottom=min(float(finite.min()), float(initial.min()))*.7)
            else:
                ax.set_ylim(bottom=0)
            ax.set(xlabel='Initial wavefront RMS [waves]',
                   ylabel=(ylabel+' (log scale)' if log_rms else ylabel) if ai == 0 else '',
                   xlim=(0, initial.max()*1.04))
        if separate_loops:
            legend = {}
            for ax in axes:
                handles, labels = ax.get_legend_handles_labels()
                legend.update(zip(labels, handles))
            if sigma_bands:
                from matplotlib.patches import Patch
                legend[r'$\pm 1\sigma$ (darker)'] = Patch(facecolor='gray', alpha=.25)
                legend[r'$\pm 2\sigma$ (lighter)'] = Patch(facecolor='gray', alpha=.10)
            fig.legend(legend.values(), legend.keys(), loc='upper center',
                       bbox_to_anchor=(.5, .99), ncol=len(legend), fontsize=8)
            fig.subplots_adjust(left=.055, right=.985, bottom=.13, top=.84, wspace=.08)
        else:
            axes[0].legend(loc='upper center', bbox_to_anchor=(.5, 1.16), ncol=2, fontsize=8)
            fig.subplots_adjust(left=.15, right=.95, bottom=.13, top=.84)
    return fig, axes if separate_loops else axes[0]


def plot_iteration_rms(data, initial_rms, snr, trial, *, histories=None,
                       time_axis=False, use_tex=True):
    """Plot sweep histories with the established example style.

    time_axis uses saved core_elapsed_seconds and loop_end_core_seconds while
    retaining the same layout, colors, boundary annotations and typography.
    use_tex=False is an explicit option for render tests without LaTeX.
    """
    if histories is None:
        with np.load(HISTORY_PATH, allow_pickle=False) as archive:
            return plot_iteration_rms(data, initial_rms, snr, trial, histories=archive,
                                      time_axis=time_axis, use_tex=use_tex)
    ri = int(np.flatnonzero(np.isclose(data['initial_rms'], initial_rms))[0])
    si = int(np.flatnonzero(np.isclose(data['target_snr'], snr))[0])
    steps = np.asarray(data['optimizer_steps'][ri, si, trial], dtype=int)
    residual = np.r_[initial_rms, data['residual_rms'][ri, si, trial]]
    if not np.all(np.isfinite(residual)):
        raise ValueError('All correction loops must be complete.')
    boundaries = (np.asarray(data['loop_end_core_seconds'][ri, si, trial]) if time_axis
                  else np.r_[0, np.cumsum(steps)])
    with plt.style.context('bmh'), plt.rc_context({'text.usetex': use_tex, 'font.size': 9, 'figure.dpi': 300}):
        fig, ax = plt.subplots(figsize=(6, 6))
        alpha = np.arcsin(1.05 / 1.333)
        maximum_error = max(residual)
        run_mask = np.all(histories['run_indices'] == [ri, si, trial], axis=1)
        for loop in range(len(steps)):
            selected = np.flatnonzero(run_mask & (histories['loop_indices'] == loop))
            selected = selected[np.argsort(histories['iteration'][selected])]
            iterations = histories['iteration'][selected]
            if not len(selected) or iterations[-1] != steps[loop]:
                raise ValueError('Recovered history does not match saved iteration count.')
            current = data['truth_strengths'][ri, trial].copy()
            if loop:
                current += data['slm_strengths'][ri, si, trial, loop-1]
            truth = zernike.Aberration(data['modes'], current)
            errors = np.array([zernike.zernike_RMS_difference(
                zernike.Aberration(data['modes'], estimate), truth, alpha)
                for estimate in histories['strengths'][selected]])
            xs = histories['core_elapsed_seconds'][selected] if time_axis else boundaries[loop] + iterations
            ax.plot(xs, errors, '.-',
                    label=f'Loop {loop+1} estimation error', markersize=4)
            maximum_error = max(maximum_error, float(errors.max()))
        ax.scatter(boundaries, residual, marker='x', color='black', s=45,
                   zorder=4, label='Applied residual (after clipping)')
        for i, boundary in enumerate(boundaries):
            ax.axvline(boundary, color='black', ls='--', alpha=.5, linewidth=1)
            label = ('1 start' if i == 0 else
                     f'{i} end / {i+1} start' if i < len(steps) else f'{i} end')
            ax.annotate(label, (boundary, 1), xycoords=('data', 'axes fraction'),
                        xytext=(4, -6), textcoords='offset points', rotation=90,
                        ha='left', va='top', fontsize=7)
        ax.set(xlabel='Cumulative core computation time [s]' if time_axis else 'Cumulative optimizer iterations',
               ylabel='RMS error [waves]')
        ax.set_ylim(bottom=0, top=maximum_error*1.25)
        if time_axis:
            margin = .025*max(float(boundaries[-1]), 1e-9)
            ax.set_xlim(-margin, boundaries[-1]+margin)
        else:
            ax.set_xlim(-1, max(1, boundaries[-1])+1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.16), ncol=2, fontsize=8)
        fig.subplots_adjust(left=.15, right=.95, bottom=.13, top=.84)
    return fig, ax


if __name__ == '__main__':
    try:
        from .plot_pd_correction_examples import DEFAULT_DATA, DEFAULT_CASES
    except ImportError:
        from plot_pd_correction_examples import DEFAULT_DATA, DEFAULT_CASES
    output = Path(__file__).parent / 'figures' / 'pd_correction_examples_acquired_planes'
    output.mkdir(parents=True, exist_ok=True)
    cases = DEFAULT_CASES + [(0.15, 5., 0), (0.15, 10., 0), (0.15, 15., 0), (0.30, 10., 2)]
    with np.load(DEFAULT_DATA, allow_pickle=False) as saved:
        for rms, snr, trial in cases:
            fig, _ = plot_iteration_rms(saved, rms, snr, trial)
            stem = f'rms{rms:g}_snr{snr:g}_trial{trial+1}_iteration_rms'
            for ext in ('png', 'pdf'):
                fig.savefig(output / f'{stem}.{ext}', dpi=300)
            plt.close(fig)
            print(stem)
