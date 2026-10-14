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
