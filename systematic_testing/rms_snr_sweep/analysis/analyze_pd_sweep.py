"""Read-only summary of saved PD sweep results; writes figures and CSV summaries."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    d = np.load(args.data, allow_pickle=False)
    r = d['residual_rms']
    initial = d['initial_rms']
    snr = d['reference_snr']
    rms_axis = np.unique(initial)
    # Duplicate RMS entries are noise replicates of the same wavefront directions.
    mean = np.array([np.nanmean(r[initial == rms], axis=(0, 2)) for rms in rms_axis])
    ratio = mean / rms_axis[:, None, None]
    plt.style.use('bmh')
    plt.rcParams.update({'font.size': 11, 'text.usetex': False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout='constrained')
    norm = plt.Normalize(0, max(1.5, float(np.nanmax(ratio))))
    for loop, ax in enumerate(axes):
        im = ax.imshow(ratio[..., loop], origin='lower', aspect='auto', cmap='viridis', norm=norm)
        ax.set_xticks(np.arange(len(snr)), [f'{v:g}' for v in snr])
        ax.set_yticks(np.arange(len(rms_axis)), [f'{v:g}' for v in rms_axis])
        ax.set(xlabel='Reference SNR', title=f'After AO loop {loop+1}')
        if loop == 0:
            ax.set_ylabel('Initial RMS [waves]')
        ax.grid(False)
        for i in range(len(rms_axis)):
            for j in range(len(snr)):
                v = ratio[i, j, loop]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if norm(v) < .55 else 'black')
    fig.colorbar(im, ax=axes, label='Mean residual / initial RMS (below 1 = improvement)', shrink=.85)
    fig.savefig(args.output/'pd_sweep_residual_ratios.png', dpi=180)
    plt.close(fig)

    cases = [(0, 6, 0), (2, 2, 2), (1, 6, 2), (3, 6, 2)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for i, j, trial in cases:
        y = np.r_[initial[i], r[i, j, trial]]
        axes[0].plot(np.arange(4), y, 'o-', label=f'RMS {initial[i]:g}, SNR {snr[j]:g}, trial {trial+1}, row {i}')
    axes[0].set(xlabel='AO loop (0 = uncorrected)', ylabel='Residual RMS [waves]', xticks=[0, 1, 2, 3])
    axes[0].legend(fontsize=8)
    previous = np.concatenate([np.broadcast_to(initial[:, None, None, None], r[..., :1].shape), r[..., :-1]], axis=-1)
    worsening = np.sum(r > previous, axis=(0, 1, 2))
    improving = np.sum(r < previous, axis=(0, 1, 2))
    axes[1].bar(np.arange(1, 4)-.18, improving, width=.36, label='RMS improved')
    axes[1].bar(np.arange(1, 4)+.18, worsening, width=.36, label='RMS worsened')
    axes[1].set(xlabel='AO loop', ylabel='Number of runs', xticks=[1, 2, 3])
    axes[1].legend()
    fig.savefig(args.output/'pd_sweep_failure_examples.png', dpi=180)
    plt.close(fig)

    truth = d['truth_strengths'][:, None, :, None, :]
    residual_coefficients = truth + d['slm_strengths']
    modal_mse = np.nanmean(residual_coefficients[..., 2, :]**2, axis=(0, 1, 2))
    order = np.argsort(modal_mse)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    ax.bar(np.arange(len(order)), np.sqrt(modal_mse[order]))
    ax.set_xticks(np.arange(len(order)), [str(tuple(map(int, d['modes'][i]))) for i in order], rotation=60)
    ax.set(xlabel='Zernike (m, n)', ylabel='Root mean square coefficient residual [waves]',
           title='Residual modes after AO loop 3; large input aberrations included')
    fig.savefig(args.output/'pd_sweep_modal_errors.png', dpi=180)
    plt.close(fig)
    rows = []
    for i, rms in enumerate(rms_axis):
        for j, reference_snr in enumerate(snr):
            for loop in range(3):
                values = r[initial == rms, j, :, loop].ravel()
                valid = np.isfinite(values)
                rows.append([rms, reference_snr, loop+1, valid.sum(), np.nanmean(values),
                             np.nanstd(values), np.nanmean(values)/rms,
                             np.mean(values[valid] > rms) if valid.any() else np.nan])
    np.savetxt(args.output/'pd_sweep_summary.csv', rows, delimiter=',',
               header='initial_rms,reference_snr,ao_loop,count,mean_residual_rms,std_residual_rms,mean_residual_ratio,fraction_worse_than_initial', comments='')
    print('Completed loops:', np.isfinite(r).sum(), '/', r.size)
    print('Improved per loop:', improving.tolist(), 'Worsened:', worsening.tolist())
    print('Saved analysis to', args.output.resolve())


if __name__ == '__main__':
    main()
