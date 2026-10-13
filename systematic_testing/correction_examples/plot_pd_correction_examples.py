"""Four-state, zero-diversity z=0 images replayed from the saved PD sweep."""
from pathlib import Path
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils import psf, plotting, zernike

DEFAULT_DATA = ROOT / 'systematic_testing/rms_snr_sweep/results/pd_rms_snr_sweep_random100_cumulative_scheduleddiversities_aberrated_z0_snr_rms030.npz'
# RMS, calibration SNR, zero-based trial: immediate, delayed, low-SNR, failed correction.
DEFAULT_CASES = [(0.15, 100., 0), (0.15, 100., 1), (0.30, 5., 2), (0.30, 100., 2)]


def plot_correction_example(data, initial_rms, snr, trial, *, noisy=False, acquired_planes=False,
                            sample=None, example_title=None):
    """Return a four-panel figure; noise is a new illustration, not archived acquisition."""
    ri = np.flatnonzero(np.isclose(data['initial_rms'], initial_rms))
    si = np.flatnonzero(np.isclose(data['target_snr'], snr))
    if len(ri) != 1 or len(si) != 1:
        raise ValueError('Select a unique saved RMS and SNR.')
    ri, si = int(ri[0]), int(si[0])
    if not 0 <= trial < data['truth_strengths'].shape[1]:
        raise ValueError('Trial index is out of range.')
    commands = data['slm_strengths'][ri, si, trial]
    if commands.shape[0] != 3 or not np.all(np.isfinite(commands)):
        raise ValueError('Example requires three completed correction loops.')
    microscope = psf.Microscope(N_order=3, lambd=1.3e-3, n=1.333,
                               num_apt=1.05, f=7.2, mag=4, w_0=3.5,
                               L_bfp=15.12, grid_bfp=32)
    grid = psf.Centered_Square_Grid(L_ffp=0.0065, grid_ffp=32, z_level=0)
    if sample is None:
        sample = psf.bead_img_3D(grid, data['sample_z'], [0], [0], [0], [0.001])
    # Commands already carry the negative correction sign and are cumulative.
    coefficients = data['truth_strengths'][ri, trial] + np.vstack([np.zeros(commands.shape[1]), commands])
    aberrations = [zernike.Aberration(data['modes'], c) for c in coefficients]
    focal_z = np.asarray(data["focal_positions"] if acquired_planes else [0.0])
    images = np.array([microscope.compute_image_3D(
        sample, a, focal_z, mode='vector', diversities=None)[3] for a in aberrations])
    images = images.transpose(1, 0, 2, 3)  # focus rows, correction-state columns
    if noisy:
        exposure = float(data['exposures'][ri, si, trial])
        rng = np.random.default_rng(np.random.SeedSequence([int(data['noise_seed']), ri, si, trial, 9902]))
        images = rng.poisson(exposure * np.maximum(images, 0)) / exposure
    peak = float(images.max())
    images = images / peak if peak > 0 else np.zeros_like(images)
    maps = [a.construct_map(microscope.alpha) for a in aberrations]
    phase_limit = 2 * np.pi
    with plt.style.context('bmh'), plt.rc_context({
        'text.usetex': True, 'font.size': 9 if acquired_planes else 12, 'figure.dpi': 300,
        'figure.figsize': [548 / 72, 390 / 72],
    }):
        nr = len(focal_z)
        fig, axes = plt.subplots(nr, 4, figsize=(6, 6) if acquired_planes else (13.5, 4.6), squeeze=False)
        rms = np.r_[initial_rms, data['residual_rms'][ri, si, trial]]
        for i, ax in enumerate(axes[0]):
            title = 'Before correction' if i == 0 else f'After loop {i}'
            if acquired_planes and i == 3:
                title += ' (sim.)'
            ax.set_title(title + f'\nRMS {rms[i]:.4f} waves', fontsize=7 if acquired_planes else 12)
        fig.suptitle(example_title or f'Initial RMS {initial_rms:g} waves | calibration SNR {snr:g} | trial {trial+1}',
                     fontsize=10, y=.98)
        # Square axes slots avoid horizontal whitespace from equal-aspect images.
        fig.subplots_adjust(left=.13, right=.85, bottom=.17 if nr > 1 else .22,
                            top=.89 if nr > 1 else .77, wspace=.04, hspace=.04)
        plotting.many_composite(fig, axes, *grid.get_xy(), images.reshape(-1, *images.shape[-2:]),
                                maps*nr, microscope.alpha, vmin=-phase_limit, vmax=phase_limit)
        for row in range(nr):
            for col in range(4):
                ax = axes[row, col]
                ax.tick_params(axis='both', labelrotation=0, labelleft=col == 0, labelbottom=row == nr-1)
                ax.set_xticks([-.002, 0, .002])
                ax.set_yticks([-.002, 0, .002])
                if acquired_planes:
                    ax.set_xticklabels(['-2', '0', '2'])
                    ax.set_yticklabels(['-2', '0', '2'])
                ax.set_xlabel((r'x [$\mu$m]' if acquired_planes else 'x [mm]') if row == nr-1 else '')
                ax.set_ylabel((f'z = {focal_z[row]*1000:+.2f} '+r'$\mu$m'+('\ny [$\mu$m]' if acquired_planes else '\ny [mm]')) if col == 0 else '', fontsize=7 if acquired_planes else 12)
        for image_ax in axes.ravel():
            image_ax.images[0].set_clim(0, 1)
        # Normalize once across all focuses and correction states, preserving relative brightness.
        fig.colorbar(axes[0, 0].images[0], cax=fig.add_axes([.88, .27, .015, .43]),
                     label='Normalized intensity')
        for inset_ax in fig.axes[nr*4:nr*8]:
            inset_ax.collections[0].set_clim(-2 * np.pi, 2 * np.pi)
            inset_ax.collections[0].set_rasterized(True)
        phase_ax = fig.axes[nr*4]
        phase_colorbar = fig.colorbar(phase_ax.collections[0],
                     cax=fig.add_axes([.30, .075 if acquired_planes else .075, .40, .018 if acquired_planes else .025]), orientation='horizontal',
                     ticks=[-2*np.pi, -np.pi, 0, np.pi, 2*np.pi],
                     label='Residual pupil phase [rad]')
        phase_colorbar.ax.set_xticklabels(
            [r'$-2\pi$', r'$-\pi$', '$0$', r'$\pi$', r'$2\pi$'])
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'figures' / 'pd_correction_examples')
    parser.add_argument('--case', nargs=3, type=float, action='append', metavar=('RMS', 'SNR', 'TRIAL'),
                        help='Repeat for custom examples; TRIAL is zero-based.')
    parser.add_argument('--noisy', action='store_true', help='Add fresh Poisson noise at saved exposure.')
    parser.add_argument('--acquired-planes', action='store_true', help='Show all four saved acquisition focal positions.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.data, allow_pickle=False) as data:
        for rms, snr, trial in args.case or DEFAULT_CASES:
            if int(trial) != trial:
                raise ValueError('Trial must be an integer.')
            fig, _ = plot_correction_example(data, rms, snr, int(trial), noisy=args.noisy, acquired_planes=args.acquired_planes)
            stem = f'rms{rms:g}_snr{snr:g}_trial{int(trial)+1}' + ('_noisy' if args.noisy else '') + ('_acquired_planes' if args.acquired_planes else '')
            for extension in ('png', 'pdf'):
                fig.savefig(args.output / f'{stem}.{extension}', dpi=300)
            plt.close(fig)
            print(args.output / f'{stem}.png')


if __name__ == '__main__':
    main()
