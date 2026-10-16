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
                            sample=None, example_title=None, precomputed_images=None,
                            microscope=None, grid=None, use_tex=True):
    """Return the shared four-state figure with optional saved acquisition images.

    precomputed_images has shape (focus, four correction states, x, y), and
    bypasses forward simulation/noise. Explicit geometry supports a higher
    image resolution without changing the plotting style.
    """
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
    if microscope is None:
        microscope = psf.Microscope(N_order=3, lambd=1.3e-3, n=1.333,
                               num_apt=1.05, f=7.2, mag=4, w_0=3.5,
                               L_bfp=15.12, grid_bfp=32)
    if grid is None:
        grid = psf.Centered_Square_Grid(L_ffp=0.0065, grid_ffp=32, z_level=0)
    if sample is None and precomputed_images is None:
        sample = psf.bead_img_3D(grid, data['sample_z'], [0], [0], [0], [0.001])
    # Commands already carry the negative correction sign and are cumulative.
    coefficients = data['truth_strengths'][ri, trial] + np.vstack([np.zeros(commands.shape[1]), commands])
    aberrations = [zernike.Aberration(data['modes'], c) for c in coefficients]
    focal_z = np.asarray(data["focal_positions"] if acquired_planes else [0.0])
    if precomputed_images is None:
        images = np.array([microscope.compute_image_3D(
            sample, a, focal_z, mode='vector', diversities=None)[3] for a in aberrations])
        images = images.transpose(1, 0, 2, 3)  # focus rows, correction-state columns
    else:
        if noisy:
            raise ValueError('Saved precomputed images already specify their noise realization.')
        images = np.asarray(precomputed_images)
        if (images.shape != (len(focal_z), 4, grid.grid_ffp_x, grid.grid_ffp_y)
                or not np.all(np.isfinite(images))):
            raise ValueError('precomputed_images must match the focal positions, four states and lateral grid.')
    if noisy:
        exposure = float(data['exposures'][ri, si, trial])
        rng = np.random.default_rng(np.random.SeedSequence([int(data['noise_seed']), ri, si, trial, 9902]))
        images = rng.poisson(exposure * np.maximum(images, 0)) / exposure
    peak = float(images.max())
    images = images / peak if peak > 0 else np.zeros_like(images)
    maps = [a.construct_map(microscope.alpha) for a in aberrations]
    phase_limit = 2 * np.pi
    with plt.style.context('bmh'), plt.rc_context({
        'text.usetex': use_tex, 'font.size': 9 if acquired_planes else 12, 'figure.dpi': 300,
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



def plot_acquired_diversities(grid, images, phases, focal_z, alpha, *,
                              column_titles, title, intensity_scale=None, use_tex=True,
                              panel_titles=None):
    """Saved probe frames in the established acquired-plane layout.

    images has axes (focus, probe, x, y); phases contains one Aberration per
    frame in the same row-major order. A supplied intensity_scale is shared
    across pages. Insets display the total simulated pupil phase, including
    the true residual and the applied probe. No images are simulated here.
    """
    images = np.asarray(images)
    nr, nc = images.shape[:2]
    if panel_titles is not None and np.shape(panel_titles) != (nr,nc):
        raise ValueError('panel_titles must match the focus/probe grid.')
    if (images.ndim != 4 or not 1 <= nr <= 4 or not 1 <= nc <= 4
            or len(focal_z) != nr or len(column_titles) != nc or len(phases) != nr*nc
            or not np.all(np.isfinite(images))):
        raise ValueError('Provide finite acquired frames, matched phases, focuses and probe titles.')
    scale = float(images.max()) if intensity_scale is None else float(intensity_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Shared intensity scale must be positive and finite.')
    with plt.style.context('bmh'), plt.rc_context({
        'text.usetex': use_tex, 'font.size': 9, 'figure.dpi': 300,
    }):
        fig, axes = plt.subplots(nr, nc, figsize=(6, 6), squeeze=False)
        fig.suptitle(title, fontsize=10, y=.98)
        fig.subplots_adjust(left=.13, right=.85, bottom=.17, top=.89,
                            wspace=.04, hspace=.32 if panel_titles is not None else .04)
        plotting.many_composite(fig, axes, *grid.get_xy(),
            (images/scale).reshape(-1, *images.shape[-2:]),
            [phase.construct_map(alpha) for phase in phases], alpha,
            vmin=-2*np.pi, vmax=2*np.pi)
        for col, label in enumerate(column_titles):
            axes[0, col].set_title(label, fontsize=8)
        for row in range(nr):
            for col in range(nc):
                ax = axes[row, col]
                if panel_titles is not None:
                    ax.set_title(panel_titles[row][col],fontsize=7)
                ax.images[0].set_clim(0, 1)
                ax.tick_params(axis='both', labelrotation=0, labelleft=col == 0,
                               labelbottom=row == nr-1)
                ax.set_xticks([-.002, 0, .002], labels=['-2', '0', '2'])
                ax.set_yticks([-.002, 0, .002], labels=['-2', '0', '2'])
                ax.set_xlabel(r'x [$\mu$m]' if row == nr-1 else '')
                ax.set_ylabel((f'z = {focal_z[row]*1000:+.2f} '+r'$\mu$m'+'\n'+r'y [$\mu$m]')
                              if col == 0 else '', fontsize=7)
        fig.colorbar(axes[0, 0].images[0], cax=fig.add_axes([.88, .27, .015, .43]),
                     label='Shared normalized intensity')
        for inset in fig.axes[nr*nc:nr*nc*2]:
            inset.collections[0].set_rasterized(True)
        phasebar = fig.colorbar(fig.axes[nr*nc].collections[0],
            cax=fig.add_axes([.30, .075, .40, .018]), orientation='horizontal',
            ticks=[-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], label='Total pupil phase [rad]')
        phasebar.ax.set_xticklabels([r'$-2\pi$', r'$-\pi$', '$0$', r'$\pi$', r'$2\pi$'])
    return fig, axes


def plot_acquisition_gallery(grid, images, commands, labels, alpha, *, title='', shared_max=1., use_tex=True,
                             intensity_label='Intensity / acquisition-wide peak'):
    """One acquired-plane contact sheet, preserving the shared image/phase style."""
    if len(images)!=len(commands) or len(images)!=len(labels) or not len(images):
        raise ValueError('Provide one command and label for every acquired image.')
    columns=int(np.ceil(np.sqrt(len(images))));rows=int(np.ceil(len(images)/columns))
    with plt.style.context('bmh'), plt.rc_context({'text.usetex':use_tex,'font.size':6,'figure.dpi':300}):
        fig,axes=plt.subplots(rows,columns,figsize=(6,6),squeeze=False)
        fig.suptitle(title,fontsize=10,y=.98)
        fig.subplots_adjust(left=.09,right=.89,bottom=.17,top=.89,wspace=.25,hspace=.65)
        active=axes.ravel()[:len(images)]
        plotting.many_composite(fig,active,*grid.get_xy(),images,
            [command.construct_map(alpha) for command in commands],alpha,vmin=-2*np.pi,vmax=2*np.pi)
        for index,(ax,label) in enumerate(zip(active,labels)):
            ax.set_title(label,fontsize=5.5,pad=3)
            ax.images[0].set_clim(0,shared_max)
            ax.set_xticks([-.002,0,.002],labels=['-2','0','2'])
            ax.set_yticks([-.002,0,.002],labels=['-2','0','2'])
            ax.tick_params(axis='both',labelrotation=0,labelsize=5,labelleft=index%columns==0,
                           labelbottom=index//columns==rows-1)
            ax.set_xlabel(r'x [$\mu$m]' if index//columns==rows-1 else '',fontsize=6)
            ax.set_ylabel(r'y [$\mu$m]' if index%columns==0 else '',fontsize=6)
        for ax in axes.ravel()[len(images):]:ax.set_visible(False)
        fig.colorbar(active[0].images[0],cax=fig.add_axes([.925,.30,.015,.40]),
                     label=intensity_label)
        polar=fig.axes[rows*columns]
        for inset in fig.axes[rows*columns:rows*columns+len(images)]:
            inset.collections[0].set_rasterized(True)
        bar=fig.colorbar(polar.collections[0],cax=fig.add_axes([.25,.075,.45,.015]),
            orientation='horizontal',ticks=[-2*np.pi,0,2*np.pi],label='Commanded pupil phase [rad]')
        bar.ax.set_xticklabels([r'$-2\pi$','$0$',r'$2\pi$'])
    return fig,axes


def plot_acquisition_comparison(grid, measured, simulated, commands, focal_z, alpha,
                                *, title='Measured and simulated bead', use_tex=True, inset_loc='upper right',
                                column_titles=('Measured','Simulated')):
    """Shared acquired-plane style, with measured/simulated columns and Z rows.

    Intensities must already share one normalization across the whole run.
    Insets show the known commanded phase in both columns, not an estimate of
    the microscope's unknown native aberration. At most four Z rows per page.
    """
    measured, simulated = np.asarray(measured), np.asarray(simulated)
    if measured.shape != simulated.shape or len(measured) != len(commands):
        raise ValueError('Provide matched measured/simulated planes and commands.')
    nr = len(measured)
    if not 1 <= nr <= 4 or len(focal_z) != nr:
        raise ValueError('Comparison pages require one to four focal planes.')
    images = np.stack([measured, simulated], axis=1)
    maps = [command.construct_map(alpha) for command in commands for _ in range(2)]
    with plt.style.context('bmh'), plt.rc_context({
        'text.usetex': use_tex, 'font.size': 9, 'figure.dpi': 300,
    }):
        fig, axes = plt.subplots(nr, 2, figsize=(6, 6), squeeze=False)
        fig.suptitle(title, fontsize=10)
        fig.subplots_adjust(left=.15, right=.82, bottom=.17, top=.89,
                            wspace=.04, hspace=.06)
        axes[0, 0].set_title(column_titles[0])
        axes[0, 1].set_title(column_titles[1])
        plotting.many_composite(fig, axes, *grid.get_xy(),
                                images.reshape(-1, *images.shape[-2:]), maps,
                                alpha, vmin=-2*np.pi, vmax=2*np.pi, inset_loc=inset_loc)
        for row in range(nr):
            for col in range(2):
                ax = axes[row, col]
                ax.images[0].set_clim(0, 1)
                ax.tick_params(axis='both', labelrotation=0, labelleft=col == 0,
                               labelbottom=row == nr-1)
                ax.set_xticks([-.002, 0, .002], labels=['-2', '0', '2'])
                ax.set_yticks([-.002, 0, .002], labels=['-2', '0', '2'])
                ax.set_xlabel(r'x [$\mu$m]' if row == nr-1 else '')
                ax.set_ylabel((f'z = {focal_z[row]*1000:+.2f} '+r'$\mu$m'+'\n'+r'y [$\mu$m]')
                              if col == 0 else '', fontsize=8)
        fig.colorbar(axes[0, 0].images[0], cax=fig.add_axes([.87, .28, .015, .43]),
                     label='Shared normalized intensity')
        polar = fig.axes[nr*2]
        for inset in fig.axes[nr*2:nr*4]:
            inset.collections[0].set_rasterized(True)
        phasebar = fig.colorbar(polar.collections[0],
            cax=fig.add_axes([.30, .075, .40, .018]), orientation='horizontal',
            ticks=[-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], label='Commanded pupil phase [rad]')
        phasebar.ax.set_xticklabels([r'$-2\pi$', r'$-\pi$', '$0$', r'$\pi$', r'$2\pi$'])
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
