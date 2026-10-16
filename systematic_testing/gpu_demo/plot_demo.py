"""Replay a GPU demo with the existing sweep and testing-notebook plot helpers.

python -m systematic_testing.gpu_demo.plot_demo results/bouchet/gpu_demo/JOBID
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from systematic_testing.correction_examples.plot_pd_correction_examples import plot_correction_example
from systematic_testing.correction_examples.plot_pd_iteration_rms import plot_iteration_rms
from utils import plotting, psf


def load_run(folder):
    folder = Path(folder)
    metadata = json.loads((folder/'run.json').read_text())
    with np.load(folder/'scene.npz', allow_pickle=False) as saved:
        scene = {key: saved[key] for key in saved.files}
    with np.load(folder/'history.npz', allow_pickle=False) as saved:
        history = {key: saved[key] for key in saved.files}
    return metadata, scene, history


def _sweep_adapter(metadata, scene, history):
    """Adapt a single demo to the shared sweep helpers' archive array layout."""
    data = dict(initial_rms=np.array([metadata['config']['initial_rms']]),
                target_snr=np.array([metadata['config']['snr']]), modes=scene['modes'],
                truth_strengths=scene['truth_strengths'][None, None],
                slm_strengths=history['applied_slm_strengths'][1:][None, None, None],
                residual_rms=history['applied_residual_rms'][1:][None, None, None],
                optimizer_steps=np.array([loop['accepted_steps'] for loop in metadata['loops']])[None, None, None],
                loop_end_core_seconds=history['loop_end_core_seconds'][None, None, None],
                focal_positions=np.array(metadata['focal_positions_mm']),
                sample_z=scene['sample_z'], exposures=np.array([[[metadata['exposure']]]]),
                noise_seed=1)
    histories = dict(run_indices=np.zeros((len(history['iteration']), 3), dtype=int),
                     loop_indices=history['loop']-1, iteration=history['iteration'],
                     strengths=history['strengths'], core_elapsed_seconds=history['core_elapsed_seconds'])
    return data, histories


def plot_rms(metadata, scene, history, *, by_time=True, use_tex=True):
    data, histories = _sweep_adapter(metadata, scene, history)
    return plot_iteration_rms(data, metadata['config']['initial_rms'], metadata['config']['snr'],
                              0, histories=histories, time_axis=by_time, use_tex=use_tex)


def _geometry(metadata):
    microscope = psf.Microscope(**metadata['microscope'])
    grid = psf.Centered_Square_Grid(L_ffp=.0065, grid_ffp=metadata['config']['size'], z_level=0)
    return microscope, grid


def _sample(grid, levels, values):
    return psf.Image_Mask_3D([psf.Image_Mask(psf._grid_at_z(grid, z), plane)
                              for z, plane in zip(levels, values)])


def make_plots(folder, output=None, *, use_tex=True):
    """Export RMS and four-state correction figures using the sweep helpers."""
    folder = Path(folder)
    metadata, scene, history = load_run(folder)
    output = Path(output) if output else folder/'figures'
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        for extension in ('png', 'pdf'):
            fig.savefig(output/f'{name}.{extension}', dpi=300)
        plt.close(fig)

    save(plot_rms(metadata, scene, history, by_time=True, use_tex=use_tex)[0], 'rms_time')
    save(plot_rms(metadata, scene, history, by_time=False, use_tex=use_tex)[0], 'rms_iterations')
    microscope, grid = _geometry(metadata)
    if (folder/'correction_preview.npz').exists():
        data, _ = _sweep_adapter(metadata, scene, history)
        with np.load(folder/'correction_preview.npz', allow_pickle=False) as saved:
            images = saved['images']
        title = (f'Initial RMS {metadata["config"]["initial_rms"]:g} waves | '
                 f'{metadata["config"]["beads"]} beads | '
                 f'calibration SNR {metadata["config"]["snr"]:g} | '
                 f'trial {metadata["config"]["trial"]+1}')
        fig, _ = plot_correction_example(data, metadata['config']['initial_rms'],
            metadata['config']['snr'], 0, acquired_planes=True, precomputed_images=images,
            microscope=microscope, grid=grid, example_title=title, use_tex=use_tex)
        save(fig, 'acquired_images')
    if len(history['static_estimate_strengths']):
        from utils.zernike import Aberration
        with plt.style.context('bmh'), plt.rc_context({
                'text.usetex': use_tex, 'font.size': 12, 'figure.dpi': 300,
                'figure.figsize': [548/72, 390/72]}):
            phase = Aberration(scene['modes'], history['static_estimate_strengths'][-1])
            fig, _ = plotting.zernike_plot(phase.construct_map(microscope.alpha),
                                          microscope.alpha, res=150, show=False)
            save(fig, 'final_aberration')
    return output


def make_replay(folder, *, stride=1, focus=0., use_tex=True, show_progress=True,
                output_path=None):
    """Render the existing AO image/wavefront viewer from saved GPU coefficients.

    No voxel rendering, object refit, or optimization is performed. Simulated
    diagnostic images are computed once per applied state (four for three loops),
    with Poisson noise at the archived fixed exposure.
    """
    from systematic_testing.gpu_demo.replay import load_gpu_case
    from systematic_testing.ao_animation.viewer import show_replay
    from tqdm.auto import tqdm

    folder = Path(folder)
    if show_progress:
        print('Preparing simulated images for the saved AO states...', flush=True)
    frames, points, title = load_gpu_case(folder, stride=stride, focus=focus)
    output_path = Path(output_path) if output_path else folder/'figures'/'ao_replay.html'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=len(frames), desc='Rendering image/wavefront replay', unit='frame',
              disable=not show_progress) as progress, plt.style.context('bmh'), plt.rc_context({
            'text.usetex': use_tex, 'font.size': 12}):
        return show_replay(frames, points, title=title, output_path=output_path,
            on_frame=lambda event: progress.update(event['completed']-progress.n))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--no-animations', action='store_true')
    parser.add_argument('--no-tex', action='store_true', help='Explicit fallback for machines without LaTeX.')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--focus', type=float, default=0., help='Diagnostic focus in mm; default z=0.')
    args = parser.parse_args()
    if args.stride < 1 or not np.isfinite(args.focus):
        parser.error('stride must be positive and focus must be finite')
    plt.switch_backend('Agg')
    output = make_plots(args.folder, use_tex=not args.no_tex)
    if not args.no_animations:
        make_replay(args.folder, stride=args.stride, focus=args.focus, use_tex=not args.no_tex)
        print(f'Open {output / "ao_replay.html"} in your browser for playback.')
    print(f'Plots saved in {output}')


if __name__ == '__main__':
    main()


def plot_acquisition_geometry(folder, *, use_tex=True):
    """Saved truth sample and every fresh acquisition diversity, grouped by loop."""
    from utils import zernike
    folder = Path(folder)
    metadata, scene, _ = load_run(folder)
    microscope, grid = _geometry(metadata)
    sample = _sample(grid, scene['sample_z'], scene['sample'])
    positions = np.asarray(metadata['focal_positions_mm'])
    schedules = []
    previous_count = 0
    for summary in metadata['loops']:
        with np.load(folder/f'loop_{summary["loop"]}_input.npz', allow_pickle=False) as data:
            fresh_count = summary['fit_images']-previous_count
            fresh = data['diversity_strengths'][-fresh_count:]
            focal = data['focal_z'][-fresh_count:]
            probes = fresh[np.isclose(focal, positions[0], rtol=0, atol=1e-12)]
            # Historical frames include SLM offsets; fresh phases are the probes.
            for position in positions:
                np.testing.assert_allclose(fresh[np.isclose(focal, position, rtol=0, atol=1e-12)], probes)
            schedules.append(probes)
        previous_count = summary['fit_images']
    columns = max(map(len, schedules))
    with plt.style.context('bmh'), plt.rc_context({'text.usetex': use_tex, 'font.size': 10,
                                                  'figure.dpi': 300}):
        fig = plt.figure(figsize=(20, 8))
        outer = fig.add_gridspec(1, 2, width_ratios=(1.15, 2.5),
                                left=.025, right=.985, bottom=.09, top=.86, wspace=.08)
        ax = fig.add_subplot(outer[0], projection='3d')
        plotting.plot_sample_3D(sample, threshold=0., alpha=.32, ax=ax, show=False)
        x, y = sample.get_xy()
        xx, yy = np.meshgrid([x.min(), x.max()], [y.min(), y.max()])
        for level in positions:
            ax.plot_surface(xx, yy, np.full_like(xx, level), color='red',
                            alpha=.13, shade=False, edgecolor='red', linewidth=.7)
        from matplotlib.ticker import FuncFormatter
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v*1000:g}'))
        ax.set_xlabel(r'$x$ ($\mu$m)')
        ax.set_ylabel(r'$y$ ($\mu$m)')
        ax.set_zlabel(r'$z$ ($\mu$m)')
        ax.view_init(elev=23, azim=-55)
        panels = outer[1].subgridspec(len(schedules), columns, wspace=.28, hspace=.5)
        phase_axes = []
        for row, probes in enumerate(schedules):
            for col, strengths in enumerate(probes):
                polar = fig.add_subplot(panels[row, col], projection='polar')
                aberration = zernike.Aberration(scene['modes'], strengths)
                plotting.zernike_plot(aberration.construct_map(microscope.alpha),
                                      microscope.alpha, res=101, ax=polar, show=False, colorbar=False)
                active = np.flatnonzero(abs(strengths)>1e-12)
                if len(active) == 1:
                    m, n = scene['modes'][active[0]]
                    rms = np.sign(strengths[active[0]])*zernike.zernike_RMS(aberration, microscope.alpha)
                    label = rf'$Z_{{{n}}}^{{{m}}}$: ${rms:+.2f}$ waves'
                elif len(active) == 0:
                    label = 'Zero diversity'
                else:
                    label = f'{zernike.zernike_RMS(aberration, microscope.alpha):.2f} waves RMS'
                polar.set_title(label, fontsize=8, pad=7)
                if col == 0:
                    polar.text(-.28, .5, f'Loop {row+1}', rotation=90,
                               va='center', ha='center', transform=polar.transAxes, fontsize=11)
                phase_axes.append(polar)
        cax = fig.add_axes([.52, .045, .30, .018])
        fig.colorbar(phase_axes[0].collections[0], cax=cax, orientation='horizontal',
                     ticks=[-np.pi, 0, np.pi], label='Applied diversity phase (rad)')
        cax.set_xticklabels([r'$-\pi$', '$0$', r'$\pi$'])
    return fig
