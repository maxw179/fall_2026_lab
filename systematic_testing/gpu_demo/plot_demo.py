"""Replay a GPU demo with the existing sweep and testing-notebook plot helpers.

python -m systematic_testing.gpu_demo.plot_demo results/bouchet/gpu_demo/JOBID
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
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


def make_plots(folder, output=None, *, use_tex=True, percentile=92):
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
    if (folder/'object_history.npy').exists():
        from utils.zernike import Aberration
        objects = np.load(folder/'object_history.npy', mmap_mode='r', allow_pickle=False)
        final = _sample(grid, scene['reconstruction_z'], objects[-1])
        with plt.style.context('bmh'), plt.rc_context({
                'text.usetex': use_tex, 'font.size': 12, 'figure.dpi': 300,
                'figure.figsize': [548/72, 390/72]}):
            phase = Aberration(scene['modes'], history['static_estimate_strengths'][-1])
            fig, _ = plotting.zernike_plot(phase.construct_map(microscope.alpha),
                                          microscope.alpha, res=150, show=False)
            save(fig, 'final_aberration')
            fig, _ = plotting.plot_sample_3D(final,
                threshold=float(np.percentile(final.image_mask, percentile)), show=False)
            save(fig, 'final_object')
    return output


def make_animations(folder, *, stride=1, fps=4, percentile=92, dpi=100, use_tex=True):
    """Use the original notebook's polar-phase and 3D-object animation helper.

    Objects come from saved GPU arrays; no object fit or optimization is run.
    Phase frames show the inferred static aberration across all three loops.
    Fixed-duration playback includes the measured core timestamp in each title.
    """
    if stride < 1 or fps <= 0 or not 0 <= percentile <= 100:
        raise ValueError('stride/fps must be positive and percentile must be in [0, 100].')
    folder = Path(folder)
    metadata, scene, history = load_run(folder)
    with np.load(folder/'object_frames.npz', allow_pickle=False) as saved:
        available = saved['history_indices']
    indices = []
    for loop in np.unique(history['loop']):
        ids = np.flatnonzero(history['loop'][available] == loop)
        indices.extend(sorted(set(ids[::stride]) | {ids[-1]}))
    selected = available[indices]
    objects = np.load(folder/'object_history.npy', mmap_mode='r', allow_pickle=False)
    microscope, grid = _geometry(metadata)
    samples = [_sample(grid, scene['reconstruction_z'], objects[index]) for index in indices]
    entries = [dict(iteration=int(history['iteration'][i]), loss=float(history['loss'][i]),
                    strengths=history['static_estimate_strengths'][i],
                    title=f'AO loop {history["loop"][i]}, iteration {history["iteration"][i]}\n'
                          f'Core time {history["core_elapsed_seconds"][i]:.2f} s; '
                          f'RMS error {history["rms_error"][i]:.4f} waves') for i in selected]
    with plt.style.context('bmh'), plt.rc_context({
            'text.usetex': use_tex, 'font.size': 12, 'figure.dpi': 300,
            'figure.figsize': [548/72, 390/72]}):
        return plotting.animate_reconstruction(dict(history=entries), scene['modes'], microscope,
            precomputed_samples=samples, interval=1000/fps, percentile=percentile, dpi=dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--no-animations', action='store_true')
    parser.add_argument('--no-tex', action='store_true', help='Explicit fallback for machines without LaTeX.')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--fps', type=float, default=4)
    parser.add_argument('--percentile', type=float, default=92)
    args = parser.parse_args()
    if args.stride < 1 or args.fps <= 0 or not 0 <= args.percentile <= 100:
        parser.error('stride/fps must be positive and percentile must be in [0, 100]')
    plt.switch_backend('Agg')
    output = make_plots(args.folder, use_tex=not args.no_tex, percentile=args.percentile)
    if not args.no_animations:
        animations = make_animations(args.folder, stride=args.stride, fps=args.fps,
                                     percentile=args.percentile, use_tex=not args.no_tex)
        for name, animation in zip(('aberration', 'object'), animations):
            animation.save(output/f'{name}.gif', writer=PillowWriter(fps=args.fps))
    print(f'Plots saved in {output}')


if __name__ == '__main__':
    main()
