"""Display saved noisy acquisition frames, grouped by probe mode and focal z."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from systematic_testing.correction_examples.plot_pd_correction_examples import plot_acquired_diversities
from utils import psf, zernike


def plot_acquisitions(case, *, loop=1, output=None, use_tex=True):
    case = Path(case)
    meta = json.loads((case/'run.json').read_text())
    if not 1 <= loop <= meta['num_ao_loops']:
        raise ValueError('Select an acquired AO loop from 1 to 3.')
    with np.load(case/'scene.npz', allow_pickle=False) as scene:
        modes, truth = scene['modes'], scene['truth_strengths']
        alpha = float(scene['alpha'])
    with np.load(case/f'acquisition_{loop}.npz', allow_pickle=False) as saved:
        images = saved['images']
        focal = saved['focal_z']
        probes = saved['diversity_strengths']
        slm = saved['acquisition_slm_strengths']
    positions = np.asarray(meta['focal_z_mm'])
    probe_modes = np.asarray(meta['diversity_modes_by_loop'][loop-1])
    settings = np.asarray(meta['diversity_rms_waves'])
    per_z = len(probe_modes)*len(settings)
    if images.shape != (len(positions)*per_z, meta['config']['size'], meta['config']['size']):
        raise ValueError('Saved frame count does not match the acquisition schedule.')
    np.testing.assert_allclose(focal, np.repeat(positions, per_z))
    grid = psf.Centered_Square_Grid(L_ffp=.0065, grid_ffp=meta['config']['size'], z_level=0.)
    output = Path(output) if output is not None else case/'figures'/'acquisitions'
    output.mkdir(parents=True, exist_ok=True)
    shared_peak = float(images.max())
    preparation = 'Prepared input' if meta['status'] == 'prepared' else 'Saved acquisition'
    title = (f'{preparation}: loop {loop}, initial RMS {meta["initial_rms"]:g} waves\n'
             f'Single bead, calibration SNR {meta["config"]["snr"]:g}, trial {meta["trial"]}')
    indices = np.arange(len(images)).reshape(len(positions), len(probe_modes), len(settings))
    with PdfPages(output/f'loop_{loop}_all_diversities.pdf') as pdf:
        for mi, mode in enumerate(probe_modes):
            selected = indices[:, mi, :]
            # Ensure these are the recorded single-mode, RMS-normalized probes.
            unit = zernike.RMSAberration([mode], [1.], 1., alpha)
            expected = np.zeros((len(settings), len(modes)))
            column = int(np.flatnonzero(np.all(modes == mode, axis=1))[0])
            expected[:, column] = settings*unit.strengths[0]
            np.testing.assert_allclose(probes[selected], np.broadcast_to(expected, probes[selected].shape))
            phases = [zernike.Aberration(modes, truth+slm+p) for p in probes[selected].reshape(-1, len(modes))]
            fig, _ = plot_acquired_diversities(grid, images[selected], phases, positions, alpha,
                column_titles=[f'{value:+.2f} waves' for value in settings],
                title=title+f' | Probe ({mode[0]}, {mode[1]})',
                intensity_scale=shared_peak, use_tex=use_tex)
            stem = f'loop_{loop}_mode_{mode[0]}_{mode[1]}'
            for extension in ('png', 'pdf'):
                fig.savefig(output/f'{stem}.{extension}', dpi=300)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)
            print(output/f'{stem}.png')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path, help='Directory containing run.json and acquisition_*.npz.')
    parser.add_argument('--loop', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-tex', action='store_true')
    args = parser.parse_args()
    plt.switch_backend('Agg')
    plot_acquisitions(args.case, loop=args.loop, output=args.output, use_tex=not args.no_tex)


if __name__ == '__main__':
    main()
