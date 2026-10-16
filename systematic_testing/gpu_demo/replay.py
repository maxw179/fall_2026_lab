"""Adapt saved GPU AO histories to the existing image/wavefront demonstration."""
from pathlib import Path

import numpy as np

from systematic_testing.ao_animation.viewer import replay_history
from systematic_testing.correction import SimulatedSLM
from systematic_testing.gpu_demo.plot_demo import load_run, _geometry, _sample
from utils import zernike


def load_gpu_case(folder, *, stride=1, focus=0.):
    """Return yesterday's AO viewer frames for an archived GPU demonstration.

    Only four diagnostic images are simulated: before correction and
    after each of the three applied updates. The optimizer and object inverse
    problem are never rerun, and object_history.npy is not needed. Every chosen
    saved estimate and each applied SLM state is replayed in its proper frame.
    Diagnostic Poisson noise uses the archived exposure, fixed across states.
    """
    if not isinstance(stride, (int, np.integer)) or isinstance(stride, (bool, np.bool_)) or stride < 1:
        raise ValueError('stride must be a positive integer.')
    if not np.isfinite(focus):
        raise ValueError('focus must be finite, in mm.')
    metadata, scene, saved = load_run(Path(folder))
    microscope, grid = _geometry(metadata)
    sample = _sample(grid, scene['sample_z'], scene['sample'])
    modes = scene['modes']
    truth = zernike.Aberration(modes, scene['truth_strengths'])
    simulator = SimulatedSLM(microscope, sample, truth, modes, metadata['exposure'],
                             np.random.default_rng(0), optical_mode='vector')
    points = [dict(images_taken=0, slm_strengths=saved['applied_slm_strengths'][0],
                   residual_rms=float(saved['applied_residual_rms'][0]), core_elapsed_seconds=0.)]
    for loop, summary in enumerate(metadata['loops'], 1):
        ids = np.flatnonzero(saved['loop'] == loop)
        ids = ids[np.argsort(saved['iteration'][ids], kind='stable')]
        if not len(ids) or saved['iteration'][ids[-1]] != summary['accepted_steps']:
            raise ValueError('Saved optimizer history does not match the loop summary.')
        selected = sorted(set(ids[::stride]) | {ids[-1]})
        entries = [dict(iteration=int(saved['iteration'][i]), strengths=saved['strengths'][i],
                        loss=float(saved['loss'][i]), core_elapsed_seconds=float(saved['core_elapsed_seconds'][i]))
                   for i in selected]
        previous = saved['applied_slm_strengths'][loop-1]
        command = saved['applied_slm_strengths'][loop]
        expected = previous - metadata['correction_gain']*np.clip(
            saved['strengths'][ids[-1]], -metadata['max_modal_update'], metadata['max_modal_update'])
        np.testing.assert_allclose(command, expected, rtol=2e-9, atol=2e-12)
        np.testing.assert_allclose(saved['acquisition_slm_strengths'][ids],
                                   np.broadcast_to(previous, (len(ids), len(modes))), rtol=2e-9, atol=2e-12)
        applied = truth + zernike.Aberration(modes, command)
        np.testing.assert_allclose(zernike.zernike_RMS(applied, microscope.alpha),
                                   saved['applied_residual_rms'][loop], rtol=2e-9, atol=2e-12)
        points.append(dict(images_taken=int(summary['fit_images']), slm_strengths=command,
                           residual_rms=float(saved['applied_residual_rms'][loop]),
                           core_elapsed_seconds=float(saved['loop_end_core_seconds'][loop]),
                           optimizer_info=dict(history=entries)))
    frames, points = replay_history(points, simulator, focus=focus, max_updates=None)
    if metadata['config'].get('shared_zero', False):
        # Display the archived noisy baseline at a real acquired focal plane.
        focal = np.asarray(metadata['focal_positions_mm'])
        plane = int(np.argmin(abs(focal-focus)))
        with np.load(Path(folder)/'correction_preview.npz', allow_pickle=False) as data:
            observed = data['images'][plane]
        for frame in frames:
            state = frame['progress']
            frame['image'] = observed[state]
            frame['initial_image_title'] = 'Initial noisy zero-diversity frame'
            suffix = ' (simulated diagnostic)' if state == len(points)-1 else ''
            frame['image_title'] = f'Noisy zero-diversity z={focal[plane]*1000:.3g} µm'+suffix
    else:
        # Preserve the requested focus (z=0 was not acquired in this demo).
        # One reproducible noise draw per SLM state prevents optimizer-frame flicker.
        rng = np.random.default_rng(20261014)
        observed = {}
        for frame in frames:
            state = frame['progress']
            if state not in observed:
                clean = np.maximum(frame['image'], 0.)
                observed[state] = rng.poisson(metadata['exposure']*clean)/metadata['exposure']
            frame['image'] = observed[state]
            frame['initial_image_title'] = 'Initial noisy image (sim.)'
            frame['image_title'] = f'Current noisy image (sim.), z={focus*1000:g} µm'
    initial_gain = 30.
    for frame in frames:
        frame['initial_display_gain'] = initial_gain
    title = (f'GPU cumulative AO: {metadata["config"]["beads"]} beads, '
             f'initial RMS {metadata["config"]["initial_rms"]:g}, SNR {metadata["config"]["snr"]:g}')
    return frames, points, title


def load_acquisition_preview(folder):
    """Show every exact first-loop acquisition in the established AO viewer."""
    import json
    from copy import copy
    folder = Path(folder)
    metadata = json.loads((folder/'run.json').read_text())
    with np.load(folder/'scene.npz', allow_pickle=False) as data:
        scene = {k: data[k] for k in data.files}
    microscope, grid = _geometry(metadata)
    sample = _sample(grid, scene['sample_z'], scene['sample'])
    truth = zernike.Aberration(scene['modes'], scene['truth_strengths'])
    simulator = SimulatedSLM(microscope, sample, truth, scene['modes'],
                             metadata['exposure'], np.random.default_rng(0), optical_mode='vector')
    points = [dict(images_taken=0, slm_strengths=np.zeros(len(scene['modes'])),
                   residual_rms=metadata['actual_initial_rms'])]
    base, points = replay_history(points, simulator)
    frames = []
    with np.load(folder/'loop_1_input.npz', allow_pickle=False) as data:
        for i, (image, focus, probe) in enumerate(zip(data['images'], data['focal_z'], data['diversity_strengths'])):
            frame = copy(base[0])
            active = np.flatnonzero(probe)
            label = 'zero diversity' if not len(active) else ', '.join(
                f'{scene["modes"][j].tolist()}: {probe[j]:+.4g} waves coefficient' for j in active)
            frame.update(image=image, count=i+1,
                         label=f'Actual noisy frame {i+1}: z={focus*1000:.3g} µm; {label}',
                         image_title='Actual noisy acquisition', initial_image_title='First noisy acquisition')
            frames.append(frame)
    return frames, points, 'First-loop acquisition preview (before AO fitting)'
