"""Load a saved cumulative SNR-sweep case, without running optimization."""
from pathlib import Path
import numpy as np
from utils import psf, zernike
from .viewer import RecordingSLM, replay_history

RESULT_NAME = 'pd_rms_snr_sweep_random100_cumulative_scheduleddiversities_aberrated_z0_snr_rms030.npz'
ITERATION_NAME = 'pd_rms_snr_sweep_random100_iteration_histories.npz'


def _load_case(results_dir=None, *, rms_index=1, snr_index=3, trial=0,
               iteration_stride=1, acquisition_stride=12, bead_sweep=False):
    """Return frames, correction history and title for one archived case.

    The archive did not store optical geometry: use the matching sweep notebook's
    32x32 setup. Acquisition steps are hidden; no measurement images are regenerated.
    SLM updates and optimizer coefficients come exclusively from saved arrays.
    Strides reduce playback frames only, not acquisition or fit calculations.
    """
    if iteration_stride < 1 or acquisition_stride < 1:
        raise ValueError('Playback strides must be positive integers.')
    folder = 'bead_count_sweep' if bead_sweep else 'rms_snr_sweep'
    result_name = 'pd_rms_bead_count_sweep_snr15_results.npz' if bead_sweep else RESULT_NAME
    iteration_name = 'pd_rms_bead_count_sweep_snr15_iteration_histories.npz' if bead_sweep else ITERATION_NAME
    root = Path(results_dir) if results_dir else Path(__file__).resolve().parents[1]/folder/'results'
    with np.load(root/result_name, allow_pickle=False) as archive:
        d = {key: archive[key].copy() for key in archive.files}
    with np.load(root/iteration_name, allow_pickle=False) as archive:
        iterations = {key: archive[key].copy() for key in archive.files}
    if Path(str(iterations['source_checkpoint' if bead_sweep else 'source_archive'])).name != result_name:
        raise ValueError('Optimizer histories do not belong to this sweep archive.')
    if bead_sweep:
        for key in ('modes', 'truth_strengths', 'slm_strengths', 'residual_rms'):
            np.testing.assert_allclose(iterations[key], d[key], rtol=1e-12, equal_nan=True)
    ri, si, ti = rms_index, snr_index, trial
    shape = d['residual_rms'].shape
    if not all(isinstance(v, (int, np.integer)) and 0 <= v < bound
               for v, bound in zip((ri, si, ti), shape[:3])):
        raise ValueError(f'Case indices must be within {shape[:3]}.')
    if not d['completed'][ri, si, ti].all():
        raise ValueError('Select a case with all three loops completed.')
    microscope = psf.Microscope(N_order=3, lambd=1.3e-3, n=1.333, num_apt=1.05,
                                f=7.2, mag=4, w_0=3.5, L_bfp=15.12, grid_bfp=32)
    grid = psf.Centered_Square_Grid(L_ffp=.0065, grid_ffp=32, z_level=0)
    if bead_sweep:
        count = int(d['bead_counts'][si])
        centers = d['bead_positions_mm'][si, ti, :count]
        if centers.shape != (count, 3) or not np.all(np.isfinite(centers)):
            raise ValueError('Saved bead centers are incomplete.')
        sample = psf.bead_img_3D(grid, d['sample_z'], centers[:, 0], centers[:, 1],
                               centers[:, 2], np.full(count, float(d['bead_radius_mm'])))
    else:
        sample = psf.bead_img_3D(grid, d['sample_z'], [0], [0], [0], [.001])
    modes = d['modes'].tolist()
    truth = zernike.Aberration(modes, d['truth_strengths'][ri, ti])
    sim = RecordingSLM(microscope, sample, truth, modes, float(d['exposures'][ri, si, ti]),
                       np.random.default_rng(np.random.SeedSequence(
                           [int(d['noise_seed']), ri, si, ti])), optical_mode='vector')
    history = [dict(images_taken=0, slm_strengths=np.zeros(len(modes)),
                    residual_rms=zernike.zernike_RMS(truth, microscope.alpha))]
    for loop in range(shape[3]):
        probes = [zernike.Aberration(d['diversity_modes'], basis*float(bias))
                  for basis in d['diversity_basis_strengths'][ti, loop]
                  for bias in d['loop_diversity_strengths'][loop]]
        positions = np.repeat(d['focal_positions'], len(probes))
        sim.images_taken += len(positions)
        if sim.images_taken != int(d['images_taken'][ri, si, ti, loop]):
            raise ValueError('Regenerated image counts disagree with the archive.')
        selected = np.all(iterations['run_indices'] == [ri, si, ti], axis=1)
        selected &= iterations['loop_indices'] == loop
        indices = np.flatnonzero(selected)
        if not len(indices):
            raise ValueError(f'No saved optimizer iterations for loop {loop+1}.')
        indices = indices[np.argsort(iterations['iteration'][indices], kind='stable')]
        final_estimate = iterations['strengths'][indices[-1]]
        expected = history[-1]['slm_strengths'] - float(d['correction_gain'])*np.clip(
            final_estimate, -float(d['max_modal_update']), float(d['max_modal_update']))
        np.testing.assert_allclose(expected, d['slm_strengths'][ri, si, ti, loop], rtol=1e-9, atol=1e-12)
        chosen = sorted(set(range(0, len(indices), iteration_stride)) | {len(indices)-1})
        entries = [dict(iteration=int(iterations['iteration'][indices[j]]),
                        strengths=iterations['strengths'][indices[j]],
                        loss=float(iterations['loss'][indices[j]])) for j in chosen]
        sim.set_slm(d['slm_strengths'][ri, si, ti, loop])
        np.testing.assert_allclose(sim.residual_rms(), d['residual_rms'][ri, si, ti, loop], rtol=1e-9)
        history.append(dict(images_taken=sim.images_taken, slm_strengths=sim.slm_strengths.copy(),
                            residual_rms=float(d['residual_rms'][ri, si, ti, loop]),
                            estimated_strengths=final_estimate, optimizer_info=dict(history=entries)))
    # Keep selected measurements in the viewer; all noise draws above were retained.
    sim.acquisitions = [a for j, a in enumerate(sim.acquisitions)
                        if j % acquisition_stride == 0 or (j+1) % 36 == 0]
    frames, points = replay_history(history, sim)
    for frame in frames:
        frame['acquisition_title'] = 'Noisy acquisition (seed replay)'
    if bead_sweep:
        title = f"Saved bead sweep: {count} beads, RMS {d['initial_rms'][ri]:g}, SNR {float(d['target_snr']):g}, trial {ti+1}"
    else:
        title = f"Saved SNR sweep: RMS {d['initial_rms'][ri]:g}, SNR {d['target_snr'][si]:g}, trial {ti+1}"
    return frames, points, title


def load_snr_case(results_dir=None, *, rms_index=1, snr_index=3, trial=0,
                  iteration_stride=1, acquisition_stride=12):
    """Load a saved single-bead SNR case without rerunning optimization."""
    return _load_case(results_dir, rms_index=rms_index, snr_index=snr_index, trial=trial,
                      iteration_stride=iteration_stride, acquisition_stride=acquisition_stride)


def load_bead_case(results_dir=None, *, rms_index=1, bead_count=3, trial=0, iteration_stride=1):
    """Load 1–5 beads with archived centers/radii and all saved optimizer iterates."""
    root = Path(results_dir) if results_dir else Path(__file__).resolve().parents[1]/'bead_count_sweep'/'results'
    with np.load(root/'pd_rms_bead_count_sweep_snr15_results.npz', allow_pickle=False) as d:
        matches = np.flatnonzero(d['bead_counts'] == bead_count)
    if len(matches) != 1:
        raise ValueError('bead_count must match one saved bead count (1–5).')
    return _load_case(root, rms_index=rms_index, snr_index=int(matches[0]), trial=trial,
                      iteration_stride=iteration_stride, bead_sweep=True)
