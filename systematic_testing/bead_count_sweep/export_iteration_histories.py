"""Export retained bead-sweep histories without replaying optimization.

After the sweep completes, run in ITS existing kernel:
    %run -i systematic_testing/bead_count_sweep/export_iteration_histories.py
Paths are resolved relative to this script, independent of notebook cwd.
"""
from pathlib import Path
import numpy as np
from utils import zernike


def export_iteration_histories(run_histories, checkpoint_path, alpha):
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    if not saved['completed'].all():
        raise RuntimeError('Wait until every AO loop has completed before exporting.')
    rows, loop_rows = [], []
    for ri, ci, trial in np.ndindex(saved['completed'].shape[:-1]):
        history = run_histories[(ri, ci, trial)]
        if len(history) != len(saved['ao_loops']) + 1:
            raise ValueError('Incomplete in-memory AO history.')
        for loop, record in enumerate(history[1:]):
            info = record['optimizer_info']
            if not np.allclose(record['slm_strengths'], saved['slm_strengths'][ri, ci, trial, loop]):
                raise ValueError('In-memory correction does not match bead-sweep checkpoint.')
            steps = info['history']
            expected = int(saved['optimizer_steps'][ri, ci, trial, loop])
            if not steps or steps[-1]['iteration'] != expected:
                raise ValueError('Iteration history does not match checkpoint.')
            prior = np.zeros(len(saved['modes'])) if loop == 0 else saved['slm_strengths'][ri, ci, trial, loop-1]
            truth = saved['truth_strengths'][ri, trial] + prior
            for step in steps:
                strengths = np.asarray(step['strengths'], dtype=float)
                error = zernike.zernike_RMS_difference(
                    zernike.Aberration(saved['modes'], strengths),
                    zernike.Aberration(saved['modes'], truth), alpha)
                rows.append(((ri, ci, trial), loop, step['iteration'],
                             step['loss'], strengths.copy(), error))
            loop_rows.append(((ri, ci, trial), loop, info['success'],
                              info['message'], record['estimated_strengths'],
                              record['update'], record['slm_strengths']))
    arrays = dict(
        run_indices=np.array([r[0] for r in rows], dtype=int),
        loop_indices=np.array([r[1] for r in rows], dtype=int),
        iteration=np.array([r[2] for r in rows], dtype=int),
        loss=np.array([r[3] for r in rows]),
        strengths=np.array([r[4] for r in rows]),
        estimation_rms=np.array([r[5] for r in rows]),
        loop_run_indices=np.array([r[0] for r in loop_rows], dtype=int),
        loop_record_indices=np.array([r[1] for r in loop_rows], dtype=int),
        optimizer_success=np.array([r[2] for r in loop_rows]),
        optimizer_message=np.array([r[3] for r in loop_rows]),
        estimated_strengths=np.array([r[4] for r in loop_rows]),
        applied_updates=np.array([r[5] for r in loop_rows]),
        applied_slm_strengths=np.array([r[6] for r in loop_rows]),
        initial_rms=saved['initial_rms'], bead_counts=saved['bead_counts'],
        modes=saved['modes'], ao_loops=saved['ao_loops'],
        residual_rms=saved['residual_rms'], optimizer_steps=saved['optimizer_steps'],
        truth_strengths=saved['truth_strengths'], slm_strengths=saved['slm_strengths'],
        source_checkpoint=str(Path(checkpoint_path).resolve()),
    )
    if not all(np.isfinite(arrays[key]).all() for key in
               ('loss', 'strengths', 'estimation_rms', 'applied_updates')):
        raise ValueError('Nonfinite history values; archive was not written.')
    destination = Path(__file__).parent / 'results'
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'pd_rms_bead_count_sweep_snr15_iteration_histories.npz'
    # Exclusive creation protects any previously exported archive.
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    print(f'Saved {len(rows)} optimizer states across {len(loop_rows)} AO loops: {path}')
    return path


if __name__ == '__main__':
    if 'run_histories' not in globals() or 'microscope' not in globals():
        raise RuntimeError('Use %run -i in the completed sweep kernel; do not restart it.')
    checkpoint = Path(__file__).resolve().parent / 'results' / 'pd_rms_bead_count_sweep_snr15_results.npz'
    iteration_history_path = export_iteration_histories(run_histories, checkpoint, microscope.alpha)
