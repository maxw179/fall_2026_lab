"""Record a three-loop cumulative AO reconstruction for offline plotting.

Run from the repository root with python -m systematic_testing.gpu_demo.run_demo.
The core timer covers workspace construction and optimization, with CUDA
synchronization at its boundaries. Acquisition, exports and object replay are
separate. CUDA is required unless --backend cpu is explicitly supplied.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from optimization import reconstruction
from systematic_testing import correction
from utils import psf, zernike


MODES = np.array([
    [-2, 2], [2, 2], [-3, 3], [-1, 3], [1, 3], [3, 3],
    [-4, 4], [-2, 4], [0, 4], [2, 4], [4, 4],
    [-5, 5], [-3, 5], [-1, 5], [1, 5], [3, 5], [5, 5],
])
DIVERSITY_MODES_BY_LOOP = np.array([
    [[-2, 2], [0, 4], [2, 2]],
    [[-3, 3], [0, 4], [3, 3]],
    [[-4, 4], [0, 4], [4, 4]],
])
DIVERSITY_RMS = np.array([-.10, 0., .10])
FOCAL_Z = np.linspace(-.002, .002, 4)
MICROSCOPE = dict(N_order=3, lambd=.0013, n=1.333, num_apt=1.05,
                  f=7.2, mag=4, w_0=3.5, L_bfp=15.12)


@dataclass(frozen=True)
class DemoConfig:
    size: int = 128
    pupil_size: int = 32
    initial_rms: float = .20
    beads: int = 5
    snr: float = 15.
    diversity_settings: tuple = (-.10, 0., .10)
    shared_zero: bool = False
    read_sigma_e: float = 0.
    checkerboard_e: float = 0.
    background_e: float = 0.
    rho: float = 1e46
    trial: int = 0
    sample_planes: int = 41
    reconstruction_planes: int = 21
    random_starts: int = 100
    max_iterations: int = 50
    screening_batch_size: int = 8
    psf_batch_size: int = 8
    derivative_batch_size: int = 8
    frequency_batch_size: int = 4096
    object_stride: int = 1
    save_objects: bool = False
    backend: str = 'cuda'


def write_json(path, payload):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Cannot serialize {type(value)}')
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, default=convert) + '\n')
    temporary.replace(path)


def save_npz(path, **arrays):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def coefficients(aberration):
    """Align known phases to the sweep basis, including historical SLM offsets."""
    lookup = {tuple(mode): strength for mode, strength in
              zip(aberration.modes, aberration.strengths)}
    if set(lookup) - {tuple(mode) for mode in MODES}:
        raise ValueError('Demo diversity is outside the sweep basis.')
    return np.array([lookup.get(tuple(mode), 0.) for mode in MODES])


def make_scene(config):
    microscope = psf.Microscope(**MICROSCOPE, grid_bfp=config.pupil_size)
    grid = psf.Centered_Square_Grid(L_ffp=.0065, grid_ffp=config.size, z_level=0)
    sample_z = np.linspace(-.002, .002, config.sample_planes)
    reconstruction_z = np.linspace(-.002, .002, config.reconstruction_planes)
    x, y = grid.get_xy()
    radius = .001  # The sweep's bead radius, in mm; overlap is allowed.
    bounds = np.array([[x.min()+radius, x.max()-radius],
                       [y.min()+radius, y.max()-radius],
                       [sample_z.min()+radius, sample_z.max()-radius]])
    centers = np.random.default_rng(np.random.SeedSequence(
        [2, config.beads, config.trial])).uniform(
            bounds[:, 0], bounds[:, 1], size=(config.beads, 3))
    sample = psf.bead_img_3D(grid, sample_z, *centers.T,
                            np.full(config.beads, radius))
    direction = np.random.default_rng(np.random.SeedSequence(
        [0, config.trial])).normal(size=len(MODES))
    truth = zernike.RMSAberration(MODES, direction, config.initial_rms, microscope.alpha)
    schedule = []
    for loop_modes in DIVERSITY_MODES_BY_LOOP:
        probes = [zernike.Aberration(MODES, np.zeros(len(MODES)))] if config.shared_zero else []
        for mode in loop_modes:
            unit = zernike.RMSAberration([mode], [1.], 1., microscope.alpha)
            probes.extend(zernike.Aberration([mode], unit.strengths*bias)
                          for bias in config.diversity_settings)
        schedule.append(probes)
    return microscope, grid, sample, reconstruction_z, centers, truth, schedule


def workspace_options(config, *, replay=False):
    options = dict(backend=config.backend, psf_batch_size=config.psf_batch_size,
                   derivative_batch_size=config.derivative_batch_size,
                   frequency_batch_size=config.frequency_batch_size,
                   reuse_svd=not replay, retain_fields=not replay)
    if config.backend == 'cuda':
        options.update(svd_method='qr', screening_method='linear',
                       screening_batch_size=config.screening_batch_size)
    return options


class RecordedSLM(correction.SimulatedSLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acquisition_seconds = []
        self.read_sigma_e = 0.
        self.checkerboard_e = 0.
        self.background_e = 0.

    def acquire(self, focal_z_levels, diversities=None):
        started = perf_counter()
        if not (self.read_sigma_e or self.checkerboard_e or self.background_e):
            result = super().acquire(focal_z_levels, diversities)
        else:
            from systematic_testing.noise_robustness.camera_noise import simulate_camera
            clean = self.microscope.compute_image_3D(
                self.sample, self._residual(), focal_z_levels, mode=self.optical_mode, diversities=diversities)[3]
            observation = simulate_camera(self.exposure*np.maximum(clean, 0),
                read_sigma_e=self.read_sigma_e, background_e=self.background_e, pattern='checkerboard',
                offset_pattern_e=self.checkerboard_e,
                poisson_seed=int(self.rng.integers(2**32)),
                read_seed=int(self.rng.integers(2**32)))
            result = observation['pedestal_only_e']/self.exposure
            self.images_taken += len(result)
            self.photons_detected += int(observation['signal_counts'].sum())
        self.acquisition_seconds.append(perf_counter()-started)
        return result


def export_history(output, records, truth, alpha):
    rows = [(loop, record, entry) for loop, record in enumerate(records, 1)
            for entry in record['optimizer_info']['history']]
    strengths = np.array([entry['strengths'] for _, _, entry in rows])
    slm = np.array([record['acquisition_slm_strengths'] for _, record, _ in rows])
    errors = np.array([zernike.zernike_RMS_difference(
        zernike.Aberration(MODES, s),
        truth + zernike.Aberration(MODES, command), alpha)
        for s, command in zip(strengths, slm)])
    save_npz(output/'history.npz', modes=MODES,
             loop=np.array([loop for loop, _, _ in rows]),
             iteration=np.array([entry['iteration'] for _, _, entry in rows]),
             loss=np.array([entry['loss'] for _, _, entry in rows]),
             strengths=strengths, acquisition_slm_strengths=slm,
             static_estimate_strengths=strengths-slm, rms_error=errors,
             core_elapsed_seconds=np.array([entry['core_elapsed_seconds']
                                            for _, _, entry in rows]),
             applied_residual_rms=np.r_[zernike.zernike_RMS(truth, alpha),
                                       [r['residual_rms'] for r in records]],
             applied_slm_strengths=np.vstack([np.zeros(len(MODES)),
                                              [r['slm_strengths'] for r in records]]),
             loop_end_core_seconds=np.r_[0., np.cumsum([
                 r['optimizer_info']['demo_timing']['core_seconds'] for r in records])])


def replay_objects(output, config, microscope, grid, records, synchronize):
    """Reconstruct saved iterates after all AO loops, outside the core timer."""
    chosen, offsets = [], 0
    for record in records:
        history = record['optimizer_info']['history']
        ids = sorted(set(range(0, len(history), config.object_stride)) | {len(history)-1})
        chosen.extend(offsets+i for i in ids)
        offsets += len(history)
    temporary = output/'object_history.tmp.npy'
    objects = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
        shape=(len(chosen), config.reconstruction_planes, config.size, config.size))
    started, row, offset = perf_counter(), 0, 0
    for loop, record in enumerate(records, 1):
        with np.load(output/f'loop_{loop}_input.npz', allow_pickle=False) as saved:
            diversities = [zernike.Aberration(MODES, s) for s in saved['diversity_strengths']]
            workspace = reconstruction.ReconstructionWorkspace(
                microscope, grid, saved['images'], saved['focal_z'], saved['sample_z'],
                MODES, config.rho, diversities=diversities, mode='vector',
                **workspace_options(config, replay=True))
        history = record['optimizer_info']['history']
        for i, entry in enumerate(history):
            if offset+i in chosen:
                print(f'Replaying object: loop {loop}, iteration {entry["iteration"]}', flush=True)
                objects[row] = workspace.estimate_sample(entry['strengths']).image_mask
                row += 1
        offset += len(history)
        del workspace
    synchronize()
    objects.flush()
    del objects
    temporary.replace(output/'object_history.npy')
    save_npz(output/'object_frames.npz', history_indices=np.array(chosen))
    return perf_counter()-started


def export_correction_preview(output, simulator, config, rms_index):
    """Save the sweep's four-state image layout without changing acquisitions.

    First three columns are actual zero-diversity measurements; the last
    column is a separate simulated diagnostic after the third correction.
    """
    started = perf_counter()
    states = []
    for loop in range(1, 4):
        with np.load(output/f'loop_{loop}_input.npz', allow_pickle=False) as saved:
            fresh = len(FOCAL_Z)*(int(config.shared_zero)+3*len(config.diversity_settings))
            zero = 0 if config.shared_zero else list(config.diversity_settings).index(0.)
            states.append(saved['images'][-fresh:][np.arange(4)*(fresh//4)+zero])
    print('Preparing four-state correction preview (outside core timing)', flush=True)
    clean = simulator.microscope.compute_image_3D(
        simulator.sample, simulator._residual(), FOCAL_Z, mode='vector')[3]
    rng = np.random.default_rng(np.random.SeedSequence(
        [1, rms_index, config.beads, config.trial, 9902]))
    from systematic_testing.noise_robustness.camera_noise import simulate_camera
    diagnostic = simulate_camera(simulator.exposure*np.maximum(clean, 0),
        read_sigma_e=config.read_sigma_e, background_e=config.background_e, pattern='checkerboard',
        offset_pattern_e=config.checkerboard_e,
        poisson_seed=int(rng.integers(2**32)), read_seed=int(rng.integers(2**32)))
    states.append(diagnostic['pedestal_only_e']/simulator.exposure)
    save_npz(output/'correction_preview.npz', images=np.stack(states, axis=1),
             focal_z=FOCAL_Z, final_state_is_simulated=True)
    return perf_counter()-started


def run_demo(config, output, *, prepare_only=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    wall_start = perf_counter()
    metadata = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
                    config=asdict(config), microscope=dict(MICROSCOPE, grid_bfp=config.pupil_size),
                    focal_positions_mm=FOCAL_Z, diversity_modes_by_loop=DIVERSITY_MODES_BY_LOOP,
                    diversity_rms_waves=config.diversity_settings, num_ao_loops=3,
                    correction_gain=1., max_modal_update=.10, reuse_acquisitions=True,
                    python=platform.python_version(), numpy=np.__version__,
                    hostname=platform.node(), slurm_job_id=os.environ.get('SLURM_JOB_ID'), loops=[])
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    metadata['git_revision'] = revision.stdout.strip() if revision.returncode == 0 else None
    write_json(output/'run.json', metadata)
    try:
        synchronize = lambda: None
        if config.backend == 'cuda' and not prepare_only:
            from optimization.reconstruction_cuda import _cupy
            cp = _cupy()
            synchronize = cp.cuda.get_current_stream().synchronize
            device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            name = device['name']
            metadata['hardware'] = dict(gpu=name.decode() if isinstance(name, bytes) else str(name),
                                        vram_bytes=int(device['totalGlobalMem']), cupy=cp.__version__,
                                        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                                        cuda_driver=cp.cuda.runtime.driverGetVersion())
        metadata['runtime_startup_seconds'] = perf_counter()-wall_start
        preparation_start = perf_counter()
        microscope, grid, sample, sample_z, centers, truth, schedule = make_scene(config)
        calibration = microscope.compute_image_3D(sample, truth, [0.], mode='vector')[3]
        exposure = correction.exposure_for_snr(calibration, config.snr)
        # Match the RMS=.20, five-bead sweep case's noise seed (zero-based trial).
        rms_index = int(np.argmin(abs(np.array([.05, .10, .15, .20, .25, .30])-config.initial_rms)))
        simulator = RecordedSLM(microscope, sample, truth, MODES, exposure,
            np.random.default_rng(np.random.SeedSequence([1, rms_index, config.beads, config.trial])),
            optical_mode='vector')
        simulator.read_sigma_e = config.read_sigma_e
        simulator.checkerboard_e = config.checkerboard_e
        simulator.background_e = config.background_e
        frames_per_loop = len(FOCAL_Z)*len(schedule[0])
        metadata.update(frames_per_loop=frames_per_loop, noise_model='signal/background poisson + gaussian read + fixed checkerboard',
                        exposure=exposure, alpha=microscope.alpha,
                        actual_initial_rms=zernike.zernike_RMS(truth, microscope.alpha),
                        preparation_seconds=perf_counter()-preparation_start,
                        timing_note='Synchronized host wall time; no full-workload warmup. '
                                    'Core includes workspace setup, screening and Newton optimization. '
                                    'Forward simulation, RMS diagnostics, exports and object replay excluded.')
        save_npz(output/'scene.npz', modes=MODES, truth_strengths=truth.strengths,
                 bead_centers_mm=centers, bead_radius_mm=.001,
                 sample=sample.image_mask, sample_z=sample.z_levels,
                 reconstruction_z=sample_z, x=grid.get_xy()[0], y=grid.get_xy()[1],
                 calibration_image=calibration, alpha=microscope.alpha)
        if prepare_only:
            focal_z = np.repeat(FOCAL_Z, len(schedule[0]))
            probes = schedule[0]*len(FOCAL_Z)
            images = simulator.acquire(focal_z, probes)
            save_npz(output/'loop_1_input.npz', images=images, focal_z=focal_z,
                     sample_z=sample_z,
                     diversity_strengths=np.array([coefficients(a) for a in probes]),
                     acquisition_slm_strengths=simulator.slm_strengths.copy())
            metadata.update(status='prepared', preview_only=True,
                            acquisition_seconds=simulator.acquisition_seconds,
                            wall_seconds=perf_counter()-wall_start)
            write_json(output/'run.json', metadata)
            return metadata
        records, timings = [], []

        def timed_optimizer(**kwargs):
            loop = len(timings)+1
            inputs = {key: kwargs.pop(key) for key in (
                'microscope', 'grid', 'images', 'focal_z_levels', 'sample_z_levels',
                'modes', 'rho', 'diversities', 'mode')}
            # Disk writes occur before timing. Preserve the exact cumulative
            # noise draws and current-residual diversity offsets for replay.
            save_npz(output/f'loop_{loop}_input.npz', images=inputs['images'],
                     focal_z=inputs['focal_z_levels'], sample_z=inputs['sample_z_levels'],
                     diversity_strengths=np.array([coefficients(a) for a in inputs['diversities']]),
                     acquisition_slm_strengths=simulator.slm_strengths.copy())
            synchronize()
            core_start = perf_counter()
            workspace = reconstruction.ReconstructionWorkspace(**inputs, **workspace_options(config))
            synchronize()
            setup_seconds = perf_counter()-core_start
            cumulative = sum(t['core_seconds'] for t in timings)
            stamps = {}

            def record_iteration(event):
                synchronize()
                stamps[event['iteration']] = cumulative + perf_counter()-core_start

            started = perf_counter()
            estimated, info = workspace.optimize(**kwargs, on_iteration=record_iteration)
            synchronize()
            stop = perf_counter()
            screening = info['initialization']
            timing = dict(loop=loop, fit_images=len(inputs['images']),
                          workspace_seconds=setup_seconds, optimization_seconds=stop-started,
                          screening_seconds=screening['seconds'] if screening else 0.,
                          core_seconds=stop-core_start, accepted_steps=info['iterations'],
                          success=bool(info['success']), message=info['message'],
                          memory_estimates=workspace.memory_summary())
            if config.backend == 'cuda':
                timing['svd_frequencies'] = workspace.svd_counts.copy()
            if screening:
                timing['screening'] = {key: screening[key] for key in (
                    'method', 'batch_size', 'refined_candidates', 'selected_index')}
                for key in ('fallback', 'linear_frequencies', 'qr_frequencies', 'svd_fallback_frequencies'):
                    if key in screening:
                        timing['screening'][key] = screening[key]
                save_npz(output/f'loop_{loop}_screening.npz',
                         strengths=screening['candidate_strengths'], losses=screening['candidate_losses'],
                         selected_index=screening['selected_index'])
            for entry in info['history']:
                entry['core_elapsed_seconds'] = stamps[entry['iteration']]
            info['demo_timing'] = timing
            timings.append(timing)
            print(f'Loop {loop}: core {timing["core_seconds"]:.3f}s, '
                  f'screening {timing["screening_seconds"]:.3f}s, '
                  f'{info["iterations"]} accepted steps; {info["message"]}', flush=True)
            del workspace
            return estimated, info

        def checkpoint(record):
            records.append(record)
            export_history(output, records, truth, microscope.alpha)
            metadata.update(loops=timings, acquisition_seconds=simulator.acquisition_seconds,
                            core_seconds=sum(t['core_seconds'] for t in timings),
                            optimization_seconds=sum(t['optimization_seconds'] for t in timings),
                            screening_seconds=sum(t['screening_seconds'] for t in timings))
            write_json(output/'run.json', metadata)
            print(f'AO loop {len(records)} applied: residual RMS {record["residual_rms"]:.6f} waves', flush=True)

        def progress(event):
            print(f'AO loop {event["update"]}: {event["stage"]}' +
                  (f' {event["fit_images"]} cumulative images' if 'fit_images' in event else ''), flush=True)

        history = correction.stack_feedback(
            simulator, grid, sample_z, FOCAL_Z, [-2, 2], config.diversity_settings,
            config.rho, 3*frames_per_loop, gain=1., max_update=.10, diversity_schedule=schedule,
            diversity_strategy='joint', reuse_acquisitions=True, optimizer=timed_optimizer,
            optimizer_kwargs=dict(method='newton', max_iterations=config.max_iterations,
                num_random_starts=config.random_starts, random_start_rms_range=(.005, .50),
                random_start_seed=0, verbose=False, show_initialization_progress=False),
            on_progress=progress, on_update=checkpoint)
        if len(history) != 4 or [t['fit_images'] for t in timings] != [frames_per_loop*i for i in (1, 2, 3)]:
            raise RuntimeError('The cumulative three-loop acquisition did not complete.')
        metadata['object_replay_and_export_seconds'] = 0.
        if config.save_objects:
            metadata['object_replay_and_export_seconds'] = replay_objects(
                output, config, microscope, grid, records, synchronize)
        metadata['correction_preview_seconds'] = export_correction_preview(
            output, simulator, config, rms_index)
        metadata.update(status='complete', wall_seconds=perf_counter()-wall_start,
                        final_residual_rms=history[-1]['residual_rms'],
                        all_optimizers_converged=all(t['success'] for t in timings))
        write_json(output/'run.json', metadata)
        print(f'Complete: core {metadata["core_seconds"]:.3f}s; '
              f'final applied RMS {metadata["final_residual_rms"]:.6f} waves. Results: {output}', flush=True)
        return metadata
    except Exception as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                        wall_seconds=perf_counter()-wall_start)
        write_json(output/'run.json', metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New result directory; never overwritten.')
    for key in ('size', 'pupil_size', 'beads', 'sample_planes', 'reconstruction_planes',
                'random_starts', 'max_iterations', 'screening_batch_size', 'object_stride',
                'psf_batch_size', 'derivative_batch_size', 'frequency_batch_size', 'trial'):
        parser.add_argument('--'+key.replace('_', '-'), type=int, default=getattr(DemoConfig(), key))
    for key in ('initial_rms', 'snr', 'rho'):
        parser.add_argument('--'+key.replace('_', '-'), type=float, default=getattr(DemoConfig(), key))
    parser.add_argument('--diversity-settings', type=float, nargs='+', default=DemoConfig().diversity_settings)
    parser.add_argument('--shared-zero', action='store_true')
    parser.add_argument('--read-sigma-e', type=float, default=0.)
    parser.add_argument('--background-e', type=float, default=0.)
    parser.add_argument('--checkerboard-e', type=float, default=0.)
    parser.add_argument('--backend', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--prepare-only', action='store_true', help='Generate exact first-loop noisy inputs without fitting or CUDA startup.')
    parser.add_argument('--save-objects', action='store_true', help='Also export 3D object estimates; image/wavefront replay does not need them.')
    args = vars(parser.parse_args())
    output = args.pop('output')
    prepare_only = args.pop('prepare_only')
    if any(args[key] < (2 if key in ('size', 'pupil_size', 'sample_planes', 'reconstruction_planes') else 1)
           for key in args if type(args[key]) is int and key not in ('trial', 'random_starts')):
        parser.error('sizes must be positive; spatial and axial grids need at least two samples')
    if args['trial'] < 0 or args['random_starts'] < 0:
        parser.error('trial and random-starts must be nonnegative')
    if any(not np.isfinite(args[key]) or args[key] <= 0 for key in ('initial_rms', 'snr', 'rho')):
        parser.error('RMS, SNR and rho must be finite and positive')
    if any(not np.isfinite(args[k]) or args[k] < 0 for k in ('read_sigma_e', 'checkerboard_e', 'background_e')):
        parser.error('Noise amplitudes must be finite and nonnegative')
    if not np.all(np.isfinite(args['diversity_settings'])):
        parser.error('Diversity settings must be finite')
    run_demo(DemoConfig(**args), output, prepare_only=prepare_only)


if __name__ == '__main__':
    main()
