"""Paired single-bead RMS sweep; one independent case per Slurm array task."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from optimization import reconstruction
from systematic_testing import correction
from systematic_testing.gpu_demo.run_demo import (
    DIVERSITY_MODES_BY_LOOP, FOCAL_Z, MICROSCOPE, MODES,
    coefficients, save_npz, write_json)
from systematic_testing.noise_robustness.noise_calibration import calibrated_noise
from utils import psf, zernike


DEFAULT_RMS = (.05, .10, .15, .20, .225, .25, .26, .27, .28, .29,
               .30, .31, .32, .33, .34, .35, .375, .40, .45, .50)


@dataclass(frozen=True)
class SweepConfig:
    rms_values: tuple = DEFAULT_RMS
    trials: int = 20
    size: int = 64
    pupil_size: int = 32
    sample_planes: int = 41
    reconstruction_planes: int = 21
    snr: float = 10.
    gaussian_to_poisson_rms: float = 1.
    aberration_seed: int = 0
    noise_seed: int = 1
    rho: float = 1e46
    random_starts: int = 100
    max_iterations: int = 50
    backend: str = 'cuda'

    def validate(self):
        rms = np.asarray(self.rms_values)
        if (rms.ndim != 1 or not len(rms) or not np.all(np.isfinite(rms))
                or np.any(rms <= 0) or np.any(np.diff(rms) <= 0)):
            raise ValueError('RMS values must be positive, finite and strictly increasing.')
        for name in ('trials', 'size', 'pupil_size', 'sample_planes',
                     'reconstruction_planes', 'max_iterations'):
            minimum = 2 if name in ('size', 'pupil_size', 'sample_planes', 'reconstruction_planes') else 1
            if getattr(self, name) < minimum:
                raise ValueError(f'{name} must be at least {minimum}.')
        if min(self.aberration_seed, self.noise_seed, self.random_starts) < 0:
            raise ValueError('Seeds and random-start count must be nonnegative.')
        if any(not np.isfinite(v) or v <= 0 for v in (
                self.snr, self.rho, self.gaussian_to_poisson_rms)):
            raise ValueError('SNR, rho and Gaussian/Poisson RMS ratio must be finite and positive.')
        if self.backend not in ('cpu', 'cuda'):
            raise ValueError('Select cpu or cuda explicitly.')


def make_scene(config, rms_index, trial):
    """Hold the bead fixed; rescale the same seeded modal direction at every RMS."""
    microscope = psf.Microscope(**MICROSCOPE, grid_bfp=config.pupil_size)
    grid = psf.Centered_Square_Grid(L_ffp=.0065, grid_ffp=config.size, z_level=0.)
    sample_z = np.linspace(-.002, .002, config.sample_planes)
    reconstruction_z = np.linspace(-.002, .002, config.reconstruction_planes)
    sample = psf.bead_img_3D(grid, sample_z, [0.], [0.], [0.], [.001])
    direction = np.random.default_rng(np.random.SeedSequence(
        [config.aberration_seed, trial])).normal(size=len(MODES))
    truth = zernike.RMSAberration(MODES, direction, config.rms_values[rms_index], microscope.alpha)
    schedule = []
    for loop_modes in DIVERSITY_MODES_BY_LOOP:
        probes = []
        for mode in loop_modes:
            unit = zernike.RMSAberration([mode], [1.], 1., microscope.alpha)
            probes.extend(zernike.Aberration([mode], unit.strengths*bias)
                          for bias in (-.10, 0., .10))
        schedule.append(probes)
    return microscope, grid, sample, reconstruction_z, direction, truth, schedule


def expected_snr(clean, exposure, gaussian_sigma):
    clean = np.maximum(np.asarray(clean, dtype=float), 0.)
    return float(np.sqrt(np.sum(clean**2) /
                         (np.sum(clean)/exposure + clean.size*gaussian_sigma**2)))


class MixedNoiseSLM(correction.SimulatedSLM):
    """Fixed detector/exposure through all loops, with separate noise streams."""
    def __init__(self, *args, gaussian_sigma, read_rng, output, **kwargs):
        super().__init__(*args, **kwargs)
        self.gaussian_sigma = gaussian_sigma
        self.read_rng = read_rng
        self.output = Path(output)
        self.acquisitions = []

    def acquire(self, focal_z_levels, diversities=None):
        started = perf_counter()
        clean = self.microscope.compute_image_3D(
            self.sample, self._residual(), focal_z_levels,
            mode=self.optical_mode, diversities=diversities)[3]
        if not np.all(np.isfinite(clean)) or clean.min() < -1e-12*max(float(clean.max()), 0.):
            raise ValueError('Forward images must be finite and nonnegative.')
        clean = np.maximum(clean, 0.)
        counts = self.rng.poisson(self.exposure*clean)
        images = counts/self.exposure + self.read_rng.normal(0., self.gaussian_sigma, clean.shape)
        self.images_taken += len(images)
        self.photons_detected += int(counts.sum())
        self.acquisitions.append(dict(
            expected_snr=expected_snr(clean, self.exposure, self.gaussian_sigma),
            realized_snr=float(np.sqrt(np.sum(clean**2)/np.sum((images-clean)**2))),
            seconds=perf_counter()-started))
        # Store fresh frames once; historical frames are reused without new noise.
        save_npz(self.output/f'acquisition_{len(self.acquisitions)}.npz',
                 images=images, focal_z=focal_z_levels,
                 diversity_strengths=np.array([coefficients(a) for a in diversities]),
                 acquisition_slm_strengths=self.slm_strengths.copy())
        return images


def run_case(config, output, case_index, *, prepare_only=False):
    config.validate()
    if not 0 <= case_index < len(config.rms_values)*config.trials:
        raise ValueError('Case index is outside the configured sweep.')
    ri, trial = divmod(case_index, config.trials)
    output = Path(output)/f'case_{case_index:04d}'
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    metadata = dict(status='running', config=asdict(config), case_index=case_index,
                    rms_index=ri, trial=trial, initial_rms=config.rms_values[ri],
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    hostname=platform.node(), python=platform.python_version(), numpy=np.__version__,
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'), loops=[])
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
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion())
        microscope, grid, sample, sample_z, direction, truth, schedule = make_scene(config, ri, trial)
        calibration = microscope.compute_image_3D(sample, truth, [0.], mode='vector')[3]
        # Reuse the shared total-error calibration with zero-mean Gaussian read noise.
        # level=1 assigns half the expected error power to each noise component.
        calibration_noise = calibrated_noise(calibration, 'gaussian',
            config.gaussian_to_poisson_rms, config.snr,
            np.random.SeedSequence([config.noise_seed, ri, trial, 100]),
            np.random.SeedSequence([config.noise_seed, ri, trial, 101]), gaussian_mean_sigma=0.)
        exposure = calibration_noise['exposure']
        sigma = calibration_noise['artifact_amplitude']
        simulator = MixedNoiseSLM(microscope, sample, truth, MODES, exposure,
            np.random.default_rng(np.random.SeedSequence([config.noise_seed, ri, trial, 0])),
            gaussian_sigma=sigma,
            read_rng=np.random.default_rng(np.random.SeedSequence([config.noise_seed, ri, trial, 1])),
            output=output, optical_mode='vector')
        metadata.update(alpha=microscope.alpha, exposure=exposure, gaussian_sigma=sigma,
            gaussian_read_sigma_e=sigma*exposure,
            expected_calibration_snr=calibration_noise['expected_snr'],
            realized_calibration_snr=calibration_noise['realized_snr'],
            noise_model='signal Poisson + independent zero-mean Gaussian; negative observations retained',
            snr_calibration='initial_aberrated_z0_no_diversity_total_expected_rms_fixed_exposure_and_read_noise',
            microscope=dict(MICROSCOPE, grid_bfp=config.pupil_size),
            bead_center_mm=[0., 0., 0.], bead_radius_mm=.001,
            focal_z_mm=FOCAL_Z, diversity_modes_by_loop=DIVERSITY_MODES_BY_LOOP,
            diversity_rms_waves=[-.10, 0., .10], num_ao_loops=3,
            gain=1., max_modal_update=.10, reuse_acquisitions=True,
            random_start_seed=0, random_start_rms_range=[.005, .50])
        save_npz(output/'scene.npz', modes=MODES, wavefront_direction=direction,
                 truth_strengths=truth.strengths, sample_z=sample.z_levels,
                 reconstruction_z=sample_z, x=grid.get_xy()[0], y=grid.get_xy()[1],
                 calibration_clean=calibration, calibration_noisy=calibration_noise['image'],
                 alpha=microscope.alpha)
        write_json(output/'run.json', metadata)
        if prepare_only:
            simulator.acquire(np.repeat(FOCAL_Z, len(schedule[0])), schedule[0]*len(FOCAL_Z))
            metadata.update(status='prepared', acquisitions=simulator.acquisitions,
                            wall_seconds=perf_counter()-started)
            write_json(output/'run.json', metadata)
            return metadata

        records, timings = [], []

        def optimizer(**kwargs):
            inputs = {key: kwargs.pop(key) for key in (
                'microscope', 'grid', 'images', 'focal_z_levels', 'sample_z_levels',
                'modes', 'rho', 'diversities', 'mode')}
            options = dict(backend=config.backend, psf_batch_size=8, derivative_batch_size=8,
                           frequency_batch_size=4096, reuse_svd=True, retain_fields=True)
            if config.backend == 'cuda':
                options.update(svd_method='qr', screening_method='linear', screening_batch_size=8)
            synchronize()
            begin = perf_counter()
            workspace = reconstruction.ReconstructionWorkspace(**inputs, **options)
            estimated, info = workspace.optimize(**kwargs)
            synchronize()
            seconds = perf_counter()-begin
            screening = info['initialization']
            if screening:
                save_npz(output/f'loop_{len(timings)+1}_screening.npz',
                    strengths=screening['candidate_strengths'], losses=screening['candidate_losses'],
                    selected_index=screening['selected_index'])
            timings.append(dict(core_seconds=seconds, fit_images=len(inputs['images']),
                success=bool(info['success']), message=info['message'],
                iterations=info['iterations'], loss=info['loss']))
            return estimated, info

        def checkpoint(record):
            records.append(record)
            n = len(records)
            residual = np.full(3, np.nan)
            estimation = np.full(3, np.nan)
            residual[:n] = [r['residual_rms'] for r in records]
            estimation[:n] = [zernike.zernike_RMS_difference(
                zernike.Aberration(MODES, r['estimated_strengths']),
                truth+zernike.Aberration(MODES, r['acquisition_slm_strengths']), microscope.alpha)
                for r in records]
            success = np.zeros(3, dtype=bool)
            success[:n] = [t['success'] for t in timings]
            save_npz(output/'results.npz', case_index=case_index, rms_index=ri, trial=trial,
                initial_rms=config.rms_values[ri], residual_rms=residual,
                estimation_rms=estimation, completed=np.isfinite(residual), optimizer_success=success,
                slm_strengths=np.array([r['slm_strengths'] for r in records]),
                estimated_strengths=np.array([r['estimated_strengths'] for r in records]),
                truth_strengths=truth.strengths)
            rows = [(i+1, entry) for i, r in enumerate(records) for entry in r['optimizer_info']['history']]
            save_npz(output/'history.npz', loop=np.array([i for i, _ in rows]),
                iteration=np.array([e['iteration'] for _, e in rows]),
                strengths=np.array([e['strengths'] for _, e in rows]),
                loss=np.array([e['loss'] for _, e in rows]))
            metadata.update(loops=timings, acquisitions=simulator.acquisitions,
                            wall_seconds=perf_counter()-started)
            write_json(output/'run.json', metadata)
            print(f'Case {case_index}: RMS {config.rms_values[ri]:g}, trial {trial+1}, '
                  f'loop {n}: applied residual {residual[n-1]:.6f}, '
                  f'estimation error {estimation[n-1]:.6f} waves; {timings[-1]["message"]}', flush=True)

        history = correction.stack_feedback(simulator, grid, sample_z, FOCAL_Z,
            [-2, 2], [-.10, 0., .10], config.rho, 108, gain=1., max_update=.10,
            diversity_schedule=schedule, diversity_strategy='joint', reuse_acquisitions=True,
            optimizer=optimizer, optimizer_kwargs=dict(method='newton',
                max_iterations=config.max_iterations, num_random_starts=config.random_starts,
                random_start_rms_range=(.005, .50), random_start_seed=0,
                verbose=False, show_initialization_progress=False), on_update=checkpoint,
            on_progress=lambda e: print(f'Case {case_index}: loop {e["update"]} {e["stage"]}', flush=True))
        if len(history) != 4 or [t['fit_images'] for t in timings] != [36, 72, 108]:
            raise RuntimeError('Incomplete cumulative three-loop fit.')
        metadata.update(status='complete', wall_seconds=perf_counter()-started,
                        final_residual_rms=history[-1]['residual_rms'])
        write_json(output/'run.json', metadata)
        return metadata
    except Exception as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                        wall_seconds=perf_counter()-started)
        write_json(output/'run.json', metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--case-index', type=int, default=0)
    parser.add_argument('--describe', action='store_true', help='Print array dimensions without computing.')
    parser.add_argument('--prepare-only', action='store_true', help='Save the first 36 noisy frames without fitting.')
    defaults = SweepConfig()
    for name in ('trials', 'size', 'pupil_size', 'sample_planes', 'reconstruction_planes',
                 'random_starts', 'max_iterations', 'aberration_seed', 'noise_seed'):
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=getattr(defaults, name))
    parser.add_argument('--rms-values', nargs='+', type=float, default=DEFAULT_RMS)
    parser.add_argument('--backend', choices=('cuda', 'cpu'), default='cuda')
    args = vars(parser.parse_args())
    output, index = args.pop('output'), args.pop('case_index')
    describe, prepare = args.pop('describe'), args.pop('prepare_only')
    config = SweepConfig(**args)
    try:
        config.validate()
        if describe:
            count = len(config.rms_values)*config.trials
            print(f'{config.trials} paired aberrations x {len(config.rms_values)} RMS values = {count} cases')
            print(f'Slurm array: --array=0-{count-1}%4')
            print('RMS [waves]:', ' '.join(f'{x:g}' for x in config.rms_values))
            return
        if output is None:
            parser.error('--output is required unless --describe is used')
        run_case(config, output, index, prepare_only=prepare)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
