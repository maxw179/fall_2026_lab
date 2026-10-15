"""Scientific controls, real small-grid fitting, and incomplete-array analysis."""
import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np

from optimization import reconstruction
from systematic_testing.gpu_demo.run_demo import MODES, coefficients
from systematic_testing.noise_robustness.noise_calibration import calibrated_noise
from systematic_testing.rms_cliff_sweep.analyze import analyze, load_sweep
from systematic_testing.rms_cliff_sweep.run_sweep import (
    SweepConfig, expected_snr, make_scene, run_case)
from utils import zernike

USE_TEX = shutil.which('latex') is not None


class ScientificControls(unittest.TestCase):
    def test_probe_magnitude_changes_only_the_diversity_phases(self):
        config = SweepConfig(size=8, pupil_size=12, sample_planes=9, reconstruction_planes=3)
        baseline = make_scene(config, 10, 7)
        for magnitude in (.20, .30):
            with self.subTest(diversity_rms=magnitude):
                changed = make_scene(replace(config, diversity_rms=magnitude), 10, 7)
                np.testing.assert_array_equal(changed[2].image_mask, baseline[2].image_mask)
                for index in (3, 4):
                    np.testing.assert_array_equal(changed[index], baseline[index])
                np.testing.assert_array_equal(changed[5].strengths, baseline[5].strengths)
                for original_loop, changed_loop in zip(baseline[6], changed[6]):
                    self.assertEqual(len(changed_loop), 9)
                    for probe, original, target in zip(changed_loop, original_loop,
                            np.tile([magnitude, 0., magnitude], 3)):
                        np.testing.assert_allclose(coefficients(probe),
                            coefficients(original)*magnitude/config.diversity_rms, atol=1e-15)
                        self.assertAlmostEqual(zernike.zernike_RMS(probe, changed[0].alpha), target, places=12)
        for invalid in (0., -.1, np.inf, np.nan):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(config, diversity_rms=invalid).validate()

    def test_prepared_variants_preserve_truth_and_noise_calibration(self):
        config = SweepConfig(size=8, pupil_size=12, sample_planes=9, reconstruction_planes=3)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            outputs = []
            for magnitude in (.10, .20, .30):
                out = Path(temporary)/str(magnitude)
                meta = run_case(replace(config, diversity_rms=magnitude), out, 200, prepare_only=True)
                self.assertEqual(meta['diversity_rms_waves'], [-magnitude, 0., magnitude])
                self.assertEqual(meta['case_index'], 200)
                self.assertEqual(meta['initial_rms'], .30)
                with np.load(out/'case_0200/scene.npz') as scene, \
                     np.load(out/'case_0200/acquisition_1.npz') as acquisition:
                    outputs.append((meta, {key: scene[key].copy() for key in scene.files}))
                    self.assertEqual(acquisition['images'].shape, (36, 8, 8))
            for meta, scene in outputs[1:]:
                baseline, original = outputs[0]
                for key in original:
                    np.testing.assert_array_equal(scene[key], original[key])
                for key in ('exposure', 'gaussian_sigma', 'gaussian_read_sigma_e', 'expected_calibration_snr'):
                    self.assertEqual(meta[key], baseline[key])
                self.assertEqual({k: v for k, v in meta['config'].items() if k != 'diversity_rms'},
                                 {k: v for k, v in baseline['config'].items() if k != 'diversity_rms'})

    def test_total_snr_and_equal_error_power(self):
        clean = np.linspace(.001, 2., 128).reshape(2, 8, 8)
        noise = calibrated_noise(clean, 'gaussian', 1., 10., 7, 8, gaussian_mean_sigma=0.)
        photon_power = clean.sum()/noise['exposure']
        gaussian_power = clean.size*noise['artifact_amplitude']**2
        self.assertAlmostEqual(photon_power, gaussian_power, places=14)
        self.assertAlmostEqual(noise['gaussian_mean'], 0.)
        self.assertAlmostEqual(noise['expected_snr'], 10., places=12)
        self.assertAlmostEqual(expected_snr(clean, noise['exposure'], noise['artifact_amplitude']), 10., places=12)

    def test_same_direction_scaled_to_exact_rms_with_identical_bead(self):
        config = SweepConfig(rms_values=(.1, .3), trials=2, size=8, pupil_size=12,
                             sample_planes=9, reconstruction_planes=3)
        a = make_scene(config, 0, 1)
        b = make_scene(config, 1, 1)
        np.testing.assert_array_equal(a[2].image_mask, b[2].image_mask)
        np.testing.assert_array_equal(a[4], b[4])
        np.testing.assert_allclose(b[5].strengths, 3*a[5].strengths, atol=1e-14)
        self.assertAlmostEqual(zernike.zernike_RMS(a[5], a[0].alpha), .1, places=12)
        self.assertAlmostEqual(zernike.zernike_RMS(b[5], b[0].alpha), .3, places=12)
        c = make_scene(config, 0, 0)
        self.assertFalse(np.allclose(c[4], a[4]))


class SmallSweepChecks:
    backend = 'cpu'
    diversity_rms = .10

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)/'sweep'
        cls.config = SweepConfig(rms_values=(.1, .3), trials=2, size=8, pupil_size=12,
            sample_planes=9, reconstruction_planes=3, random_starts=2,
            max_iterations=1, backend=cls.backend, diversity_rms=cls.diversity_rms)
        cls.fit_inputs = []
        original = reconstruction.ReconstructionWorkspace

        def capture(**kwargs):
            cls.fit_inputs.append((kwargs['images'].copy(),
                                  np.array([coefficients(a) for a in kwargs['diversities']])))
            return original(**kwargs)

        with contextlib.redirect_stdout(io.StringIO()), patch.object(
                reconstruction, 'ReconstructionWorkspace', side_effect=capture):
            run_case(cls.config, cls.output, 0)
            run_case(cls.config, cls.output, 2)
            run_case(cls.config, cls.output, 1, prepare_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_noise_and_historical_slm_offsets(self):
        microscope, _, sample, _, _, truth, _ = make_scene(self.config, 0, 0)
        root = self.output/'case_0000'
        meta = json.loads((root/'run.json').read_text())
        self.assertAlmostEqual(meta['expected_calibration_snr'], 10., places=12)
        self.assertEqual([r['fit_images'] for r in meta['loops']], [36, 72, 108])
        photon = np.random.default_rng(np.random.SeedSequence([1, 0, 0, 0]))
        read = np.random.default_rng(np.random.SeedSequence([1, 0, 0, 1]))
        images, known_phases = [], []
        for loop in range(3):
            with np.load(root/f'acquisition_{loop+1}.npz', allow_pickle=False) as acquisition:
                slm = acquisition['acquisition_slm_strengths']
                clean = microscope.compute_image_3D(sample,
                    truth+zernike.Aberration(MODES, slm), acquisition['focal_z'], mode='vector',
                    diversities=[zernike.Aberration(MODES, s) for s in acquisition['diversity_strengths']])[3]
                clean = np.maximum(clean, 0.)
                expected = photon.poisson(meta['exposure']*clean)/meta['exposure']
                expected += read.normal(0., meta['gaussian_sigma'], clean.shape)
                np.testing.assert_allclose(acquisition['images'], expected, atol=0, rtol=0)
                self.assertTrue(np.any(acquisition['images'] < 0))
                images.append(acquisition['images'])
                known_phases.append(acquisition['diversity_strengths']+slm)
                np.testing.assert_array_equal(self.fit_inputs[loop][0], np.concatenate(images))
                np.testing.assert_allclose(self.fit_inputs[loop][1]+slm,
                                           np.concatenate(known_phases), atol=1e-15)

    def test_saved_rms_uses_pupil_geometry_and_clipped_updates(self):
        for index in (0, 2):
            root = self.output/f'case_{index:04d}'
            _, _, _, _, _, truth, _ = make_scene(self.config, index//2, 0)
            meta = json.loads((root/'run.json').read_text())
            self.assertEqual(meta['status'], 'complete')
            with np.load(root/'results.npz', allow_pickle=False) as result:
                slm = np.zeros(len(MODES))
                for loop in range(3):
                    estimate = result['estimated_strengths'][loop]
                    rms = zernike.zernike_RMS_difference(zernike.Aberration(MODES, estimate),
                        truth+zernike.Aberration(MODES, slm), meta['alpha'])
                    self.assertAlmostEqual(result['estimation_rms'][loop], rms, places=12)
                    slm -= np.clip(estimate, -.1, .1)
                    np.testing.assert_allclose(slm, result['slm_strengths'][loop], atol=1e-15)
                    rms = zernike.zernike_RMS(truth+zernike.Aberration(MODES, slm), meta['alpha'])
                    self.assertAlmostEqual(result['residual_rms'][loop], rms, places=12)

    def test_partial_analysis_reports_missing_cases_without_refitting(self):
        d = load_sweep(self.output)
        self.assertEqual(d['completed'].sum(), 6)
        self.assertEqual(d['status'][0, 1], 'prepared')
        self.assertEqual(d['status'][1, 1], 'missing')
        with self.assertRaisesRegex(ValueError, 'allow-partial'):
            analyze(self.output, use_tex=USE_TEX)
        with patch.object(reconstruction, 'ReconstructionWorkspace', side_effect=AssertionError('Unexpected fit')), \
             contextlib.redirect_stdout(io.StringIO()):
            analyze(self.output, allow_partial=True, use_tex=USE_TEX)
        import csv
        with (self.output/'analysis/summary.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r['completed_trials'] == '1' and r['missing_trials'] == '1' for r in rows))
        for name in ('residual_vs_initial_rms.png', 'estimation_vs_initial_rms.pdf', 'sweep.npz'):
            self.assertTrue((self.output/'analysis'/name).exists())
        with self.assertRaises(FileExistsError):
            run_case(self.config, self.output, 0)

    def test_prepare_only_reproduces_first_acquisition_without_cuda(self):
        prepared = self.output/'preview_repeat'
        with contextlib.redirect_stdout(io.StringIO()):
            run_case(replace(self.config, backend='cuda'), prepared, 0, prepare_only=True)
        with np.load(self.output/'case_0000/acquisition_1.npz') as fitted, \
             np.load(prepared/'case_0000/acquisition_1.npz') as preview:
            np.testing.assert_array_equal(fitted['images'], preview['images'])


class CPUSmallSweep(SmallSweepChecks, unittest.TestCase):
    pass


class CPUDiversity020(SmallSweepChecks, unittest.TestCase):
    diversity_rms = .20


class CPUDiversity030(SmallSweepChecks, unittest.TestCase):
    diversity_rms = .30


@unittest.skipUnless(os.environ.get('REQUIRE_CUDA') == '1', 'Set REQUIRE_CUDA=1 on a GPU node.')
class CUDASmallSweep(SmallSweepChecks, unittest.TestCase):
    backend = 'cuda'


class CUDADiversity020(CUDASmallSweep):
    diversity_rms = .20


class CUDADiversity030(CUDASmallSweep):
    diversity_rms = .30


class SubmissionChecks(unittest.TestCase):
    def test_both_arrays_have_400_cases_and_requested_gpu_partitions(self):
        source = Path(__file__).resolve().parents[1]/'systematic_testing/rms_cliff_sweep/submit_stronger_diversities.sh'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'source'
            script = root/'systematic_testing/rms_cliff_sweep'/source.name
            script.parent.mkdir(parents=True)
            shutil.copy2(source, script)
            bin_dir = Path(temporary)/'bin'
            bin_dir.mkdir()
            calls = Path(temporary)/'sbatch_calls.jsonl'
            fake = bin_dir/'sbatch'
            fake.write_text('#!'+sys.executable+'\n'
                'import json,os,sys\nfrom pathlib import Path\n'
                'p=Path(os.environ["TEST_SBATCH_CALLS"])\n'
                'with p.open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                'print(600000+len(p.read_text().splitlines()))\n')
            fake.chmod(0o755)
            env = dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ.get('PATH', ''),
                       TEST_SBATCH_CALLS=str(calls))
            dry = subprocess.run(['bash', str(script), '--dry-run'], env=env,
                                 capture_output=True, text=True, check=True)
            self.assertEqual(len(dry.stdout.splitlines()), 2)
            self.assertFalse(calls.exists())
            self.assertFalse((root/'logs').exists())
            subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True, check=True)
            arguments = [json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(len(arguments), 2)
            for args, magnitude in zip(arguments, ('0.20', '0.30')):
                self.assertIn('--array=0-399%12', args)
                self.assertIn('--partition=gpu_h100,gpu_h200,gpu_b200,gpu_rtx6000', args)
                self.assertEqual(args[-2:], ['--diversity-rms', magnitude])
                self.assertNotIn('--min-rms', args)
                self.assertFalse(any(a.startswith('--dependency') for a in args))
            subprocess.run(['bash', str(script), '--concurrency', '6', '--partitions', 'gpu_h100,gpu_h200'],
                           env=env, capture_output=True, text=True, check=True)
            overridden = [json.loads(line) for line in calls.read_text().splitlines()][2:]
            self.assertTrue(all('--array=0-399%6' in args for args in overridden))
            self.assertTrue(all('--partition=gpu_h100,gpu_h200' in args for args in overridden))
            invalid = subprocess.run(['bash', str(script), '--concurrency', '0'],
                                     env=env, capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertEqual(len(calls.read_text().splitlines()), 4)


if __name__ == '__main__':
    unittest.main()
