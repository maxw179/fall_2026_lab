"""Data ordering, phase bookkeeping, replay and explicit remote-job lifecycle."""
import json
import os
import subprocess
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
import zipfile
import csv
from io import BytesIO

import numpy as np
from scipy.io import savemat,loadmat
from experimental_10_15.bead_pd_io import load_acquisition,block_average,write_json,performance_settings
from experimental_10_15.bead_pd import prepare_run,compare_run,solve_input,acquisition_grid
from experimental_10_15.submit_bead_pd import bundle_run,submit_run,watch_job,fetch_job
from utils import psf,zernike
from experimental_10_15.quality import score_frames


def fixture(folder):
    modes = np.array([[2,2],[0,4]])
    # Four commands per plane: baseline reference, zero, plus, minus.
    names = np.array(['reference','zero','plus','minus']*2,dtype=object)
    is_pd = np.array([False,True,True,True]*2)
    diversities = np.tile([[0,0],[0,0],[.1,0],[-.1,0]],(2,1))
    baseline = np.array([.03,-.02]); imposed = np.array([.04,.07])
    optics = dict(excitationWavelengthNm=1300,nonlinearOrder=3,NA=1.05,
        refractiveIndex=1.333,focalLengthMm=7.2,magnification=4,beamWaistMm=3.5,
        backFocalPlaneWidthMm=15.12,fieldOfViewUm=6.5,beadDiameterUm=1,focalPositionUm=0)
    meta = dict(config=dict(imageSize=[8,8],optics=optics),coefficientUnits='waves',
        modes_mn=modes,baselineWaves=baseline,imposedWaves=imposed,
        diversityWaves=diversities,totalCommandWaves=baseline+diversities+is_pd[:,None]*imposed,
        names=names,isPDImage=is_pd,planeIndex=np.repeat([1,2],4),
        focalPositionsMm=np.repeat([-.001,.001],4),status='complete')
    images = np.arange(8*8*8,dtype=float).reshape(8,8,8)-20
    logs = np.array([dict(acquired=True,name=name) for name in names],dtype=object)
    savemat(folder/'metadata.mat',dict(metadata=meta))
    savemat(folder/'acquisition.mat',dict(images=images,log=logs))
    return meta,images,logs


class ExperimentalBeadTests(unittest.TestCase):
    def test_border_background_subtraction_is_bundled_with_offsets_and_keeps_signed_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            archive,_=bundle_run(root,root/'background',size=4,pupil_size=8,exclude_images=[2],
                                 flip_x=True,background_border_pixels=1)
            acq=load_acquisition(root);ids=np.array([2,3,5,6,7])
            raw=block_average(acq.images[ids],4)[:,::-1,:]
            border=np.ones((4,4),bool);border[1:-1,1:-1]=False
            offsets=np.median(raw[:,border],axis=1)
            with zipfile.ZipFile(archive) as bundle:
                with np.load(BytesIO(bundle.read('input.npz'))) as data:
                    np.testing.assert_array_equal(data['source_indices'],ids)
                    np.testing.assert_array_equal(data['images'],raw-offsets[:,None,None])
                    np.testing.assert_array_equal(data['image_background_subtracted'],offsets)
                    self.assertTrue(np.any(data['images']<0))
                    np.testing.assert_array_equal(data['diversity_strengths'],acq.diversities[ids]+acq.baseline)
                provenance=json.loads(bundle.read('preparation.json'))
                self.assertEqual(provenance['background_border_pixels'],1)
                np.testing.assert_array_equal(provenance['image_background_subtracted'],offsets)
            with self.assertRaises(ValueError):
                prepare_run(root,root/'invalid',size=4,background_border_pixels=2)

    def test_simulated_input_origin_is_preserved_and_replay_is_labeled(self):
        from experimental_10_15.replay import build_replay
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,_,_=fixture(root)
            meta['config']['dataOrigin']='simulation'
            savemat(root/'metadata.mat',dict(metadata=meta))
            path=prepare_run(root,root/'prepared',size=4,pupil_size=8)
            result=solve_input(path,root/'solve',backend='cpu',sample_planes=3,random_starts=0,max_iterations=1)
            self.assertEqual(result['data_origin'],'simulation')
            with np.load(root/'solve/solution.npz') as data:self.assertEqual(str(data['data_origin']),'simulation')
            frames,_,details=build_replay(root/'solve',path)
            self.assertTrue(frames[0]['initial_image_title'].startswith('Simulated'))
            self.assertEqual(details['data_origin'],'simulation')
            self.assertIn('retained simulated frame',details['notice'])

    def test_x_mirror_changes_pixels_only_and_is_preserved_in_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            original=prepare_run(root,root/'original',size=4,pupil_size=8,exclude_images=[2])
            archive,_=bundle_run(root,root/'flipped',size=4,pupil_size=8,exclude_images=[2],flip_x=True)
            with np.load(original) as before,zipfile.ZipFile(archive) as bundle:
                with np.load(BytesIO(bundle.read('input.npz'))) as after:
                    np.testing.assert_array_equal(after['images'],before['images'][:,::-1,:])
                    self.assertEqual(str(after['image_orientation']),'flip_x')
                    for key in ['source_indices','modes','diversity_strengths','focal_z','imposed_for_validation','baseline']:
                        np.testing.assert_array_equal(after[key],before[key])
                self.assertEqual(json.loads(bundle.read('preparation.json'))['image_orientation'],'flip_x')

    def test_original_one_based_image_exclusions_survive_bundling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            archive,_=bundle_run(root,root/'submission',size=4,pupil_size=8,exclude_images=[1,2,7])
            with zipfile.ZipFile(archive) as bundle:
                with np.load(BytesIO(bundle.read('input.npz'))) as data:
                    np.testing.assert_array_equal(data['source_indices'],[2,3,5,7])
                    original=load_acquisition(root)
                    np.testing.assert_array_equal(data['images'],block_average(original.images[[2,3,5,7]],4))
                    np.testing.assert_array_equal(data['diversity_strengths'],original.diversities[[2,3,5,7]]+original.baseline)
                report=json.loads(bundle.read('preparation.json'))
                self.assertEqual(report['excluded_image_numbers'],[1,2,7])
            self.assertEqual(load_acquisition(root).acquired.sum(),8)
            for numbers in [[0],[9],[1.5],[True]]:
                with self.assertRaisesRegex(ValueError,'1-based'):
                    load_acquisition(root,exclude_images=numbers)

    def test_peak_snr_is_pedestal_and_scale_invariant_on_signed_averages(self):
        rng=np.random.default_rng(24)
        frame=rng.normal(-20,2,(128,128))
        x,y=np.indices(frame.shape)
        frame+=40*np.exp(-((x-65)**2+(y-60)**2)/(2*3**2))
        metrics=score_frames(frame[None])
        self.assertAlmostEqual(metrics['noise_sigma'][0],2.,delta=.08)
        self.assertAlmostEqual(metrics['background'][0],-20.,delta=.1)
        self.assertGreater(metrics['snr'][0],15.)
        self.assertLess(metrics['snr'][0],22.)
        self.assertAlmostEqual(metrics['peak_x'][0],65,delta=1)
        np.testing.assert_allclose(score_frames((frame*7+300)[None])['snr'],metrics['snr'],rtol=1e-12)
        self.assertEqual(score_frames(np.zeros((1,8,8)))['snr'][0],0.)
        noiseless=np.zeros((1,8,8));noiseless[:,3:6,3:6]=10
        with self.assertRaisesRegex(ValueError,'zero background noise'):score_frames(noiseless)

    def test_snr_filter_preserves_selected_pixels_and_bundles_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,_,logs=fixture(root)
            rng=np.random.default_rng(25)
            raw=rng.normal(-30,2,(8,8,8))
            for i in [2,3,6]:raw[3:6,3:6,i]+=70
            savemat(root/'acquisition.mat',dict(images=raw,log=logs))
            path=prepare_run(root,root/'prepared',size=8,pupil_size=8,min_snr=5)
            with np.load(path) as data:
                np.testing.assert_array_equal(data['source_indices'],[2,3,6])
                np.testing.assert_array_equal(data['images'],raw.transpose(2,1,0)[[2,3,6]])
                self.assertTrue(np.all(data['image_snr']>=5))
                self.assertLess(data['images'].min(),0)
                np.testing.assert_array_equal(data['diversity_strengths'],
                    meta['baselineWaves']+meta['diversityWaves'][[2,3,6]])
            report=json.loads((root/'prepared/preparation.json').read_text())
            self.assertEqual(report['snr_excluded_indices'],[1,5,7])
            self.assertTrue(set([1,5,7]).issubset(report['excluded_indices']))
            with (root/'prepared/snr_qc.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),6)
            self.assertEqual(sum(row['included']=='True' for row in rows),3)
            archive,manifest=bundle_run(root,root/'bundle',size=8,pupil_size=8,min_snr=5)
            with zipfile.ZipFile(archive) as z:
                self.assertIn('snr_qc.csv',z.namelist())
                self.assertIn('snr_qc.csv',manifest['sha256'])
            # Only plane 2 has a single good frame; don't submit an empty/one-frame fit.
            with self.assertRaisesRegex(ValueError,'at least two'):
                prepare_run(root,root/'too_few',size=8,pupil_size=8,min_snr=5,exclude_planes=[1])

    def test_signed_data_orientation_and_known_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); meta,raw,_=fixture(root)
            acq=load_acquisition(root)
            self.assertEqual(acq.images[0,2,1],raw[1,2,0])
            self.assertLess(acq.images.min(),0)
            path=prepare_run(root,root/'prepared',size=4,pupil_size=8,background=2,z_sign=-1)
            with np.load(path) as data:
                self.assertEqual(data['images'].shape,(6,4,4))
                np.testing.assert_array_equal(data['source_indices'],[1,2,3,5,6,7])
                np.testing.assert_allclose(data['diversity_strengths'],
                    meta['diversityWaves'][[1,2,3,5,6,7]]+meta['baselineWaves'])
                self.assertAlmostEqual(data['images'][0,0,0],raw[:2,:2,1].mean()-2)
                np.testing.assert_allclose(data['focal_z'],[.001]*3+[-.001]*3)
            grid=acquisition_grid(6.5,(4,4))
            self.assertAlmostEqual(np.diff(grid.get_xy()[0])[0],.0065/4)

    def test_missing_images_and_invalid_phase_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,raw,logs=fixture(root)
            raw[:,:,2]=np.nan;logs[2]['acquired']=False
            savemat(root/'acquisition.mat',dict(images=raw,log=logs))
            path=prepare_run(root,root/'prepared',size=4,pupil_size=8)
            with np.load(path) as data: self.assertNotIn(2,data['source_indices'])
            meta['totalCommandWaves'][0,0]+=.1
            savemat(root/'metadata.mat',dict(metadata=meta))
            with self.assertRaisesRegex(ValueError,'Total commands'):load_acquisition(root)
        with self.assertRaises(ValueError):block_average(np.zeros((2,8,8)),3)

    def test_replay_sphere_radius_and_global_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,_,logs=fixture(root)
            # Generate an independent measured stack from the public forward model.
            meta['config']['imageSize']=[16,16]
            savemat(root/'metadata.mat',dict(metadata=meta))
            grid=acquisition_grid(6.5,(16,16))
            microscope=psf.Microscope(3,.0013,1.333,1.05,7.2,4,3.5,15.12,8)
            sample=psf.bead_img_3D(grid,np.linspace(-.0005,.0005,3),[0],[0],[0],[.0005])
            generated=np.array([microscope.compute_image_3D(sample,zernike.Aberration(meta['modes_mn'],a),[z])[3][0]
                for a,z in zip(meta['totalCommandWaves'],meta['focalPositionsMm'])])
            savemat(root/'acquisition.mat',dict(images=(generated*2.5+3).transpose(2,1,0),log=logs))
            compare_run(root,root/'comparison',size=16,pupil_size=8,bead_planes=3,background=3,use_tex=False)
            report=json.loads((root/'comparison/comparison.json').read_text())
            self.assertAlmostEqual(report['global_gain'],2.5)
            self.assertLess(max(report['relative_rmse']),1e-12)
            self.assertEqual(report['bead_diameter_um'],1)
            self.assertEqual(len(list((root/'comparison/figures').glob('*.png'))),4)

    def test_reference_center_and_plot_settings_come_from_matlab(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,_,logs=fixture(root)
            meta['config']['analysis']=dict(size=8,pupilSize=8,background=7,zSign=1,
                beadCenterUm=np.array([]),beadPlanes=3,useTex=False)
            savemat(root/'metadata.mat',dict(metadata=meta))
            x,y=acquisition_grid(6.5,(8,8)).get_xy()
            spot=np.exp(-((x[:,None]-.00040625)**2+(y[None,:]-.00040625)**2)/(2*.0002**2))*100
            # Another bright bead must not pull the modeled center into empty space.
            spot += np.exp(-((x[:,None]+.00203125)**2+(y[None,:]+.00203125)**2)/(2*.0002**2))*60
            raw=np.repeat((spot.T+7)[:,:,None],8,axis=2)
            savemat(root/'acquisition.mat',dict(images=raw,log=logs))
            compare_run(root,root/'comparison')
            report=json.loads((root/'comparison/comparison.json').read_text())
            np.testing.assert_allclose(report['bead_center_um'],[.40625,.40625,0],atol=1e-10)
            self.assertEqual(report['background_subtracted'],7)

    def test_solver_truth_is_not_used_and_outputs_are_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            path=prepare_run(root,root/'prepared',size=4,pupil_size=8)
            class Workspace:
                def optimize(self,**kwargs):
                    self_case.assertNotIn('initial_strengths',kwargs)
                    kwargs['on_iteration'](dict(iteration=0,loss=2.,strengths=np.zeros(2)))
                    return zernike.Aberration([[2,2],[0,4]],[.02,.01]),dict(success=False,message='limit',history=[])
                def estimate_sample(self,strengths):
                    return type('Sample',(),dict(image_mask=np.zeros((3,4,4))))()
            self_case=self
            with patch('optimization.reconstruction.ReconstructionWorkspace',return_value=Workspace()) as create:
                result=solve_input(path,root/'solution',backend='cpu',sample_planes=3,random_starts=0)
                np.testing.assert_allclose(create.call_args.kwargs['diversities'][0].strengths,[.03,-.02])
                self.assertNotIn('svd_method',create.call_args.kwargs)
                self.assertNotIn('screening_batch_size',create.call_args.kwargs)
            self.assertFalse(result['optimizer_success'])
            with np.load(root/'solution/solution.npz') as data:
                np.testing.assert_allclose(data['correction_strengths'],[-.02,-.01])
            self.assertEqual(json.loads((root/'solution/progress.json').read_text())['status'],'complete')

    def test_cuda_execution_uses_fast_defaults_and_throttles_shared_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            path=prepare_run(root,root/'prepared',size=4,pupil_size=8)
            class Workspace:
                svd_counts = dict(qr=22,direct=1)
                def optimize(self,**kwargs):
                    # A whole Hessian batch storm occurs before time advances.
                    for completed in range(153):
                        kwargs['on_progress'](dict(stage='second_derivatives',iteration=0,
                                                  completed=completed,total=153))
                    kwargs['on_iteration'](dict(iteration=0,loss=2.,strengths=np.zeros(2)))
                    return zernike.Aberration([[2,2],[0,4]],[0,0]),dict(
                        success=True,message='done',history=[dict(iteration=0,loss=2.,strengths=[0,0])])
                def estimate_sample(self,strengths):
                    return type('Sample',(),dict(image_mask=np.zeros((3,4,4))))()
            with patch('optimization.reconstruction.get_backend',return_value='cuda'), \
                 patch('optimization.reconstruction.ReconstructionWorkspace',return_value=Workspace()) as create, \
                 patch('experimental_10_15.bead_pd.perf_counter',return_value=100.), \
                 patch('experimental_10_15.bead_pd.write_json',wraps=write_json) as writes:
                result=solve_input(path,root/'solution',sample_planes=3,random_starts=0)
            options=create.call_args.kwargs
            self.assertEqual(options['svd_method'],'qr')
            self.assertEqual(options['screening_method'],'linear')
            self.assertEqual(options['screening_fallback'],'qr')
            self.assertEqual(options['screening_batch_size'],8)
            for key in ('reuse_svd','spatial_hessian','deduplicate_psfs','retain_fields','cache_last_fit'):
                self.assertTrue(options[key])
            progress_writes=[call for call in writes.call_args_list if call.args[0].name=='progress.json']
            self.assertEqual(len(progress_writes),4)  # starting, stage, evaluated state, complete
            self.assertEqual(result['runtime']['svd_counts'],Workspace.svd_counts)
            runtime=json.loads((root/'solution/runtime.json').read_text())
            self.assertEqual(runtime,result['runtime'])
            history=json.loads((root/'solution/optimizer.json').read_text())['history']
            self.assertEqual(history[0]['elapsed_seconds'],0.)

    def test_cpu_solve_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            path=prepare_run(root,root/'prepared',size=4,pupil_size=8)
            solve_input(path,root/'solution',backend='cpu',sample_planes=3,random_starts=0,max_iterations=1)
            self.assertTrue((root/'solution/solution.npz').exists())

    def test_bundle_and_remote_lifecycle_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            with patch('experimental_10_15.submit_bead_pd.remote_command') as remote:
                archive,manifest=bundle_run(root,root/'bundle',size=4,pupil_size=8,random_starts=0)
                remote.assert_not_called()
            self.assertEqual(manifest['performance']['svd_method'],'qr')
            self.assertIn('--svd-method',manifest['solver_args'])
            self.assertIn('--screening-batch-size',manifest['solver_args'])
            with zipfile.ZipFile(archive) as zip:
                self.assertIn('original_acquisition/metadata.mat',zip.namelist())
                self.assertIn('original_acquisition/acquisition.mat',zip.namelist())
                self.assertIn('optimization/reconstruction_cuda.py',zip.namelist())
                self.assertIn('input.npz',zip.namelist())
                self.assertIn('experimental_10_15/bead_pd.py',zip.namelist())
                self.assertIn('experimental_10_15/bouchet_bead_pd.sbatch',zip.namelist())
                isolated=root/'isolated'
                zip.extractall(isolated)
            # Execute the uploaded snapshot outside the repository, without its
            # PYTHONPATH, to detect missing package files and renamed imports.
            subprocess.run([sys.executable,'-m','experimental_10_15.bead_pd','solve',
                'input.npz','--output','smoke_results','--backend','cpu',
                *manifest['solver_args'],
                '--sample-planes','3','--random-starts','0','--max-iterations','1'],
                cwd=isolated,env={**os.environ,'PYTHONPATH':''},
                check=True,capture_output=True,text=True)
            self.assertTrue((isolated/'smoke_results/correction.mat').exists())
            with patch('experimental_10_15.submit_bead_pd.remote_command',side_effect=['','','12345\n']) as remote, \
                    patch('experimental_10_15.submit_bead_pd.subprocess.run',return_value=
                        subprocess.CompletedProcess([],0,stdout='testrevision\n')) as process:
                receipt=submit_run(root,root/'submission',host='mcw77@my-login',size=4,pupil_size=8)
                self.assertIn('sbatch --parsable',remote.call_args.args[1])
                self.assertTrue(receipt.exists())
            with patch('experimental_10_15.submit_bead_pd.remote_command',side_effect=[
                '{"status":"complete","optimizer_success":true}', '', 'COMPLETED|\n']):
                update=watch_job(receipt,once=True)
                self.assertEqual(update['status'],'complete')
            with patch('experimental_10_15.submit_bead_pd.subprocess.run') as process:
                fetch_job(receipt,root/'fetched')
                self.assertEqual(process.call_count,4)
                self.assertTrue((root/'fetched/job.json').exists())
            with self.assertRaises(ValueError):fetch_job(receipt,root/'fetched')

    def test_execution_overrides_are_bundled_and_invalid_batches_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            archive,manifest=bundle_run(root,root/'bundle',size=4,pupil_size=8,
                svd_method='direct',screening_batch_size=2,psf_batch_size=4,
                derivative_batch_size=3,frequency_batch_size=64)
            with zipfile.ZipFile(archive) as zip:
                self.assertEqual(json.loads(zip.read('bundle.json'))['performance'],manifest['performance'])
            args=manifest['solver_args']
            self.assertEqual(args[args.index('--svd-method')+1],'direct')
            self.assertEqual(args[args.index('--derivative-batch-size')+1],'3')
            for value in (0,-1,True,1.5):
                with self.assertRaises(ValueError):performance_settings(psf_batch_size=value)
            with self.assertRaises(ValueError):
                bundle_run(root,root/'invalid',screening_batch_size=0)
            self.assertFalse((root/'invalid').exists())

    def test_ssh_and_scp_share_a_persistent_connection(self):
        from experimental_10_15.submit_bead_pd import ssh_options,remote_command
        host='mcw77@test-login'
        options=ssh_options(host)
        self.assertIn('ControlMaster=auto',options)
        self.assertIn('ControlPersist=600',options)
        self.assertEqual(options,ssh_options(host))
        self.assertNotEqual(options,ssh_options('someone@another-login'))
        with patch('experimental_10_15.submit_bead_pd.subprocess.run',return_value=
                   subprocess.CompletedProcess([],0,stdout='ok')) as run:
            self.assertEqual(remote_command(host,'hostname'),'ok')
            self.assertEqual(run.call_args.args[0],['ssh',*options,host,'hostname'])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);receipt=root/'job.json'
            receipt.write_text(json.dumps(dict(host=host,job_id='12345',remote_dir='/tmp/test-job')))
            with patch('experimental_10_15.submit_bead_pd.subprocess.run') as run:
                fetch_job(receipt,root/'fetched')
                for call in run.call_args_list:
                    self.assertEqual(call.args[0][:1+len(options)],['scp',*options])

    def test_watcher_retries_status_errors_and_reports_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt=Path(tmp)/'job.json'
            receipt.write_text(json.dumps(dict(host='mcw77@test-login',job_id='12345',remote_dir='/tmp/test-job')))
            failure=subprocess.CalledProcessError(1,['ssh'],stderr='Transient remote filesystem error')
            with patch('experimental_10_15.submit_bead_pd.remote_command',side_effect=[failure,
                    '{"status":"complete"}', '', 'COMPLETED|']), \
                    patch('experimental_10_15.submit_bead_pd.time.sleep') as sleep, \
                    patch('experimental_10_15.submit_bead_pd.submit_run') as submit, \
                    patch('sys.stderr') as stderr:
                result=watch_job(receipt,interval=1)
                self.assertEqual(result['status'],'complete')
                sleep.assert_called_once_with(1)
                submit.assert_not_called()
                self.assertIn('Transient remote filesystem error',str(stderr.write.call_args_list))
            with patch('experimental_10_15.submit_bead_pd.remote_command',side_effect=failure), \
                    patch('experimental_10_15.submit_bead_pd.time.sleep'),patch('sys.stderr'):
                with self.assertRaisesRegex(RuntimeError,'not cancelled or resubmitted'):
                    watch_job(receipt,interval=1,max_failures=2)
            invalid=subprocess.CalledProcessError(1,['ssh'],stderr='Invalid job id specified')
            with patch('experimental_10_15.submit_bead_pd.remote_command',side_effect=[
                    '{"status":"complete"}',invalid,'COMPLETED|']):
                self.assertEqual(watch_job(receipt,once=True)['status'],'complete')

    def test_solver_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            path=prepare_run(root,root/'prepared',size=4,pupil_size=8)
            with patch('optimization.reconstruction.ReconstructionWorkspace',side_effect=RuntimeError('no device')):
                with self.assertRaisesRegex(RuntimeError,'no device'):
                    solve_input(path,root/'failed')
            progress=json.loads((root/'failed/progress.json').read_text())
            self.assertEqual(progress['status'],'failed')
            self.assertEqual(progress['error'],'no device')

    def test_one_command_pipeline_and_invalid_host(self):
        from experimental_10_15.submit_bead_pd import main
        with patch.object(sys,'argv',['submit_bead_pd','submit','RUN','--output','OUTPUT',
                '--host','mcw77@my-login','--fetch-output','FETCHED']), \
                patch('experimental_10_15.submit_bead_pd.submit_run',return_value=Path('job.json')) as submit, \
                patch('experimental_10_15.submit_bead_pd.watch_job') as watch, \
                patch('experimental_10_15.submit_bead_pd.fetch_job') as fetch:
            main()
            submit.assert_called_once()
            watch.assert_called_once_with(Path('job.json'))
            fetch.assert_called_once_with(Path('job.json'),Path('FETCHED'))
        with patch('experimental_10_15.submit_bead_pd.remote_command') as remote:
            with self.assertRaises(ValueError):submit_run('RUN','OUTPUT',host='-oProxyCommand=bad')
            remote.assert_not_called()

    def test_metadata_settings_and_cumulative_correction_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,raw,logs=fixture(root)
            current=np.array([-.02,-.01])
            meta.update(cumulativeCorrectionWaves=current,loopNumber=2,runId='loop_two')
            meta['config']['zReferenceReadoutUm']=12.3
            meta['config']['analysis']=dict(size=4,pupilSize=8,background=2,zSign=-1)
            meta['config']['solver']=dict(rho=1e44,samplePlanes=3,sampleExtentUm=2,
                                         maxIterations=1,randomStarts=0)
            meta['totalCommandWaves']+=meta['isPDImage'][:,None]*current
            savemat(root/'metadata.mat',dict(metadata=meta))
            input_path=prepare_run(root,root/'prepared')
            with np.load(input_path) as data:
                self.assertEqual(data['images'].shape,(6,4,4))
                self.assertAlmostEqual(data['images'][0,0,0],raw[:2,:2,1].mean()-2)
                np.testing.assert_allclose(data['diversity_strengths'][0],meta['baselineWaves'])
                np.testing.assert_allclose(data['expected_residual_for_validation'],[.02,.06])
            class Workspace:
                def optimize(self,**kwargs):
                    self_case.assertEqual(kwargs['max_iterations'],1)
                    self_case.assertEqual(kwargs['num_random_starts'],0)
                    return zernike.Aberration([[2,2],[0,4]],[.02,.06]),dict(success=True,message='ok',history=[])
                def estimate_sample(self,strengths):
                    return type('Sample',(),dict(image_mask=np.zeros((3,4,4))))()
            self_case=self
            with patch('optimization.reconstruction.ReconstructionWorkspace',return_value=Workspace()) as create:
                solve_input(input_path,root/'solution',backend='cpu')
                self.assertEqual(create.call_args.args[6],1e44)
            correction=loadmat(root/'solution/correction.mat',simplify_cells=True)
            np.testing.assert_allclose(correction['cumulativeCorrectionWaves'],[-.04,-.07])
            self.assertTrue(correction['optimizerSuccess'])
            self.assertEqual(correction['parentRunId'],'loop_two')
            self.assertEqual(correction['parentLoopNumber'],2)
            self.assertEqual(correction['zReferenceReadoutUm'],12.3)
            # Mirror the next MATLAB command: original imposed + NEW cumulative.
            meta['cumulativeCorrectionWaves']=correction['cumulativeCorrectionWaves']
            meta['totalCommandWaves']=meta['baselineWaves']+meta['diversityWaves']+meta['isPDImage'][:,None]*(
                meta['imposedWaves']+meta['cumulativeCorrectionWaves'])
            savemat(root/'metadata.mat',dict(metadata=meta))
            following=load_acquisition(root)
            np.testing.assert_allclose(following.imposed+following.cumulative,0,atol=1e-15)
            _,manifest=bundle_run(root,root/'bundle')
            self.assertEqual(manifest['solver_args'][1],'1e+44')
            self.assertEqual(manifest['solver_args'][manifest['solver_args'].index('--random-starts')+1],'0')

    def test_system_ao_is_preserved_as_fixed_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,_,_=fixture(root)
            phase=np.array([[0.,64.,128.],[192.,254.,255.]])
            meta['systemAOPhaseSLM']=phase;meta['slmSize']=[2,3]
            savemat(root/'metadata.mat',dict(metadata=meta))
            path=prepare_run(root,root/'prepared',size=4,pupil_size=8)
            with np.load(path) as data:
                np.testing.assert_array_equal(data['system_ao_phase_slm'],phase)
                # Fit the phase remaining after the system calibration, without
                # counting that same map again as a known Zernike diversity.
                np.testing.assert_allclose(data['diversity_strengths'][0],meta['baselineWaves'])
            class Workspace:
                def optimize(self,**kwargs):
                    return zernike.Aberration([[2,2],[0,4]],[.02,.01]),dict(success=True,message='ok',history=[])
                def estimate_sample(self,strengths):
                    return type('Sample',(),dict(image_mask=np.zeros((3,4,4))))()
            with patch('optimization.reconstruction.ReconstructionWorkspace',return_value=Workspace()):
                solve_input(path,root/'solution',backend='cpu',sample_planes=3,random_starts=0)
            exported=loadmat(root/'solution/correction.mat',simplify_cells=True)
            np.testing.assert_array_equal(exported['systemAOPhaseSLM'],phase)
            meta['systemAOPhaseSLM'][0,0]=256
            savemat(root/'metadata.mat',dict(metadata=meta))
            with self.assertRaisesRegex(ValueError,'System AO'):load_acquisition(root)
            meta['systemAOPhaseSLM']=phase[:1]
            savemat(root/'metadata.mat',dict(metadata=meta))
            with self.assertRaisesRegex(ValueError,'System AO'):load_acquisition(root)

    def test_replay_compares_saved_estimates_to_metadata_imposed_phase(self):
        from experimental_10_15.replay import build_replay,export_replay
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,_,_=fixture(root)
            path=prepare_run(root,root/'prepared',size=4,pupil_size=8)
            results=root/'results';results.mkdir()
            modes=meta['modes_mn'];final=np.array([.1,.05])
            history=[dict(iteration=0,strengths=[0.,0.],loss=2.),
                     dict(iteration=1,strengths=final.tolist(),loss=1.)]
            (results/'optimizer.json').write_text(json.dumps(dict(history=history)))
            (results/'result.json').write_text(json.dumps(dict(optimizer_success=True,loop_number=1,elapsed_seconds=10)))
            with np.load(path) as data:
                np.savez(results/'solution.npz',modes=modes,estimated_strengths=final,
                    focal_z=data['focal_z'],cumulative_correction=np.zeros(2),
                    sample=np.ones((3,4,4)),sample_z=[-.001,0,.001])
            with patch.object(psf.Microscope,'compute_image_3D',return_value=(None,None,None,np.ones((1,4,4)))) as forward:
                frames,points,details=build_replay(results,path)
                forward.assert_not_called()
            self.assertEqual(len(frames),3)
            self.assertTrue(all(f['count']==6 for f in frames))
            alpha=np.arcsin(1.05/1.333)
            expected=zernike.zernike_RMS(zernike.Aberration(modes,final-meta['imposedWaves']),alpha)
            self.assertAlmostEqual(details['estimation_error_rms_waves'][-1],expected)
            self.assertGreater(expected,0)
            self.assertTrue(all(np.array_equal(f['image'],frames[0]['image']) for f in frames))
            self.assertTrue(all(f['show_loss_panel'] for f in frames))
            self.assertFalse(details['per_iteration_times_available'])
            self.assertNotIn('core_elapsed_seconds',frames[-1])
            self.assertIn('Saved iteration',frames[-1]['label'])
            np.testing.assert_allclose(details['estimated_imposed_waves'][-1],final)
            np.testing.assert_allclose(details['imposed_waves'],meta['imposedWaves'])
            with patch('experimental_10_15.replay.build_replay',return_value=(frames,points,details)), \
                    patch('experimental_10_15.replay.render_replay',return_value='<body>viewer</body>'):
                html=export_replay(results,path,root/'replay.html')
            self.assertIn('reference wavefront is the imposed',html.read_text())
            # A strict SNR cut may leave probes only. Replay must use a retained
            # probe with an honest label, without a new fit or rejected pixels.
            with np.load(path) as data:
                values={key:data[key] for key in data.files}
            probes=~np.all(np.isclose(values['diversity_strengths'],values['baseline']),axis=1)
            for key in ['images','focal_z','diversity_strengths','source_indices','names','plane_index']:
                values[key]=values[key][probes]
            probe_path=root/'probes_only.npz';np.savez(probe_path,**values)
            np.savez(results/'solution.npz',modes=modes,estimated_strengths=final,
                focal_z=values['focal_z'],cumulative_correction=np.zeros(2))
            with patch.object(psf.Microscope,'compute_image_3D',side_effect=AssertionError('Replay must not simulate')):
                probe_frames,_,probe_details=build_replay(results,probe_path)
            self.assertFalse(probe_details['displayed_image_is_zero_diversity'])
            self.assertIn('Measured probe',probe_frames[0]['initial_image_title'])
            source=probe_details['displayed_image_source_index']
            self.assertIn(source,values['source_indices'])
            index=int(np.flatnonzero(values['source_indices']==source)[0])
            np.testing.assert_array_equal(probe_frames[0]['image'],values['images'][index])
            np.testing.assert_allclose(probe_details['displayed_image_diversity_waves'],
                                       values['diversity_strengths'][index]-values['baseline'])

    def test_v73_import_when_h5py_available(self):
        try:import h5py
        except ImportError:self.skipTest('h5py is optional for legacy v7.3 acquisitions')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,raw,_=fixture(root)
            hdf=root/'acquisition.mat'
            with h5py.File(hdf,'w',userblock_size=512) as file:
                file.create_dataset('images',data=raw.transpose(2,1,0))
                refs=file.create_group('#refs#'); log=file.create_group('log')
                values=[]
                for i in range(8): values.append(refs.create_dataset(str(i),data=[[1]]).ref)
                log.create_dataset('acquired',data=np.array(values,dtype=h5py.ref_dtype).reshape(1,8))
            with hdf.open('r+b') as stream:
                header=b'MATLAB 7.3 MAT-file, Platform: test, Created on: test HDF5 schema 1.00 .'
                stream.write(header.ljust(116,b' ')+b'\x00'*8+b'\x00\x02IM')
            acquisition=load_acquisition(root)
            np.testing.assert_array_equal(acquisition.images,raw.transpose(2,1,0))
            self.assertTrue(acquisition.acquired.all())


if __name__ == '__main__':unittest.main()
