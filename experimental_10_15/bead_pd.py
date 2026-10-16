"""Compare measured bead stacks, prepare solver input, and fit measured aberrations.

Run from the repository root: python -m experimental_10_15.bead_pd --help.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from experimental_10_15.bead_pd_io import (load_acquisition,block_average,optics_parameters,write_json,
                                     analysis_settings,solver_settings,performance_settings,subtract_border_background)
from utils import psf,zernike


def acquisition_grid(field_of_view_um, shape):
    # ScanImage pixels represent bins over the FOV; model axes are endpoint-inclusive.
    nx,ny = shape
    fov = float(field_of_view_um)/1000
    return psf.Arbitrary_Grid(fov*(nx-1)/nx,fov*(ny-1)/ny,nx,ny,0,0,0)


def prepare_run(run, output, *, size=None, pupil_size=None, background=None, z_sign=None, exclude_planes=None, exclude_images=None,
                min_snr=None,flip_x=False,background_border_pixels=None):
    """Keep all measured PD images; exclude reference frames and missing acquisitions."""
    acquisition = load_acquisition(run,exclude_planes=exclude_planes,exclude_images=exclude_images)
    options = analysis_settings(acquisition,size=size,pupil_size=pupil_size,background=background,z_sign=z_sign)
    size,pupil_size,background,z_sign = (options[k] for k in ('size','pupil_size','background','z_sign'))
    if z_sign not in (-1,1) or not np.isfinite(background) or pupil_size < 4:
        raise ValueError('Invalid focus sign, background, or pupil size.')
    selected = np.flatnonzero(acquisition.acquired & acquisition.is_pd)
    if len(selected) < 2:
        raise ValueError('At least two acquired PD images are required; dry runs have no measurements.')
    images = block_average(acquisition.images[selected],size)-background
    if flip_x:images=images[:,::-1,:].copy()
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    quality_data={};quality_report={}
    if background_border_pixels is not None:
        images,offsets=subtract_border_background(images,background_border_pixels)
        quality_data.update(background_border_pixels=int(background_border_pixels),
                            image_background_subtracted=offsets+background)
        quality_report.update(background_method='per-image border median after block means',
                              background_border_pixels=int(background_border_pixels))
    if flip_x:quality_data['image_orientation']=np.array('flip_x')
    data_origin=str(acquisition.config.get('dataOrigin','measured'))
    if data_origin!='measured':quality_data['data_origin']=np.array(data_origin)
    if min_snr is not None:
        from experimental_10_15.quality import score_frames,write_snr_csv,SNR_METHOD
        if not np.isfinite(min_snr) or min_snr < 0:
            raise ValueError('min_snr must be finite and nonnegative.')
        metrics=score_frames(images);keep=metrics['snr']>=min_snr
        write_snr_csv(output/'snr_qc.csv',acquisition,selected,metrics,min_snr)
        quality_report.update(dict(min_snr=float(min_snr),snr_method=SNR_METHOD,
            snr_candidate_indices=selected.copy(),candidate_snr=metrics['snr'],
            snr_excluded_indices=selected[~keep],snr_included_count=int(keep.sum())))
        if 'image_background_subtracted' in quality_data:
            quality_data['image_background_subtracted']=quality_data['image_background_subtracted'][keep]
        quality_data.update(image_snr=metrics['snr'][keep],snr_method_json=np.array(json.dumps(SNR_METHOD)),
                            min_snr=float(min_snr))
        selected=selected[keep];images=images[keep]
        if len(selected)<2:
            raise ValueError(f'SNR screening leaves {len(selected)} images; at least two are required. See snr_qc.csv.')
    params = optics_parameters(acquisition.optics,pupil_size)
    if background_border_pixels is not None:
        quality_report['image_background_subtracted']=quality_data['image_background_subtracted']
    # Baseline is known; fit remaining imposed + native + cumulative phase.
    # The imposed coefficients are NEVER inserted into diversities or initialization.
    known = acquisition.diversities[selected]+acquisition.baseline
    np.savez_compressed(output/'input.npz', images=images, modes=acquisition.modes,
        focal_z=z_sign*acquisition.focal_z[selected], diversity_strengths=known,
        source_indices=selected, names=acquisition.names[selected],
        plane_index=acquisition.plane_index[selected],
        imposed_for_validation=acquisition.imposed, baseline=acquisition.baseline,
        system_ao_phase_slm=acquisition.system_ao,
        cumulative_correction=acquisition.cumulative,
        expected_residual_for_validation=acquisition.imposed+acquisition.cumulative,
        loop_number=acquisition.loop_number,run_id=np.array(acquisition.run_id),
        z_reference_um=np.array(acquisition.config.get('zReferenceReadoutUm',[]),dtype=float),
        optics_json=np.array(json.dumps(acquisition.optics)),
        solver_json=np.array(json.dumps(solver_settings(acquisition))),
        field_of_view_um=float(acquisition.optics['fieldOfViewUm']),
        microscope_json=np.array(json.dumps(params)),**quality_data)
    write_json(output/'preparation.json', dict(source=acquisition.source,source_status=acquisition.status,
        excluded_planes=list(exclude_planes or []),excluded_image_numbers=list(exclude_images or []),selected_indices=selected,
        excluded_indices=np.setdiff1d(np.arange(len(acquisition.images)),selected),
        input_shape=images.shape,data_origin=data_origin,background_subtracted=background,z_sign=z_sign,**quality_report,
        array_order='image,x(column),y(row)',image_orientation='flip_x' if flip_x else 'identity',
        downsampling='block mean, no per-image normalization',
        field_of_view_um=acquisition.optics['fieldOfViewUm'],microscope=params,
        unknown='remaining imposed + native + fixed system AO + cumulative correction; baseline is a known offset',
        system_ao_treatment='Fixed system calibration; forward comparison assumes it cancels native aberration.',
        cumulative_correction_waves=acquisition.cumulative,loop_number=acquisition.loop_number,
        stage_positions='manually confirmed requested offsets; not measured'))
    return output/'input.npz'


def compare_run(run, output, *, size=None, pupil_size=None, background=None, z_sign=None,
                bead_center_um=None, bead_planes=None, use_tex=None, exclude_planes=None,min_snr=None,exclude_images=None,flip_x=False,
                background_border_pixels=None):
    """Replay every acquired command for a uniform fluorescent sphere."""
    from systematic_testing.correction_examples.plot_pd_correction_examples import plot_acquisition_comparison
    import matplotlib.pyplot as plt
    acquisition = load_acquisition(run,exclude_planes=exclude_planes,exclude_images=exclude_images)
    options = analysis_settings(acquisition,size=size,pupil_size=pupil_size,background=background,
        z_sign=z_sign,bead_center_um=bead_center_um,bead_planes=bead_planes,use_tex=use_tex)
    size,pupil_size,background,z_sign,bead_center_um,bead_planes,use_tex = (options[k] for k in
        ('size','pupil_size','background','z_sign','bead_center_um','bead_planes','use_tex'))
    if bead_planes < 3 or bead_planes % 2 != 1:
        raise ValueError('Bead sample plane count must be odd and >=3.')
    input_path = prepare_run(run,output,size=size,pupil_size=pupil_size,background=background,z_sign=z_sign,
                            exclude_planes=exclude_planes,min_snr=min_snr,exclude_images=exclude_images,flip_x=flip_x,
                            background_border_pixels=background_border_pixels)
    indices = np.flatnonzero(acquisition.acquired)
    measured = block_average(acquisition.images[indices],size)-background
    if flip_x:measured=measured[:,::-1,:].copy()
    if background_border_pixels is not None:measured,_=subtract_border_background(measured,background_border_pixels)
    grid = acquisition_grid(acquisition.optics['fieldOfViewUm'],measured.shape[-2:])
    microscope = psf.Microscope(**optics_parameters(acquisition.optics,pupil_size))
    if np.asarray(bead_center_um).size == 0:
        reference = np.flatnonzero(~acquisition.is_pd[indices])
        if not len(reference):
            raise ValueError('Automatic bead centering needs an acquired reference image.')
        ref = reference[np.argmin(np.abs(acquisition.focal_z[indices[reference]]))]
        frame = measured[ref]
        edge = np.concatenate([frame[0],frame[-1],frame[:,0],frame[:,-1]])
        from scipy.ndimage import gaussian_filter,label
        signal = np.maximum(frame-np.median(edge),0)
        smoothed = gaussian_filter(signal,1.)
        if not smoothed.max(): raise ValueError('No bead signal found in reference image.')
        # A field can contain several beads: do not center the model between them.
        components,_ = label(smoothed >= .3*smoothed.max())
        peak = np.unravel_index(np.argmax(smoothed),smoothed.shape)
        weights = signal*(components == components[peak])
        if not weights.sum(): raise ValueError('No bead signal found in reference image.')
        x,y = grid.get_xy()
        bead_center_um = (1000*np.sum(weights*x[:,None])/weights.sum(),
                          1000*np.sum(weights*y[None,:])/weights.sum(),0.)
    center = np.asarray(bead_center_um,dtype=float)/1000
    radius = float(acquisition.optics['beadDiameterUm'])/2000  # API expects radius, not diameter
    if center.shape != (3,) or not np.all(np.isfinite(center)) or radius <= 0:
        raise ValueError('Bead center and diameter must be finite, diameter positive.')
    sample_z = np.linspace(center[2]-radius,center[2]+radius,bead_planes)
    sample = psf.bead_img_3D(grid,sample_z,[center[0]],[center[1]],[center[2]],[radius])
    if not np.any(sample.image_mask):
        raise ValueError('Bead is outside the lateral grid or too small for this pixel sampling.')
    commands = [zernike.Aberration(acquisition.modes,a) for a in acquisition.total[indices]]
    focus = z_sign*acquisition.focal_z[indices]
    print(f'Simulating {len(indices)} acquired images at size {measured.shape[-1]}...',flush=True)
    predictions = []
    cache = {}
    for row,(command,z) in enumerate(zip(commands,focus),1):
        key = (float(z),tuple(command.strengths))
        if key not in cache:
            cache[key] = microscope.compute_image_3D(sample,command,[z],mode='vector')[3][0]
        predictions.append(cache[key])
        print(f'Simulated {row}/{len(indices)}',flush=True)
    simulated = np.asarray(predictions)
    # One global nonnegative gain. Never normalize each plane separately.
    sim_scale = float(simulated.max())
    if sim_scale <= 0 or not np.isfinite(sim_scale):
        raise ValueError('The simulated bead has no finite positive signal.')
    unit = simulated/sim_scale
    gain = max(0.,float(np.sum(unit*measured)/np.sum(unit*unit)))
    if gain == 0:
        raise ValueError('Measured data has no positive correlation with the modeled bead; check background/alignment.')
    matched = unit*gain
    rmse = np.sqrt(np.mean((measured-matched)**2,axis=(1,2)))
    denom = np.maximum(np.sqrt(np.mean(measured**2,axis=(1,2))),np.finfo(float).tiny)
    report = dict(global_gain=gain/sim_scale,background_subtracted=background,
        source_indices=indices,names=acquisition.names[indices],focal_z_mm=focus,
        rmse=rmse,relative_rmse=rmse/denom,bead_center_um=bead_center_um,
        bead_diameter_um=2*radius*1000,bead_planes=bead_planes,
        model='3P vector optics, uniform sphere, Zernike commands; fixed system AO assumed to cancel native phase; no noise model',
        normalization='one least-squares brightness scale across all acquired frames',
        centering='automatic: brightest connected reference object' if np.asarray(options['bead_center_um']).size == 0 else 'explicit center',
        excluded_planes=list(exclude_planes or []))
    output = Path(output)
    np.savez_compressed(output/'comparison.npz', measured=measured,simulated_raw=simulated,
        simulated_scaled=matched,source_indices=indices,focal_z=focus,modes=acquisition.modes,
        command_strengths=acquisition.total[indices],rmse=rmse)
    write_json(output/'comparison.json',report)
    figures = output/'figures'; figures.mkdir(exist_ok=True)
    peak = max(float(measured.max()),float(matched.max()))
    nx,ny = measured.shape[-2:]
    cx,cy = max(1,int(.35*nx)),max(1,int(.35*ny))
    corner_slices = {'upper left':(slice(0,cx),slice(-cy,None)),
        'upper right':(slice(-cx,None),slice(-cy,None)),
        'lower left':(slice(0,cx),slice(0,cy)),
        'lower right':(slice(-cx,None),slice(0,cy))}
    energy = np.maximum(measured,0).sum(axis=0)+matched.sum(axis=0)
    inset_loc = min(corner_slices,key=lambda name:energy[corner_slices[name]].sum())
    report['phase_inset_location'] = inset_loc
    write_json(output/'comparison.json',report)
    # One figure per command slot, with Z rows and measured/simulated columns.
    # Repeated image names are intentional: the same command occurs at each Z.
    ordered_names = list(dict.fromkeys(acquisition.names[indices]))
    for number,name in enumerate(ordered_names,1):
        chosen = np.flatnonzero(acquisition.names[indices] == name)
        for page,start in enumerate(range(0,len(chosen),4),1):
            rows = chosen[start:start+4]
            fig,_ = plot_acquisition_comparison(grid,measured[rows]/peak,matched[rows]/peak,
                [commands[i] for i in rows],focus[rows],microscope.alpha,
                title=f'Command {number}: '+str(name).replace('_',' '),use_tex=use_tex,inset_loc=inset_loc)
            fig.savefig(figures/f'command_{number:02d}_page_{page:02d}.png',dpi=300)
            fig.savefig(figures/f'command_{number:02d}_page_{page:02d}.pdf',dpi=300)
            plt.close(fig)
    return input_path


def solve_input(input_path, output, *, rho=None, backend='cuda', sample_planes=None,
                sample_extent_um=None, max_iterations=None, random_starts=None,
                svd_method=None, screening_batch_size=None, psf_batch_size=None,
                derivative_batch_size=None, frequency_batch_size=None):
    import json
    from optimization import reconstruction
    with np.load(input_path,allow_pickle=False) as data:
        saved = json.loads(str(data['solver_json'])) if 'solver_json' in data else {}
    defaults = dict(rho=1e46,sample_planes=21,sample_extent_um=4.,max_iterations=50,random_starts=100)
    overrides = dict(rho=rho,sample_planes=sample_planes,sample_extent_um=sample_extent_um,
                     max_iterations=max_iterations,random_starts=random_starts)
    settings = {key:(overrides[key] if overrides[key] is not None else saved.get(key,value))
                for key,value in defaults.items()}
    rho,sample_planes,sample_extent_um,max_iterations,random_starts = (settings[k] for k in defaults)
    if rho <= 0 or not np.isfinite(rho) or sample_planes < 2 or sample_extent_um <= 0 or not np.isfinite(sample_extent_um):
        raise ValueError('Require finite positive rho, sample extent, and >=2 sample planes.')
    performance = performance_settings(svd_method=svd_method,screening_batch_size=screening_batch_size,
        psf_batch_size=psf_batch_size,derivative_batch_size=derivative_batch_size,
        frequency_batch_size=frequency_batch_size)
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    started = perf_counter()
    write_json(output/'progress.json',dict(status='starting',elapsed_seconds=0))
    try:
        # Explicit CUDA is validated by workspace creation below, inside the
        # failure-reporting block. Resolve auto before choosing backend options.
        if backend == 'auto': backend = reconstruction.get_backend(backend)
        options = {key:performance[key] for key in ('psf_batch_size','derivative_batch_size','frequency_batch_size')}
        options.update(reuse_svd=True,spatial_hessian=True,deduplicate_psfs=True,
                       retain_fields=True,cache_last_fit=True)
        if backend == 'cuda':
            options.update(svd_method=performance['svd_method'],screening_method='linear',
                screening_batch_size=performance['screening_batch_size'],screening_fallback='qr')
        runtime = dict(backend=backend,workspace_options=options,progress_interval_seconds=1.)
        write_json(output/'runtime.json',runtime)
        with np.load(input_path,allow_pickle=False) as data:
            images = data['images']; modes = data['modes']; focus = data['focal_z']
            known = data['diversity_strengths']; truth = data['imposed_for_validation']
            cumulative = data['cumulative_correction'] if 'cumulative_correction' in data else np.zeros_like(truth)
            expected_residual = truth+cumulative
            run_id = str(data['run_id']) if 'run_id' in data else str(Path(input_path).parent.name)
            loop_number = int(data['loop_number']) if 'loop_number' in data else 1
            z_reference = data['z_reference_um'] if 'z_reference_um' in data else np.array([])
            baseline = data['baseline']
            system_ao = data['system_ao_phase_slm'] if 'system_ao_phase_slm' in data else np.array([])
            optics_json = str(data['optics_json']) if 'optics_json' in data else '{}'
            image_orientation = str(data['image_orientation']) if 'image_orientation' in data else 'identity'
            data_origin = str(data['data_origin']) if 'data_origin' in data else 'measured'

            microscope = psf.Microscope(**json.loads(str(data['microscope_json'])))
            grid = acquisition_grid(float(data['field_of_view_um']),images.shape[-2:])
        sample_z = np.linspace(-sample_extent_um/2000,sample_extent_um/2000,sample_planes)
        iteration_times = {}
        last_progress_time = float('-inf')
        last_stage = None
        def progress(event):
            nonlocal last_progress_time,last_stage
            elapsed = perf_counter()-started
            # Hessian batches emit thousands of callbacks. Avoid thousands of
            # small atomic file replacements on the cluster's shared storage.
            stage = event.get('stage')
            if ('strengths' not in event and stage == last_stage
                    and elapsed-last_progress_time < runtime['progress_interval_seconds']):
                return
            if 'strengths' in event:
                iteration_times[event['iteration']] = elapsed
                np.savez_compressed(output/'latest_estimate.npz',modes=modes,
                    estimated_strengths=event['strengths'],iteration=event['iteration'])
            write_json(output/'progress.json',dict(status='running',elapsed_seconds=elapsed,**event))
            last_progress_time,last_stage = elapsed,stage
        print('Execution settings: '+json.dumps(runtime),flush=True)
        workspace = reconstruction.ReconstructionWorkspace(microscope,grid,images,focus,sample_z,modes,rho,
            diversities=[zernike.Aberration(modes,a) for a in known],mode='vector',backend=backend,**options)
        estimate,info = workspace.optimize(max_iterations=max_iterations,num_random_starts=random_starts,
            on_iteration=progress,on_progress=progress,show_initialization_progress=False,verbose=True)
        for state in info.get('history',[]):
            if state['iteration'] in iteration_times:
                state['elapsed_seconds'] = iteration_times[state['iteration']]
        info['runtime'] = runtime
        write_json(output/'optimizer.json',info)
        sample = workspace.estimate_sample(estimate.strengths)
        runtime['svd_counts'] = getattr(workspace,'svd_counts',None)
        write_json(output/'runtime.json',runtime)
        np.savez_compressed(output/'solution.npz',modes=modes,estimated_strengths=estimate.strengths,
            correction_strengths=-np.asarray(estimate.strengths),sample=sample.image_mask,sample_z=sample_z,
            focal_z=focus,imposed_for_validation=truth,image_orientation=image_orientation,data_origin=data_origin,
            cumulative_correction=cumulative,next_cumulative_correction=cumulative-np.asarray(estimate.strengths),
            expected_residual_for_validation=expected_residual,
            coefficient_error_against_expected_residual=np.asarray(estimate.strengths)-expected_residual)
        from scipy.io import savemat
        savemat(output/'correction.mat',dict(schemaVersion=1,modes_mn=modes,
            coefficientUnits='waves',estimatedResidualWaves=np.asarray(estimate.strengths).reshape(-1,1),
            cumulativeCorrectionWaves=(cumulative-np.asarray(estimate.strengths)).reshape(-1,1),
            imposedWaves=truth.reshape(-1,1),baselineWaves=baseline.reshape(-1,1),
            systemAOPhaseSLM=system_ao,
            parentRunId=run_id,parentLoopNumber=loop_number,zReferenceReadoutUm=z_reference,
            optics_json=optics_json,imageOrientation=image_orientation,optimizerSuccess=bool(info['success'])),do_compression=True)
        np.savetxt(output/'coefficients.csv',np.column_stack([modes,truth,estimate.strengths,
            -np.asarray(estimate.strengths),np.asarray(estimate.strengths)-expected_residual]),delimiter=',',
            header='m,n,imposed_waves,estimated_waves,correction_waves,error_against_expected_residual_waves',
            comments='',fmt=['%d','%d','%.12g','%.12g','%.12g','%.12g'])
        summary = dict(status='complete',optimizer_success=bool(info['success']),message=info['message'],
            elapsed_seconds=perf_counter()-started,rho=rho,backend=backend,random_starts=random_starts,
            sample_planes=sample_planes,sample_extent_um=sample_extent_um,
            runtime=runtime,image_orientation=image_orientation,data_origin=data_origin,
            modes=modes,estimated_waves=estimate.strengths,loop_number=loop_number,
            next_cumulative_correction_waves=cumulative-np.asarray(estimate.strengths),
            warning='Estimate includes residual native + fixed system AO phase; imposed coefficients are validation only.')
        write_json(output/'result.json',summary)
        write_json(output/'progress.json',summary)
        return summary
    except Exception as exc:
        write_json(output/'progress.json',dict(status='failed',error=str(exc),elapsed_seconds=perf_counter()-started))
        raise


def preparation_arguments(parser):
    parser.add_argument('run',type=Path,help='Copied MATLAB acquisition directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--exclude-plane',dest='exclude_planes',type=int,action='append',
                        help='Exclude this 1-based plane index from all processing; repeat for multiple planes')
    parser.add_argument('--exclude-image',dest='exclude_images',type=int,action='append',
                        help='Exclude this original 1-based acquisition image number; repeat for multiple images')
    parser.add_argument('--flip-x',action='store_true',help='Mirror measured model x axis (left-right in plots); keep phase commands unchanged')
    parser.add_argument('--min-snr',type=float,default=None,
                        help='Optional minimum bead-peak/background-noise SNR on averaged solver-resolution frames')
    parser.add_argument('--size',type=int,default=None,help='Metadata default; block-average to this size; must divide acquisition size')
    parser.add_argument('--pupil-size',type=int,default=None)
    parser.add_argument('--background',type=float,default=None,help='Metadata default; known PMT pedestal to subtract; no clipping')
    parser.add_argument('--background-border-pixels',type=int,default=None,
                        help='Subtract each image outer-border median at final binned resolution; preserve signed values')
    parser.add_argument('--z-sign',type=int,choices=[-1,1],default=None,help='Metadata default; map stage offsets to model focal positions')


def solver_arguments(parser):
    parser.add_argument('--rho',type=float,default=None)
    parser.add_argument('--sample-planes',type=int,default=None)
    parser.add_argument('--sample-extent-um',type=float,default=None)
    parser.add_argument('--max-iterations',type=int,default=None)
    parser.add_argument('--random-starts',type=int,default=None)
    parser.add_argument('--svd-method',choices=['qr','direct'],default=None,
                        help='CUDA factorization (default: qr, with numerical fallback)')
    for name,default in [('screening-batch-size',8),('psf-batch-size',8),
                         ('derivative-batch-size',8),('frequency-batch-size',4096)]:
        parser.add_argument('--'+name,type=int,default=None,
                            help=f'Execution batch size (default: {default})')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare'); preparation_arguments(prep)
    compare = sub.add_parser('compare'); preparation_arguments(compare)
    compare.add_argument('--bead-center-um',type=float,nargs=3,default=None,metavar=('X','Y','Z'))
    compare.add_argument('--bead-planes',type=int,default=None)
    compare.add_argument('--no-tex',action='store_true',help='Disable LaTeX explicitly')
    solve = sub.add_parser('solve'); solver_arguments(solve)
    solve.add_argument('input',type=Path); solve.add_argument('--output',type=Path,required=True)
    solve.add_argument('--backend',choices=['cpu','cuda','auto'],default='cuda')
    args = vars(parser.parse_args()); command = args.pop('command')
    if command == 'compare':
        args['use_tex'] = False if args.pop('no_tex') else None
        print(compare_run(**args))
    elif command == 'prepare': print(prepare_run(**args))
    else:
        args['input_path'] = args.pop('input'); solve_input(**args)


if __name__ == '__main__': main()
