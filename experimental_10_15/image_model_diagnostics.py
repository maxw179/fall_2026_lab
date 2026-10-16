"""Local, read-only checks of the many-bead image-formation assumptions.

Run with ``python -m experimental_10_15.image_model_diagnostics``. Results are
checkpointed by model variant. No registration, acquisition edits or cluster
submission is performed. These are sensitivity tests, not calibration estimates.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import fft, ndimage, optimize, stats
from scipy.io import loadmat
from experimental_10_15.bead_pd import acquisition_grid
from experimental_10_15.bead_pd_io import load_acquisition, block_average, write_json
from experimental_10_15.empirical_noise import background_noise
from experimental_10_15.plot_many_bead_images import SOURCE
from experimental_10_15.reference_bead_simulation import FLAGGED
from optimization import reconstruction
from utils import psf, zernike

ROOT=Path(__file__).resolve().parent
CASE=ROOT/'multiple_beads_64x64_flipx_artifacts_excluded_20261015/multiple_beads_64x64'
SIM=ROOT/'multiple_beads_64x64_simulated_noisy_matched_20261015/multiple_beads_64x64'
DEFAULT=ROOT/'image_model_diagnostics'


def csv_rows(path, rows):
    if not rows:return
    with Path(path).open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def robust_affine(x,y):
    """Estimate y=g*x+b without allowing negative fluorescence gain."""
    x=np.asarray(x).ravel();y=np.asarray(y).ravel()
    scale=max(float(np.std(x)),1.)
    design=np.c_[x/scale,np.ones(len(x))]
    coefficients=np.linalg.lstsq(design,y,rcond=None)[0]
    for _ in range(8):
        residual=y-design@coefficients
        sigma=max(float(np.median(abs(residual-np.median(residual)))/.67448975),1.)
        w=np.minimum(1.,2.5*sigma/np.maximum(abs(residual),1e-12))
        coefficients=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
    return max(float(coefficients[0]/scale),0.),float(coefficients[1])


def correlation(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    a=a-a.mean();b=b-b.mean()
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator else 0.


def load_case(folder=CASE):
    with np.load(folder/'submission/input.npz') as data:
        case={k:data[k].copy() for k in data.files}
    with np.load(folder/'solve/results/solution.npz') as solved:
        case['estimate']=solved['estimated_strengths'].copy()
        case['sample_z']=solved['sample_z'].copy()
    case['params']=json.loads(str(case['microscope_json']))
    case['solver']=json.loads(str(case['solver_json']))
    # Submission CLI overrides (notably rho) supersede acquisition defaults.
    result_path=folder/'solve/results/result.json'
    if result_path.is_file():
        effective=json.loads(result_path.read_text())
        for key in ['rho','sample_planes','sample_extent_um','random_starts']:
            if key in effective:case['solver'][key]=effective[key]
    return case


def workspace(case, *, images=None, known=None, focus=None, params=None, fov_scale=1.,
              y_scale=1., rho_scale=1., sample_z=None, mode='vector'):
    images=case['images'] if images is None else images
    known=case['diversity_strengths'] if known is None else known
    focus=case['focal_z'] if focus is None else focus
    grid=acquisition_grid(float(case['field_of_view_um'])*fov_scale,images.shape[-2:])
    grid.L_ffp_y*=y_scale
    return reconstruction.ReconstructionWorkspace(psf.Microscope(**(case['params'] if params is None else params)),
        grid,images,focus,case['sample_z'] if sample_z is None else sample_z,case['modes'],
        case['solver']['rho']*rho_scale,
        diversities=[zernike.Aberration(case['modes'],v) for v in known],mode=mode,backend='cpu',
        psf_batch_size=8,frequency_batch_size=512,reuse_svd=False,retain_fields=False,
        cache_last_fit=False,deduplicate_psfs=True,propagation='zoom')


def fit_diagnostics(ws, strengths, images):
    # Fourier residual is D-SF, including the padded exterior in the objective.
    fit=ws._get_fit(np.asarray(strengths))[0]
    residual=fft.irfft2(fit.residual.reshape(len(images),ws.shape[0],ws.shape[1]//2+1),s=ws.shape)
    sample=fft.irfft2(fit.estimated.reshape(len(ws.sample_z),ws.shape[0],ws.shape[1]//2+1),s=ws.shape)
    interior=residual[ws.crop]
    sse=float(np.sum(residual**2));regularizer=fit.loss/np.prod(ws.shape)-sse
    np.testing.assert_allclose(sse+ws.rho*np.sum(sample**2),fit.loss/np.prod(ws.shape),rtol=1e-9)
    return dict(loss=fit.loss,measured_rmse=float(np.sqrt(np.mean(interior**2))),
                padded_sse=sse,regularizer=regularizer,
                negative_object_fraction=float(np.mean(sample<0)),
                negative_object_mass_fraction=float(np.sum(abs(sample[sample<0]))/max(np.sum(abs(sample)),1e-300))),interior,sample


def audit(output):
    acq=load_acquisition(SOURCE);images=block_average(acq.images,64)[:,::-1,:]
    b,sigma=background_noise(images)
    meta=loadmat(SOURCE/'metadata.mat',simplify_cells=True)['metadata']
    log=loadmat(SOURCE/'acquisition.mat',simplify_cells=True)['log']
    times=np.array([datetime.strptime(v['finishedAt'],'%Y%m%dT%H%M%S').timestamp() for v in log])
    elapsed=(times-times[0])/60
    fit=json.loads((ROOT/'reference_bead_orientation_scan/flip_x/reference_fit.json').read_text())
    centers=np.asarray(fit['beads'])[:,:2]
    x,y=acquisition_grid(acq.optics['fieldOfViewUm'],(64,64)).get_xy();x=x*1000;y=y*1000
    aperture=np.array([(x[:,None]-xc)**2+(y[None,:]-yc)**2<.65**2 for xc,yc in centers])
    signal=images-b[:,None,None]
    flux=np.array([signal[:,mask].sum(axis=1) for mask in aperture]).T
    centroids=[]
    for mask in aperture:
        weights=np.maximum(ndimage.gaussian_filter(signal,(0,.7,.7)),0)*mask
        norm=np.maximum(weights.sum(axis=(1,2)),1e-12)
        centroids.append(np.c_[(weights*x[None,:,None]).sum(axis=(1,2))/norm,
                              (weights*y[None,None,:]).sum(axis=(1,2))/norm])
    centroids=np.stack(centroids,axis=1)
    rows=[];pairs=[]
    usable=~np.isin(np.arange(len(images))+1,FLAGGED)
    for i in range(len(images)):
        row=dict(image=i+1,name=str(acq.names[i]),z_um=acq.focal_z[i]*1000,minutes=float(elapsed[i]),
                 excluded=not bool(usable[i]),reference=not bool(acq.is_pd[i]),background=b[i],sigma=sigma[i],
                 signed_min=float(images[i].min()),peak=float(images[i].max()),
                 negative_fraction=float(np.mean(images[i]<0)))
        for bead in range(len(centers)):
            row.update({f'bead_{bead+1}_aperture_sum':flux[i,bead],
                        f'bead_{bead+1}_x_um':centroids[i,bead,0],f'bead_{bead+1}_y_um':centroids[i,bead,1]})
        rows.append(row)
    for plane in np.unique(acq.plane_index):
        for before,after in [('reference_before','reference_after'),('aberrated_zero','aberrated_zero_after')]:
            i=int(np.flatnonzero((acq.plane_index==plane)&(acq.names==before))[0])
            j=int(np.flatnonzero((acq.plane_index==plane)&(acq.names==after))[0])
            # Both repeats are noisy. Smooth both before regression to reduce
            # errors-in-variables attenuation; aperture ratio remains primary.
            gain,offset=robust_affine(ndimage.gaussian_filter(signal[i],1.),ndimage.gaussian_filter(signal[j],1.))
            residual=signal[j]-gain*signal[i]-offset
            # Position differences are reference centroids only; no registration applied.
            motion=np.linalg.norm(centroids[j]-centroids[i],axis=1)
            pairs.append(dict(first=i+1,second=j+1,reference=before.startswith('reference'),z_um=acq.focal_z[i]*1000,
                excluded=not bool(usable[i]&usable[j]),elapsed_minutes=float(elapsed[j]-elapsed[i]),
                robust_gain_after_over_before=gain,offset_after_background=offset,
                total_aperture_ratio=float(flux[j].sum()/flux[i].sum()) if flux[i].sum()>0 else np.nan,
                raw_difference_rmse=float(np.sqrt(np.mean((images[j]-images[i])**2))),
                affine_difference_rmse=float(np.sqrt(np.mean(residual**2))),
                median_centroid_displacement_um=float(np.median(motion)),max_centroid_displacement_um=float(motion.max()),
                residual_neighbor_x_correlation=correlation(residual[:-1],residual[1:]),
                residual_neighbor_y_correlation=correlation(residual[:,:-1],residual[:,1:])))
    csv_rows(output/'frames.csv',rows);csv_rows(output/'repeats.csv',pairs)
    np.savez_compressed(output/'reference_measurements.npz',images=images,background=b,sigma=sigma,flux=flux,
                        centroids=centroids,elapsed_minutes=elapsed)
    # Independently reproduce the MATLAB polynomial, rather than calling the same Python evaluator.
    radius=np.linspace(0,1,101);phi=np.linspace(0,2*np.pi,101);r,p=np.meshgrid(radius,phi)
    errors=[]
    for m,n in acq.modes:
        a=abs(int(m));radial=np.zeros_like(r)
        for s in range((int(n)-a)//2+1):
            c=(-1)**s*math.factorial(int(n)-s)/(math.factorial(s)*math.factorial((int(n)+a)//2-s)*math.factorial((int(n)-a)//2-s))
            radial+=c*r**(int(n)-2*s)
        value=radial*(np.sqrt(n+1) if m==0 else np.sqrt(2*(n+1))*(np.cos(a*p) if m>0 else np.sin(a*p)))
        errors.append(float(np.max(abs(value-zernike.zernike_mode(m,n,r,p)))))
    controls=[]
    for source in sorted(ROOT.glob('3d_fluorobeads_*/tp*/metadata.mat')):
        if not (source.parent/'acquisition.mat').is_file():continue
        try:
            control=load_acquisition(source.parent)
            if control.acquired.sum()<2:continue
            bi,si=background_noise(block_average(control.images[control.acquired],64))
            controls.append(dict(source=str(source.parent),run=source.parent.parent.name,
                acquired=int(control.acquired.sum()),same_system_ao=bool(np.array_equal(control.system_ao,acq.system_ao)),
                zero_imposed=bool(np.allclose(control.imposed,0)),sigma_median=float(np.median(si)),
                background_median=float(np.median(bi))))
        except (ValueError,FileNotFoundError):continue
    csv_rows(output/'acquisition_inventory.csv',controls)
    report=dict(source=SOURCE,image_count=len(images),pd_used=int(np.sum(acq.is_pd&usable)),
        duration_minutes=float(elapsed[-1]),frames_per_saved_image=acq.config['framesPerImage'],
        exposures_available=False,z_position_source=meta['zPositionSource'],
        native_system_aberration=meta['nativeSystemAberration'],
        matlab_python_polynomial_max_error=max(errors),wave_units='waves; exp(2*pi*i*waves); raw 256 units per cycle',
        raw_sha256=hashlib.sha256((SOURCE/'acquisition.mat').read_bytes()).hexdigest(),
        reference_centroid_warning='Fixed 0.65 um apertures and positive smoothed weights. Defocus/background bias centroid estimates; these are motion indicators, not measured stage displacements.',
        repeat_pairs=pairs,controls=controls)
    write_json(output/'audit.json',report)
    return report


def sweeps(output):
    case=load_case();acq=load_acquisition(SOURCE)
    ids=case['source_indices'];bg,sigma=background_noise(case['images'])
    truth=case['imposed_for_validation']+case['cumulative_correction'];estimate=case['estimate']
    variants=[('nominal',{})]
    for scale in [.75,.9,1.1,1.25,-1.]:variants.append((f'z_scale_{scale:g}',dict(focus=case['focal_z']*scale)))
    for offset in [-.3,.3]:variants.append((f'z_offset_{offset:g}',dict(focus=case['focal_z']+offset/1000)))
    for scale in [.9,1.1]:variants.append((f'fov_scale_{scale:g}',dict(fov_scale=scale)))
    for scale in [.9,1.1]:variants.append((f'y_scale_{scale:g}',dict(y_scale=scale)))
    for na in [.9,1.,1.1]:variants.append((f'NA_{na:g}',dict(params={**case['params'],'num_apt':na})))
    for scale in [.5,2.]:variants.append((f'beam_waist_{scale:g}',dict(params={**case['params'],'w_0':case['params']['w_0']*scale})))
    variants.append(('scalar',dict(mode='scalar')))
    for scale in [.01,100.]:variants.append((f'rho_scale_{scale:g}',dict(rho_scale=scale)))
    for planes in [11,41]:variants.append((f'object_planes_{planes}',dict(sample_z=np.linspace(-.002,.002,planes))))
    variants.append(('object_extent_6um',dict(sample_z=np.linspace(-.003,.003,31))))
    variants.extend([('subtract_border_background',dict(images=case['images']-bg[:,None,None])),
                     ('subtract_global_background',dict(images=case['images']-np.median(bg)))])
    # A row/column additive baseline from the bead-free outer border only.
    im=case['images'];w=10
    row=np.median(np.concatenate([im[:,:,:w],im[:,:,-w:]],axis=2),axis=2)
    col=np.median(np.concatenate([im[:,:w,:],im[:,-w:,:]],axis=1),axis=1)
    variants.append(('subtract_border_row_column',dict(images=im-row[:,:,None]-col[:,None,:]+bg[:,None,None])))
    # Repeat-calibrated gain: interpolate reference before/after ratios over image order within each plane.
    all_im=block_average(acq.images,64)[:,::-1,:];all_bg,_=background_noise(all_im)
    with np.load(output/'reference_measurements.npz') as data:reference_flux=data['flux'].sum(axis=1)
    gains=np.ones(len(acq.images))
    for plane in np.unique(acq.plane_index):
        before=int(np.flatnonzero((acq.plane_index==plane)&(acq.names=='reference_before'))[0])
        after=int(np.flatnonzero((acq.plane_index==plane)&(acq.names=='reference_after'))[0])
        gain=reference_flux[after]/reference_flux[before]
        gains[before:after+1]=np.exp(np.linspace(0,np.log(max(gain,.05)),after-before+1))
    variants.append(('reference_aperture_repeat_gain',dict(images=(im-bg[:,None,None])/gains[ids,None,None])))
    # All candidate strengths here are fixed, not newly optimized for each altered model.
    rows=[];cache=output/'sweeps';cache.mkdir(exist_ok=True)
    for name,options in variants:
        path=cache/(name+'.json')
        if path.exists():row=json.loads(path.read_text())
        else:
            print('Sensitivity:',name,flush=True)
            ws=workspace(case,**options)
            images=options.get('images',im)
            metrics=[]
            for label,strengths in [('truth',truth),('estimate',estimate)]:
                metric,residual,sample=fit_diagnostics(ws,strengths,images);metric['candidate']=label;metrics.append(metric)
                if name=='nominal':
                    np.savez_compressed(output/(label+'_fit.npz'),residual=residual,predicted=images-residual,sample=sample)
            row=dict(variant=name,truth_loss=metrics[0]['loss'],estimate_loss=metrics[1]['loss'],
                     estimate_over_truth=metrics[1]['loss']/metrics[0]['loss'],
                     truth_rmse=metrics[0]['measured_rmse'],estimate_rmse=metrics[1]['measured_rmse'],metrics=metrics)
            write_json(path,row)
            del ws
        rows.append(row)
        csv_rows(output/'model_sweeps.csv',[{k:v for k,v in row.items() if k!='metrics'} for row in rows])
    # Phase tests use unchanged raw signed data, diversities scaled separately from imposed phase.
    phase=[]
    for probe_scale in [-1.,.5,.75,1.,1.25,1.5]:
        path=cache/(f'phase_{probe_scale:g}.json')
        if path.exists():record=json.loads(path.read_text())
        else:
            print('Phase probe scale:',probe_scale,flush=True)
            ws=workspace(case,known=case['diversity_strengths']*probe_scale)
            # Includes a simultaneous reversal of imposed and probe phase.
            candidates=np.array([case['cumulative_correction']+a*case['imposed_for_validation'] for a in [-1.,0.,.5,.75,1.,1.25,1.5]])
            losses=ws.screen_losses(candidates)
            record=dict(probe_scale=probe_scale,imposed_scales=[-1.,0.,.5,.75,1.,1.25,1.5],losses=losses,
                        estimate_loss=float(ws.loss(estimate)))
            write_json(path,record);del ws
        phase.append(record)
    write_json(output/'phase_scales.json',phase)
    # Subset sensitivity, not independent validation of the all-data learned estimate.
    subsets=[]
    for name,keep in [('middle_Z',abs(case['focal_z'])<=.001000001),
                      ('outer_Z',abs(case['focal_z'])>.001000001),
                      ('positive_probes',case['diversity_strengths'].sum(axis=1)>0),
                      ('negative_probes',case['diversity_strengths'].sum(axis=1)<0),
                      ('zero_probe',np.all(case['diversity_strengths']==0,axis=1))]:
        path=cache/(name+'.json')
        if path.exists():row=json.loads(path.read_text())
        else:
            print('Subset:',name,flush=True)
            ws=workspace(case,images=im[keep],known=case['diversity_strengths'][keep],focus=case['focal_z'][keep])
            losses=[float(ws.loss(c)) for c in [truth,estimate]]
            row=dict(subset=name,images=int(keep.sum()),truth_loss=losses[0],estimate_loss=losses[1],estimate_over_truth=losses[1]/losses[0])
            write_json(path,row);del ws
        subsets.append(row)
    csv_rows(output/'subsets.csv',subsets)


def phase_transform(coefficients,modes,angle=0.,reflect=False):
    """Evaluate phase at phi+angle (and optionally -phi), in the same basis."""
    coefficients=np.asarray(coefficients);out=coefficients.copy()
    lookup={tuple(mode):i for i,mode in enumerate(modes)}
    for i,(m,n) in enumerate(modes):
        if m<=0:continue
        j=lookup[(-m,n)];c=np.cos(m*angle);s=np.sin(m*angle)
        out[...,i]=coefficients[...,i]*c+coefficients[...,j]*s
        out[...,j]=(-coefficients[...,i]*s+coefficients[...,j]*c)*(-1 if reflect else 1)
    return out


def extra_checks(output):
    case=load_case();truth=case['imposed_for_validation']+case['cumulative_correction']
    cache=output/'sweeps';cache.mkdir(exist_ok=True)
    # Rotate/reflect every SLM mode, including the known diversities. Camera
    # images and the laboratory polarization stay fixed.
    orientation=[]
    for reflect in [False,True]:
        for degrees in ([0,90,180,270,-10,10] if not reflect else [0,90,180,270]):
            name=f'pupil_angle_{degrees}_reflect_{int(reflect)}';path=cache/(name+'.json')
            if path.exists():row=json.loads(path.read_text())
            else:
                print('Pupil axes:',name,flush=True)
                angle=np.deg2rad(degrees)
                known=phase_transform(case['diversity_strengths'],case['modes'],angle,reflect)
                ws=workspace(case,known=known)
                loss=float(ws.loss(phase_transform(truth,case['modes'],angle,reflect)))
                row=dict(name=name,angle_degrees=degrees,reflect=reflect,truth_loss=loss)
                write_json(path,row);del ws
            orientation.append(row)
    csv_rows(output/'pupil_axes.csv',orientation)
    # Test existing zero-control coefficient vectors as fixed common offsets.
    # Their original camera convention was not mirrored, so report both parity
    # choices explicitly; neither is an independently calibrated wavefront.
    ws=workspace(case);rows=[]
    for tag in ['system_AO','no_aberration']:
        folder=ROOT/f'binning_all_images_20261015_161419/3d_fluorobeads_{tag}_64x64'
        if not (folder/'solve/results/solution.npz').is_file():continue
        control=load_case(folder)
        lookup={tuple(m):i for i,m in enumerate(control['modes'])}
        offset=(control['estimate']-control['cumulative_correction'])[[lookup[tuple(m)] for m in case['modes']]]
        same=bool(np.array_equal(case['system_ao_phase_slm'],control['system_ao_phase_slm']))
        for parity in ['original','x_mirrored']:
            vector=offset if parity=='original' else phase_transform(offset,case['modes'],np.pi,True)
            for sign in [-1,1]:
                name=f'control_{tag}_{parity}_{sign}';path=cache/(name+'.json')
                if path.exists():row=json.loads(path.read_text())
                else:
                    print('Control offset:',name,flush=True)
                    row=dict(control=tag,same_system_ao=same,parity=parity,sign=sign,
                             loss=float(ws.loss(truth+sign*vector)),offset_waves=vector.tolist())
                    write_json(path,row)
                rows.append(row)
    csv_rows(output/'control_offsets.csv',[{k:v for k,v in row.items() if k!='offset_waves'} for row in rows])


def forward_checks(output):
    """Reference-constrained sphere residuals: photometry and blur upper bounds."""
    acq=load_acquisition(SOURCE)
    with np.load(ROOT/'reference_bead_orientation_scan/flip_x/comparison.npz') as data:
        measured=data['measured'];predicted=data['simulated'];ids=data['source_indices']
    noise=json.loads((ROOT/'multiple_beads_64x64_simulated_noisy_matched_20261015/simulated_acquisition/noise_calibration.json').read_text())
    fit=json.loads((ROOT/'reference_bead_orientation_scan/flip_x/reference_fit.json').read_text())
    bg=fit['background'];gain_noise=noise['effective_poisson_gain']
    sigma=np.asarray(noise['gaussian_sigma_by_image'])[ids]
    rows=[]
    for row,i in enumerate(ids):
        gain,offset=robust_affine(predicted[row],measured[row])
        residual=measured[row]-predicted[row]
        adjusted=measured[row]-gain*predicted[row]-offset
        variance=gain_noise*np.maximum(predicted[row]-bg,0)+sigma[row]**2
        rows.append(dict(image=i+1,name=str(acq.names[i]),z_um=acq.focal_z[i]*1000,
            reference=not bool(acq.is_pd[i]),excluded=i+1 in FLAGGED,
            fixed_sphere_rmse=float(np.sqrt(np.mean(residual**2))),
            affine_sphere_rmse=float(np.sqrt(np.mean(adjusted**2))),
            fitted_gain=gain,fitted_offset=offset,
            noise_expected_rmse=float(np.sqrt(np.mean(variance))),
            standardized_residual_rms=float(np.sqrt(np.mean(residual**2/variance))),
            neighbor_x_correlation=correlation(residual[:-1],residual[1:]),
            neighbor_y_correlation=correlation(residual[:,:-1],residual[:,1:])))
    csv_rows(output/'sphere_residuals.csv',rows)
    keep=acq.is_pd[ids]&~np.isin(ids+1,FLAGGED)
    blur=[]
    for sx,sy in [(0,0),(.5,0),(0,.5),(.5,.5),(1,0),(0,1),(1,1)]:
        blurred=ndimage.gaussian_filter(predicted,(0,sx,sy))
        losses=[]
        for i in np.flatnonzero(keep):
            # Deliberately generous nuisance fit. This is an explanatory upper
            # bound, not a calibrated detector blur or a phase retrieval run.
            design=np.c_[blurred[i].ravel(),np.ones(blurred[i].size)]
            c=np.linalg.lstsq(design,measured[i].ravel(),rcond=None)[0]
            c[0]=max(c[0],0)
            losses.append(np.mean((measured[i]-c[0]*blurred[i]-c[1])**2))
        blur.append(dict(sigma_x_pixels=sx,sigma_y_pixels=sy,affine_rmse=float(np.sqrt(np.mean(losses)))))
    csv_rows(output/'sphere_blur.csv',blur)
    write_json(output/'forward_checks.json',dict(photometry='Robust positive gain and offset per frame, evaluated on the same frame; explanatory upper bound only.',
        blur_method='Gaussian blur in final 64x64 pixels applied to sphere predictions, per-image unconstrained offset/positive gain refit; no measured images altered.',
        standardized_residual='Approximate repeat-calibrated variance, not an independent chi-square or an absolute photon calibration.',
        rows=rows,blur=blur))


def background_checks(output):
    case=load_case();sim=load_case(SIM)
    np.testing.assert_array_equal(case['source_indices'],sim['source_indices'])
    np.testing.assert_array_equal(case['diversity_strengths'],sim['diversity_strengths'])
    im=case['images'];bg,sigma=background_noise(im);sim_bg,_=background_noise(sim['images'])
    truth=case['imposed_for_validation']+case['cumulative_correction']
    cache=output/'background_checks';cache.mkdir(exist_ok=True)
    # Independent border geometries test contamination by bead tails.
    border_estimates={}
    for width in [5,10,16]:
        mask=np.ones(im.shape[-2:],bool);mask[width:-width,width:-width]=False
        border_estimates[f'border_{width}']=np.median(im[:,mask],axis=1)
    corners=np.zeros(im.shape[-2:],bool)
    for xs in [slice(0,10),slice(-10,None)]:
        for ys in [slice(0,10),slice(-10,None)]:corners[xs,ys]=True
    border_estimates['corners']=np.median(im[:,corners],axis=1)
    acq=load_acquisition(SOURCE)
    all_im=block_average(acq.images,64)[:,::-1,:];all_bg,_=background_noise(all_im)
    reference_baseline=np.zeros(len(all_im))
    for plane in np.unique(acq.plane_index):
        before=int(np.flatnonzero((acq.plane_index==plane)&(acq.names=='reference_before'))[0])
        after=int(np.flatnonzero((acq.plane_index==plane)&(acq.names=='reference_after'))[0])
        reference_baseline[before:after+1]=np.linspace(all_bg[before],all_bg[after],after-before+1)
    variants=[(name,case,im-values[:,None,None],{}) for name,values in border_estimates.items()]
    variants.extend([
        ('reference_interpolated_background',case,im-reference_baseline[case['source_indices'],None,None],{}),
        ('simulation_original',sim,sim['images'],{}),
        ('simulation_border_subtracted',sim,sim['images']-sim_bg[:,None,None],{}),
        ('simulation_experimental_background',sim,sim['images']+(bg-sim_bg)[:,None,None],{}),
    ])
    # Repeat physical sensitivity checks after removing the dominant baseline.
    corrected=im-bg[:,None,None]
    settings=[('zscale_0.9',dict(focus=case['focal_z']*.9)),('zscale_1.1',dict(focus=case['focal_z']*1.1)),
              ('z_reversed',dict(focus=-case['focal_z'])),
              ('NA_1',dict(params={**case['params'],'num_apt':1.})),
              ('NA_1.1',dict(params={**case['params'],'num_apt':1.1})),
              ('fov_0.9',dict(fov_scale=.9)),('fov_1.1',dict(fov_scale=1.1)),
              ('beam_0.5',dict(params={**case['params'],'w_0':case['params']['w_0']*.5})),
              ('beam_2',dict(params={**case['params'],'w_0':case['params']['w_0']*2})),
              ('object_41',dict(sample_z=np.linspace(-.002,.002,41)))]
    variants.extend([('corrected_'+name,case,corrected,options) for name,options in settings])
    rows=[]
    for name,source,images,options in variants:
        path=cache/(name+'.json')
        if path.exists():row=json.loads(path.read_text())
        else:
            print('Background validation:',name,flush=True)
            ws=workspace(source,images=images,**options)
            metrics=[]
            candidates=[('truth',truth),('experimental_estimate',case['estimate'])]
            if name.startswith('simulation_'):candidates.append(('simulation_estimate',sim['estimate']))
            for label,coeff in candidates:
                metric,residual,sample=fit_diagnostics(ws,coeff,images);metric['candidate']=label;metrics.append(metric)
                if name in ['border_10','simulation_border_subtracted','simulation_experimental_background']:
                    np.savez_compressed(cache/(name+'_'+label+'.npz'),residual=residual,sample=sample,predicted=images-residual)
            row=dict(variant=name,truth_loss=metrics[0]['loss'],experimental_estimate_loss=metrics[1]['loss'],
                     experimental_estimate_over_truth=metrics[1]['loss']/metrics[0]['loss'],
                     truth_rmse=metrics[0]['measured_rmse'],experimental_estimate_rmse=metrics[1]['measured_rmse'],metrics=metrics)
            write_json(path,row);del ws
        rows.append(row)
        csv_rows(output/'background_validation.csv',[{k:v for k,v in row.items() if k!='metrics'} for row in rows])
    write_json(output/'background_estimators.json',dict(estimates=border_estimates,nominal=bg,
        disagreement_rmse={k:float(np.sqrt(np.mean((v-bg)**2))) for k,v in border_estimates.items()},
        model='Imagewise spatial constant estimated from bead-free borders. Raw images remain unchanged.',
        simulation_control='Frozen noisy simulation plus experimental-minus-simulation border median per image; wavefront/object/noise realization unchanged.'))
    # One short scaling curve on the corrected data, with the object refit each point.
    path=cache/'corrected_phase_scales.json'
    if not path.exists():
        rows=[]
        for scale in [.75,1.,1.25,-1.]:
            print('Corrected probe scale:',scale,flush=True)
            ws=workspace(case,images=corrected,known=case['diversity_strengths']*scale)
            fractions=np.array([-.0,.5,.75,1.,1.25,1.5])
            losses=ws.screen_losses(case['cumulative_correction']+fractions[:,None]*case['imposed_for_validation'])
            rows.append(dict(probe_scale=scale,fractions=fractions,losses=losses))
            del ws
        write_json(path,rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['all','audit','sweeps','extra','forward','background','report'],default='all')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.stage in ('all','audit'):audit(args.output)
    if args.stage in ('all','sweeps'):sweeps(args.output)
    if args.stage in ('all','extra'):extra_checks(args.output)
    if args.stage in ('all','forward'):forward_checks(args.output)
    if args.stage in ('all','background'):background_checks(args.output)
    if args.stage in ('all','report'):
        from experimental_10_15.image_model_report import report
        report(args.output)


if __name__=='__main__':main()
