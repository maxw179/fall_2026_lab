"""Compare old/new experimental estimates with truth, locally and read-only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import fft
from scipy.optimize import brentq
from experimental_10_15.image_model_diagnostics import (
    ROOT, CASE, SIM, load_case, workspace, fit_diagnostics, csv_rows, correlation)
from experimental_10_15.bead_pd_io import load_acquisition,block_average,subtract_border_background,write_json
from experimental_10_15.plot_many_bead_images import SOURCE
from experimental_10_15.reference_bead_simulation import FLAGGED
from utils import psf,zernike

NEW=ROOT/'multiple_beads_64x64_flipx_background_subtracted_20261015/multiple_beads_64x64'
OUTPUT=ROOT/'background_solution_comparison'


def compare(output=OUTPUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    old=load_case(CASE);new=load_case(NEW);sim=load_case(SIM)
    keys=['source_indices','modes','focal_z','diversity_strengths','sample_z','imposed_for_validation','cumulative_correction','microscope_json']
    for key in keys:np.testing.assert_array_equal(old[key],new[key])
    expected,offsets=subtract_border_background(old['images'],10)
    np.testing.assert_array_equal(new['images'],expected)
    truth=new['imposed_for_validation']+new['cumulative_correction']
    candidates={'Truth':truth,'Old estimate':old['estimate'],'New estimate':new['estimate']}
    alpha=psf.Microscope(**new['params']).alpha
    rms={label:float(zernike.zernike_RMS(zernike.Aberration(new['modes'],v-truth),alpha)) for label,v in candidates.items()}
    np.savez_compressed(output/'coefficients.npz',modes=new['modes'],truth=truth,old=old['estimate'],new=new['estimate'])
    csv_rows(output/'coefficients.csv',[dict(m=int(m),n=int(n),truth=float(t),old=float(o),new=float(v),
                 old_error=float(o-t),new_error=float(v-t)) for (m,n),t,o,v in zip(new['modes'],truth,old['estimate'],new['estimate'])])
    # Paired comparisons always use the same data and object regularization.
    acq=load_acquisition(SOURCE);all_images=block_average(acq.images,64)[:,::-1,:]
    corrected_all,all_backgrounds=subtract_border_background(all_images,10)
    reference_ids=np.flatnonzero(acq.acquired&~acq.is_pd)
    ref={**new,'images':corrected_all[reference_ids],'focal_z':acq.focal_z[reference_ids],
         'diversity_strengths':acq.diversities[reference_ids]+acq.baseline-truth}
    # Candidate includes imposed+cumulative. Remove these from the known
    # reference command to model only the inferred common system residual there.
    joint_ids=np.flatnonzero(acq.acquired&~np.isin(np.arange(len(all_images))+1,FLAGGED))
    known=acq.diversities[joint_ids]+acq.baseline-(~acq.is_pd[joint_ids])[:,None]*truth
    joint={**new,'images':corrected_all[joint_ids],'focal_z':acq.focal_z[joint_ids],'diversity_strengths':known}
    cases=[('raw_PD',old,{}),('background_PD',new,{}),
           ('background_PD_and_references',joint,{}),
           ('noisy_simulation',sim,dict(images=subtract_border_background(sim['images'],10)[0]))]
    for width in [5,16]:cases.append((f'border_{width}',new,dict(images=subtract_border_background(old['images'],width)[0])))
    rows=[];reference_rows=[]
    for name,case,options in cases:
        cache=output/(name+'.json')
        if cache.exists():rows.extend(json.loads(cache.read_text()));continue
        print('Evaluating',name,flush=True)
        images=options.get('images',case['images']);ws=workspace(case,**options)
        metrics=[]
        for label,coefficients in candidates.items():
            metric,residual,sample=fit_diagnostics(ws,coefficients,images)
            metric.update(dataset=name,candidate=label,rms_error_waves=rms[label])
            metrics.append(metric)
            if name in ['raw_PD','background_PD','noisy_simulation','background_PD_and_references']:
                np.savez_compressed(output/(name+'_'+label.replace(' ','_')+'.npz'),residual=residual,
                                    predicted=images-residual,sample=sample)
            if name=='background_PD':
                # Held-out reference prediction: object fitted to PD only;
                # reference images did not enter either saved aberration solve.
                refws=workspace(ref)
                otfs=refws._build_otfs(coefficients)[0]
                sample_ft=fft.rfft2(sample).reshape(len(new['sample_z']),-1)
                predicted=fft.irfft2(np.einsum('kpf,pf->kf',otfs,sample_ft).reshape(
                    len(reference_ids),refws.shape[0],refws.shape[1]//2+1),s=refws.shape)[refws.crop]
                residual_ref=ref['images']-predicted
                np.savez_compressed(output/('reference_prediction_'+label.replace(' ','_')+'.npz'),
                                    predicted=predicted,residual=residual_ref,source_indices=reference_ids)
                for k,i in enumerate(reference_ids):
                    p=predicted[k];d=ref['images'][k]
                    gain=max(float(np.sum(p*d)/max(np.sum(p*p),1e-300)),0.)
                    reference_rows.append(dict(candidate=label,image=int(i+1),z_um=float(acq.focal_z[i]*1000),
                        rmse=float(np.sqrt(np.mean((d-p)**2))),correlation=correlation(d,p),
                        fitted_gain=gain,gain_adjusted_rmse=float(np.sqrt(np.mean((d-gain*p)**2)))))
                del refws
        # Canonical objective must replay the two saved optimizer endpoints.
        if name in ['raw_PD','background_PD']:
            source=CASE if name=='raw_PD' else NEW
            label='Old estimate' if name=='raw_PD' else 'New estimate'
            recorded=json.loads((source/'solve/results/optimizer.json').read_text())['history'][-1]['loss']
            actual=next(m['loss'] for m in metrics if m['candidate']==label)
            np.testing.assert_allclose(actual,recorded,rtol=1e-10)
        write_json(cache,metrics);rows.extend(metrics)
        if reference_rows:csv_rows(output/'heldout_references.csv',reference_rows)
        del ws
    csv_rows(output/'loss_comparison.csv',rows)
    reference_model_checks(output,new,sim,candidates,truth,reference_ids,joint_ids,known,corrected_all)
    photometric_checks(output,new,sim,candidates)
    # Local paths distinguish model preference from a separate optimizer basin.
    curves=[];ws=workspace(new)
    fractions=np.array([-.25,0,.25,.5,.75,1,1.25])
    for label,finish in [('Truth to old',old['estimate']),('Truth to new',new['estimate'])]:
        path=output/(label.replace(' ','_')+'.json')
        if path.exists():curve=json.loads(path.read_text())
        else:
            print('Loss path:',label,flush=True)
            losses=ws.screen_losses(truth+fractions[:,None]*(finish-truth))
            curve=dict(label=label,fractions=fractions,losses=losses);write_json(path,curve)
        curves.append(curve)
    write_json(output/'paths.json',curves)
    # Fixed-region and fixed-noise metrics diagnose where the gain occurs.
    with np.load(ROOT/'reference_bead_orientation_scan/flip_x/comparison.npz') as data:clean=data['simulated'][new['source_indices']]
    noise=json.loads((ROOT/'multiple_beads_64x64_simulated_noisy_matched_20261015/simulated_acquisition/noise_calibration.json').read_text())
    fit=json.loads((ROOT/'reference_bead_orientation_scan/flip_x/reference_fit.json').read_text())
    signal=np.maximum(clean-fit['background'],0)
    sigmas=np.asarray(noise['gaussian_sigma_by_image'])[new['source_indices']]
    variance=noise['effective_poisson_gain']*signal+sigmas[:,None,None]**2
    foreground=signal>3*sigmas[:,None,None]
    pixel_rows=[]
    for label in candidates:
        with np.load(output/('background_PD_'+label.replace(' ','_')+'.npz')) as data:r=data['residual']
        for j,i in enumerate(new['source_indices']):
            fg=foreground[j]
            pixel_rows.append(dict(candidate=label,image=int(i+1),name=str(acq.names[i]),z_um=float(acq.focal_z[i]*1000),
                rmse=float(np.sqrt(np.mean(r[j]**2))),
                foreground_rmse=float(np.sqrt(np.mean(r[j][fg]**2))) if fg.any() else None,
                nonforeground_rmse=float(np.sqrt(np.mean(r[j][~fg]**2))) if (~fg).any() else None,
                standardized_residual_rms=float(np.sqrt(np.mean(r[j]**2/variance[j]))),
                residual_mean=float(r[j].mean()),neighbor_x=correlation(r[j,:-1],r[j,1:]),
                neighbor_y=correlation(r[j,:,:-1],r[j,:,1:])) )
    csv_rows(output/'frame_residuals.csv',pixel_rows)
    # Refit objects separately in meaningful subsets; estimates remain frozen.
    for name,keep in [('middle_Z',abs(new['focal_z'])<=.001000001),('outer_Z',abs(new['focal_z'])>.001000001),
                      ('positive_probes',new['diversity_strengths'].sum(axis=1)>0),
                      ('negative_probes',new['diversity_strengths'].sum(axis=1)<0)]:
        path=output/(name+'.json')
        if path.exists():continue
        print('Subset:',name,flush=True)
        ws_sub=workspace(new,images=new['images'][keep],focus=new['focal_z'][keep],known=new['diversity_strengths'][keep])
        metrics=[dict(candidate=label,loss=float(ws_sub.loss(coeff)),images=int(keep.sum())) for label,coeff in candidates.items()]
        write_json(path,metrics);del ws_sub
    summary=dict(rms_error_waves=rms,losses=rows,candidates={k:v for k,v in candidates.items()},
        source_indices=new['source_indices'],reference_indices=reference_ids,joint_indices=joint_ids,
        interpretation='Fixed candidate wavefronts, object refit at each point. Held-out reference predictions use a PD-only fitted object and remove imposed+cumulative phase from reference commands. Joint fits add references but do not reoptimize wavefronts.',
        noise_weighting='Standardized residuals use fixed reference-sphere/repeat variance after unweighted object fitting; not a refitted Poisson-likelihood objective.')
    write_json(output/'summary.json',summary)
    from experimental_10_15.compare_background_solutions_report import report
    report(output)


def reference_model_checks(output,new,sim,candidates,truth,reference_ids,joint_ids,known,corrected_all):
    """Separate a common residual hypothesis from fixed-reference object constraints."""
    noisy=load_acquisition(SIM.parents[0]/'simulated_acquisition')
    noisy_images,_=subtract_border_background(noisy.images,10)
    acq=load_acquisition(SOURCE)
    refmask=~acq.is_pd[joint_ids]
    rows=[];predictions=[]
    for dataset,images in [('experiment',corrected_all),('noisy_simulation',noisy_images)]:
        joint={**new,'images':images[joint_ids],'focal_z':acq.focal_z[joint_ids],'diversity_strengths':known}
        for hypothesis in ['common_residual','fixed_unaberrated_references']:
            path=output/(dataset+'_'+hypothesis+'.json')
            if path.exists():rows.extend(json.loads(path.read_text()));continue
            print('Reference constraint:',dataset,hypothesis,flush=True)
            ws=workspace(joint)
            if hypothesis=='fixed_unaberrated_references':
                # Reference optics do not depend on the PD-only unknown.
                fixed=ws._build_otfs(truth)[0][refmask].copy()
                build=ws._build_otfs
                def with_fixed_reference(coefficients,**options):
                    otfs,derivatives,records=build(coefficients,**options)
                    otfs[refmask]=fixed
                    return otfs,derivatives,records
                ws._build_otfs=with_fixed_reference
            metrics=[]
            for label,coeff in candidates.items():
                metric,residual,sample=fit_diagnostics(ws,coeff,joint['images'])
                metric.update(dataset=dataset,hypothesis=hypothesis,candidate=label,
                    pd_rmse=float(np.sqrt(np.mean(residual[~refmask]**2))),
                    reference_rmse=float(np.sqrt(np.mean(residual[refmask]**2))))
                metrics.append(metric)
            write_json(path,metrics);rows.extend(metrics)
            del ws
        # Evaluate PD-only objects on held-out references, under either phase hypothesis.
        reference={**new,'images':images[reference_ids],'focal_z':acq.focal_z[reference_ids],
                   'diversity_strengths':acq.diversities[reference_ids]+acq.baseline-truth}
        ws_ref=workspace(reference)
        fixed=ws_ref._build_otfs(truth)[0]
        source='background_PD' if dataset=='experiment' else 'noisy_simulation'
        for label,coeff in candidates.items():
            with np.load(output/(source+'_'+label.replace(' ','_')+'.npz')) as data:sample=data['sample']
            spectrum=fft.rfft2(sample).reshape(len(new['sample_z']),-1)
            for hypothesis in ['common_residual','fixed_unaberrated_references']:
                otfs=fixed if hypothesis=='fixed_unaberrated_references' else ws_ref._build_otfs(coeff)[0]
                prediction=fft.irfft2(np.einsum('kpf,pf->kf',otfs,spectrum).reshape(
                    len(reference_ids),ws_ref.shape[0],ws_ref.shape[1]//2+1),s=ws_ref.shape)[ws_ref.crop]
                for k,i in enumerate(reference_ids):
                    residual=reference['images'][k]-prediction[k]
                    predictions.append(dict(dataset=dataset,hypothesis=hypothesis,candidate=label,image=int(i+1),
                        z_um=float(acq.focal_z[i]*1000),rmse=float(np.sqrt(np.mean(residual**2))),
                        correlation=correlation(prediction[k],reference['images'][k])))
        del ws_ref
    csv_rows(output/'reference_constraints.csv',rows)
    csv_rows(output/'reference_prediction_controls.csv',predictions)


def constrained_gains(a,b):
    """Minimize sum(b*g^2-2*a*g), g>0, with mean(g^2)=1."""
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if np.any(a<=0) or np.any(b<=0):raise ValueError('Gain update requires positive image correlations and prediction powers.')
    scale=np.median(b);a=a/scale;b=b/scale
    def equation(lam):return float(np.mean((a/(b+lam))**2)-1)
    lower=-float(b.min())+max(float(b.min())*1e-12,1e-15)
    upper=max(float(np.max(a)),1.)
    while equation(upper)>0:upper*=2
    lam=brentq(equation,lower,upper,xtol=1e-13)
    gains=a/(b+lam)
    np.testing.assert_allclose(np.mean(gains**2),1.,rtol=1e-9)
    return gains


def photometric_checks(output,new,sim,candidates):
    """Fixed-object gains and alternating object/gain profile at fixed phases."""
    rows=[];single=[];grouped=[]
    command_ids=np.unique(new['diversity_strengths'],axis=0,return_inverse=True)[1]
    for dataset,source in [('experiment',new),('noisy_simulation',sim)]:
        images=new['images'] if dataset=='experiment' else subtract_border_background(sim['images'],10)[0]
        prefix='background_PD' if dataset=='experiment' else 'noisy_simulation'
        for label,coeff in candidates.items():
            with np.load(output/(prefix+'_'+label.replace(' ','_')+'.npz')) as data:prediction=data['predicted']
            gains=np.maximum(np.sum(prediction*images,axis=(1,2))/np.sum(prediction**2,axis=(1,2)),0.)
            residual=images-gains[:,None,None]*prediction
            single.append(dict(dataset=dataset,candidate=label,rmse=float(np.sqrt(np.mean(residual**2))),
                gain_p10=float(np.quantile(gains,.1)),gain_p50=float(np.median(gains)),gain_p90=float(np.quantile(gains,.9))))
            for kind,groups in [('Z_plane',new['plane_index']),('diversity_command',command_ids)]:
                group_gains=np.ones(len(images))
                for group in np.unique(groups):
                    keep=groups==group
                    group_gains[keep]=max(float(np.sum(prediction[keep]*images[keep])/np.sum(prediction[keep]**2)),0.)
                grouped.append(dict(dataset=dataset,candidate=label,grouping=kind,groups=len(np.unique(groups)),
                    rmse=float(np.sqrt(np.mean((images-group_gains[:,None,None]*prediction)**2)))))
            path=output/('gains_'+dataset+'_'+label.replace(' ','_')+'.json')
            if path.exists():rows.append(json.loads(path.read_text()));continue
            print('Alternating gains/object:',dataset,label,flush=True)
            ws=workspace(source,images=images)
            base=ws._build_otfs(coeff)[0];gains=np.ones(len(images));history=[]
            for iteration in range(13):
                estimated,residual,_=ws._solve(base*gains[:,None,None])
                loss=float(np.sum(ws.weights*(np.sum(abs(residual)**2,axis=0)+ws.rho*np.sum(abs(estimated)**2,axis=0))))
                if history and loss>history[-1]['loss']*(1+1e-10):raise ValueError('Alternating objective increased.')
                history.append(dict(iteration=iteration,loss=loss))
                if iteration>0 and (history[-2]['loss']-loss)/history[-2]['loss']<1e-5:break
                prediction_ft=np.einsum('kpf,pf->kf',base,estimated)
                a=np.sum(ws.weights*np.real(prediction_ft.conj()*ws.observed),axis=1)
                b=np.sum(ws.weights*abs(prediction_ft)**2,axis=1)
                if iteration<12:gains=constrained_gains(a,b)
            interior=fft.irfft2(residual.reshape(len(images),ws.shape[0],ws.shape[1]//2+1),s=ws.shape)[ws.crop]
            row=dict(dataset=dataset,candidate=label,loss=loss,rmse=float(np.sqrt(np.mean(interior**2))),
                gain_p10=float(np.quantile(gains,.1)),gain_p50=float(np.median(gains)),gain_p90=float(np.quantile(gains,.9)),
                gains=gains,history=history,iterations=iteration,
                constraint='positive image gains; mean(gain**2)=1; same rho and signed free object',
                limitation='At most 12 alternating updates at a fixed wavefront, not proof of the global joint minimum; no aberration reoptimization.')
            write_json(path,row);rows.append(row)
            del ws,base
    csv_rows(output/'postfit_gains.csv',single)
    csv_rows(output/'postfit_group_gains.csv',grouped)
    csv_rows(output/'alternating_gains.csv',[{k:v for k,v in r.items() if k not in ['gains','history']} for r in rows])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    compare(**vars(parser.parse_args()))
