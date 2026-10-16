"""Profile the saved experimental objective along alpha * imposed + cumulative.

The regularized, unconstrained object is re-estimated at every point. Fixed
diversities, calibration, images, padding, rho and object grid stay unchanged.
Truth is used only for this diagnostic; optimization outputs are never edited.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from experimental_10_15.bead_pd import acquisition_grid
from experimental_10_15.bead_pd_io import write_json
from optimization import reconstruction
from utils import psf, zernike


def profile_case(input_path, references, *, fractions=None, backend='cuda', progress=None,
                 image_indices=None):
    fractions=np.unique(np.r_[np.linspace(-.5,2.,51) if fractions is None else fractions,0.,1.])
    if not np.all(np.isfinite(fractions)):raise ValueError('Fractions must be finite.')
    with np.load(input_path,allow_pickle=False) as data:
        images=data['images'];modes=data['modes'];known=data['diversity_strengths']
        focus=data['focal_z'];truth=data['imposed_for_validation'];cumulative=data['cumulative_correction']
        microscope=psf.Microscope(**json.loads(str(data['microscope_json'])))
        grid=acquisition_grid(float(data['field_of_view_um']),images.shape[-2:])
        saved=json.loads(str(data['solver_json']))
    if not np.any(truth):raise ValueError('A zero imposed wavefront gives a constant scaling curve.')
    comparison=[]
    sample_z=None
    for ref in references:
        results=Path(ref['results'])
        result=json.loads((results/'result.json').read_text())
        with np.load(results/'solution.npz',allow_pickle=False) as solved:
            for key,value in [('modes',modes),('focal_z',focus),('cumulative_correction',cumulative),
                              ('imposed_for_validation',truth)]:
                if not np.array_equal(solved[key],value):raise ValueError(f'Reference {key} mismatch.')
            if sample_z is not None and not np.array_equal(sample_z,solved['sample_z']):
                raise ValueError('Reference object grids differ.')
            sample_z=solved['sample_z'].copy()
            strengths=solved['estimated_strengths'].copy()
        settings={key:result[key] for key in ('rho','sample_planes','sample_extent_um')}
        if comparison and settings!=comparison[0]['settings']:raise ValueError('Reference solver settings differ.')
        history=json.loads((results/'optimizer.json').read_text())['history']
        if not np.allclose(history[-1]['strengths'],strengths,rtol=1e-10,atol=1e-12):
            raise ValueError('Reference final iterate does not match solution.')
        comparison.append(dict(label=ref['label'],strengths=strengths,settings=settings,
                               recorded_loss=float(history[-1]['loss']),random_starts=result['random_starts']))
    settings=comparison[0]['settings'] if comparison else saved
    if sample_z is None:
        sample_z=np.linspace(-settings['sample_extent_um']/2000,settings['sample_extent_um']/2000,settings['sample_planes'])
    original_count=len(images)
    indices=np.arange(original_count) if image_indices is None else np.asarray(image_indices)
    if (indices.ndim!=1 or len(indices)<2 or not np.issubdtype(indices.dtype,np.integer)
            or np.any(indices<0) or np.any(indices>=original_count) or len(np.unique(indices))!=len(indices)):
        raise ValueError('Select at least two distinct valid integer image indices.')
    same_objective=np.array_equal(indices,np.arange(original_count))
    images=images[indices];focus=focus[indices];known=known[indices]
    backend=reconstruction.get_backend(backend)
    options=dict(psf_batch_size=8,frequency_batch_size=4096,reuse_svd=False,
                 retain_fields=False,cache_last_fit=False,deduplicate_psfs=True)
    if backend=='cuda':options.update(svd_method='qr',screening_method='linear',screening_batch_size=8,screening_fallback='qr')
    workspace=reconstruction.ReconstructionWorkspace(microscope,grid,images,focus,sample_z,modes,
        settings['rho'],diversities=[zernike.Aberration(modes,value) for value in known],
        mode='vector',backend=backend,**options)
    candidates=cumulative+fractions[:,None]*truth
    losses=workspace.screen_losses(candidates,on_completed=progress)
    # Check the endpoints, truth and sampled winner using canonical full-fit loss.
    anchors=sorted({0,len(fractions)-1,int(np.argmin(losses)),
                    int(np.flatnonzero(fractions==0)[0]),int(np.flatnonzero(fractions==1)[0])})
    checks=[]
    for index in anchors:
        canonical=float(workspace.loss(candidates[index]))
        relative=abs(canonical-losses[index])/max(abs(canonical),np.finfo(float).tiny)
        if relative>1e-7:raise ValueError(f'Batched/canonical loss mismatch: {relative:g}')
        checks.append(dict(fraction=float(fractions[index]),relative_error=relative))
        losses[index]=canonical
    if not np.all(np.isfinite(losses)) or np.any(losses<=0):raise ValueError('Losses must be finite and positive.')
    true_loss=float(losses[np.flatnonzero(fractions==1)[0]])
    for ref in comparison:
        ref['loss']=float(workspace.loss(ref['strengths']))
        ref['objective_matches_original']=same_objective
        ref['relative_replay_error']=(abs(ref['loss']-ref['recorded_loss'])/max(abs(ref['recorded_loss']),np.finfo(float).tiny)
                                      if same_objective else None)
        if same_objective and ref['relative_replay_error']>1e-7:
            raise ValueError('Replayed loss disagrees with the saved optimizer objective.')
        ref['loss_over_truth']=ref['loss']/true_loss
    index=int(np.argmin(losses))
    return dict(fractions=fractions,losses=losses,loss_over_truth=losses/true_loss,
        truth_loss=true_loss,best_fraction=float(fractions[index]),best_loss=float(losses[index]),
        minimum_at_boundary=index in (0,len(fractions)-1),references=comparison,
        canonical_checks=checks,settings=settings,backend=backend,shape=images.shape,
        imposed_waves=truth,cumulative_correction_waves=cumulative,modes=modes,
        sample_z_mm=sample_z,input_path=str(input_path),image_indices=indices,
        original_image_count=original_count,retained_z_um=np.unique(focus)*1000,
        reference_treatment='Saved coefficient vectors reevaluated on selected images; no reoptimization.')


def run(manifest, output, *, backend='cuda'):
    manifest=Path(manifest);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    config=json.loads(manifest.read_text());started=perf_counter();summaries=[]
    try:
        for case in config['cases']:
            def progress(count):
                write_json(output/'progress.json',dict(status='running',case=case['name'],completed=count,total=51,
                    elapsed_seconds=perf_counter()-started))
            print('Profiling '+case['name'],flush=True)
            references=[dict(ref,results=manifest.parent/ref['results']) for ref in case['references']]
            data=profile_case(manifest.parent/case['input'],references,backend=backend,progress=progress,
                              **case.get('profile_options',{}))
            data.update(name=case['name'],label=case['label'])
            write_json(output/(case['name']+'.json'),data)
            summaries.append(dict(name=case['name'],label=case['label'],best_fraction=data['best_fraction'],
                                  best_loss_over_truth=data['best_loss']/data['truth_loss']))
            print(summaries[-1],flush=True);gc.collect()
        write_json(output/'result.json',dict(status='complete',cases=summaries,elapsed_seconds=perf_counter()-started))
        write_json(output/'progress.json',dict(status='complete',elapsed_seconds=perf_counter()-started))
    except Exception as exc:
        write_json(output/'progress.json',dict(status='failed',error=str(exc),elapsed_seconds=perf_counter()-started))
        raise


def plot(results, output, *, use_tex=True):
    import matplotlib.pyplot as plt
    from systematic_testing.rms_cliff_sweep.diagnose_failures import plot_scaled_truth_profiles
    results=Path(results);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((results/'result.json').read_text());curves=[];rows=[];minima=[]
    for case in summary['cases']:
        data=json.loads((results/(case['name']+'.json')).read_text())
        curve=(case['label'],data['fractions'],data['loss_over_truth']);curves.append(curve)
        levels={ref['label']+' fit (all 17 modes)':ref['loss_over_truth'] for ref in data['references']}
        fig,_=plot_scaled_truth_profiles([curve],levels=levels,use_tex=use_tex)
        for ext in ['png','pdf']:fig.savefig(output/(case['name']+'.'+ext),dpi=300)
        plt.close(fig)
        minima.append(dict(case=case['name'],best_fraction=data['best_fraction'],
            best_loss_over_truth=data['best_loss']/data['truth_loss'],
            minimum_at_boundary=data['minimum_at_boundary'],
            **{f'fit_{ref["random_starts"]}_starts_over_truth':ref['loss_over_truth'] for ref in data['references']}))
        rows.extend(dict(case=case['name'],fraction=fraction,loss=loss,loss_over_truth=ratio)
                    for fraction,loss,ratio in zip(data['fractions'],data['losses'],data['loss_over_truth']))
    fig,_=plot_scaled_truth_profiles(curves,use_tex=use_tex)
    for ext in ['png','pdf']:fig.savefig(output/('all_resolutions.'+ext),dpi=300)
    plt.close(fig)
    with (output/'profiles.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (output/'minima.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(minima[0]));writer.writeheader();writer.writerows(minima)
    print('Figures and CSV data saved in '+str(output),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('run');p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',choices=['cpu','cuda','auto'],default='cuda')
    p=sub.add_parser('plot');p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-tex',dest='use_tex',action='store_false')
    args=vars(parser.parse_args());command=args.pop('command')
    (run if command=='run' else plot)(**args)


if __name__=='__main__':main()
