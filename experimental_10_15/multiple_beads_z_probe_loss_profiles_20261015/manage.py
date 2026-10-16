"""Compare Z windows and coma/trefoil probe inclusion in the saved objective."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
import uuid
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experimental_10_15.bead_pd_io import write_json

BASE=HERE.parent/'multiple_beads_truth_loss_profiles_20261015'
spec=importlib.util.spec_from_file_location('truth_profile_transport',BASE/'manage.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
transport.HERE=HERE


def select_images(data, z_limit_um, exclude_coma_trefoil):
    modes=data['modes'];probes=data['diversity_strengths']-data['baseline']
    third_order=(modes[:,1]==3)&np.isin(abs(modes[:,0]),[1,3])
    keep=abs(data['focal_z']*1000)<=z_limit_um+1e-8
    if exclude_coma_trefoil:
        keep &= ~np.any(abs(probes[:,third_order])>1e-12,axis=1)
    return np.flatnonzero(keep)


def prepare():
    if (HERE/'prepared.json').exists():return json.loads((HERE/'prepared.json').read_text())
    original=json.loads((BASE/'prepared.json').read_text())
    identity=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_z_probe_profile_'+uuid.uuid4().hex[:8]
    archive=HERE/(identity+'.zip');cases=[]
    with zipfile.ZipFile(original['archive']) as source:
        initial=json.loads(source.read('profile_manifest.json'))
        for case in initial['cases']:
            with np.load(BytesIO(source.read(case['input']))) as data:
                size=int(data['images'].shape[-1])
                for z,z_tag in [(2.,'all'),(1.5,'1p5'),(1.,'1p0')]:
                    for excluded in (False,True):
                        indices=select_images(data,z,excluded)
                        policy='without_ct' if excluded else 'all_probes'
                        label=('All Z' if z==2 else f'Z +/-{z:g} um')
                        label+='; '+('no coma/trefoil probes' if excluded else 'all probes')
                        cases.append(dict(case,name=f'{case["name"]}_z{z_tag}_{policy}',label=label,
                            size=size,z_limit_um=z,exclude_coma_trefoil_probes=excluded,image_count=len(indices),
                            profile_options=dict(image_indices=indices.tolist())))
        config=dict(cases=cases,scan='alpha * true imposed + cumulative',
                    mode_treatment='All 17 imposed/candidate modes unchanged; remove only third-order coma/trefoil probe images.',
                    reference_treatment='Reevaluate saved full-stack estimates on each subset; no new optimization.')
        manifest=dict(identity=identity,diagnostic=config['scan'],
                      source_archive=original['archive'],source_identity=original['identity'],sha256={})
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as target:
            for name in source.namelist():
                if name in ['profile_manifest.json','bundle.json']:continue
                contents=(HERE.parent/'profile_loss.py').read_bytes() if name=='experimental_10_15/profile_loss.py' else source.read(name)
                if name=='profile.sbatch':contents=contents.replace(b'pd-truth-profile',b'pd-z-probe-profile')
                target.writestr(name,contents)
                manifest['sha256'][name]=hashlib.sha256(contents).hexdigest()
            target.writestr('profile_manifest.json',json.dumps(config,indent=2))
            target.writestr('bundle.json',json.dumps(manifest,indent=2))
    write_json(HERE/'profile_manifest.json',config);write_json(HERE/'bundle.json',manifest)
    prepared=dict(identity=identity,archive=str(archive.resolve()),host=original['host'],
                  remote_dir=transport.DEFAULT_REMOTE_ROOT+'/'+identity)
    write_json(HERE/'prepared.json',prepared)
    print(f'Prepared {len(cases)} scans x 51 fractions; original wavefront and diversities preserved.',flush=True)
    return prepared


def plot():
    import matplotlib.pyplot as plt
    from systematic_testing.rms_cliff_sweep.diagnose_failures import plot_scaled_truth_profiles
    output=HERE/'figures';output.mkdir(exist_ok=True)
    config=json.loads((HERE/'profile_manifest.json').read_text());rows=[];minima=[];data=[]
    for case in config['cases']:
        saved=json.loads((HERE/'solve/results'/(case['name']+'.json')).read_text())
        assert saved['shape'][0]==case['image_count']
        assert np.array_equal(saved['image_indices'],case['profile_options']['image_indices'])
        data.append((case,saved))
        for fraction,loss,ratio in zip(saved['fractions'],saved['losses'],saved['loss_over_truth']):
            rows.append(dict(size=case['size'],z_limit_um=case['z_limit_um'],
                excluded_coma_trefoil_probes=case['exclude_coma_trefoil_probes'],images=case['image_count'],
                fraction=fraction,loss=loss,loss_over_truth=ratio))
        minima.append(dict(size=case['size'],z_limit_um=case['z_limit_um'],
            excluded_coma_trefoil_probes=case['exclude_coma_trefoil_probes'],images=case['image_count'],
            best_fraction=saved['best_fraction'],best_loss_over_truth=saved['best_loss']/saved['truth_loss'],
            truth_loss=saved['truth_loss'],minimum_at_boundary=saved['minimum_at_boundary'],
            **{f'saved_{ref["random_starts"]}_start_estimate_over_truth':ref['loss_over_truth'] for ref in saved['references']}))
    def save(stem,curves,levels=None):
        fig,_=plot_scaled_truth_profiles(curves,levels=levels)
        for ext in ('png','pdf'):fig.savefig(output/(stem+'.'+ext),dpi=300)
        plt.close(fig)
    for size in (128,64,32):
        for excluded in (False,True):
            subset=[(case,saved) for case,saved in data if case['size']==size and case['exclude_coma_trefoil_probes']==excluded]
            curves=[('All Z' if case['z_limit_um']==2 else fr'$|z|\leq {case["z_limit_um"]:g}$ $\mu$m',
                     saved['fractions'],saved['loss_over_truth']) for case,saved in subset]
            save(f'z_windows_{size}_'+('without_ct' if excluded else 'all_probes'),curves)
        for z,z_tag in [(2.,'all'),(1.5,'1p5'),(1.,'1p0')]:
            subset=[(case,saved) for case,saved in data if case['size']==size and case['z_limit_um']==z]
            curves=[('Without coma/trefoil probes' if case['exclude_coma_trefoil_probes'] else 'All probes',
                     saved['fractions'],saved['loss_over_truth']) for case,saved in subset]
            save(f'probe_comparison_{size}_z{z_tag}',curves)
            # Separate plots retain horizontal ratios of the saved estimates.
            for case,saved in subset:
                label='Without coma/trefoil probes' if case['exclude_coma_trefoil_probes'] else 'All probes'
                levels={ref['label']+' saved estimate':ref['loss_over_truth'] for ref in saved['references']}
                save(case['name'],[(label,saved['fractions'],saved['loss_over_truth'])],levels)
    for filename,values in [('profiles.csv',rows),('minima.csv',minima)]:
        with (output/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    print('Figures and CSV data: '+str(output),flush=True)


def collect(local_only=False):
    if not (HERE/'solve/results/result.json').exists():
        if local_only:raise ValueError('No downloaded results.')
        receipts=list(HERE.glob('job_*.json'))
        if len(receipts)!=1:raise ValueError('Submit first.')
        status=transport.watch_job(receipts[0])
        if status.get('status')!='complete':raise ValueError('Diagnostic failed; inspect remote logs.')
        transport.fetch_job(receipts[0],HERE/'solve')
    plot()


transport.prepare=prepare


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','submit','collect'])
    parser.add_argument('--local-only',action='store_true');args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='submit':transport.submit()
    else:collect(args.local_only)
