"""Prepare, submit or collect the multiple-bead truth-scaling diagnostic."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experimental_10_15.bead_pd_io import write_json
from experimental_10_15.submit_bead_pd import (DEFAULT_REMOTE_ROOT,remote_command,ssh_options,
                                              watch_job,fetch_job,read_receipt)


def prepare():
    if (HERE/'prepared.json').exists():return json.loads((HERE/'prepared.json').read_text())
    cases=[];files={}
    for size in (128,64,32):
        name=f'multiple_beads_{size}x{size}'
        original=HERE.parent/'multiple_beads_native_64_32_20261015'/name
        repeated=HERE.parent/'multiple_beads_random1000_native_64_32_20261015'/name
        with np.load(original/'submission/input.npz') as first,np.load(repeated/'submission/input.npz') as second:
            assert set(first.files)==set(second.files)
            for key in first.files:np.testing.assert_array_equal(first[key],second[key])
        input_name=f'cases/{size}/input.npz'
        files[input_name]=original/'submission/input.npz'
        references=[]
        for starts,folder in [(100,original),(1000,repeated)]:
            target=f'cases/{size}/starts{starts}'
            for filename in ['solution.npz','result.json','optimizer.json']:
                source=folder/'solve/results'/filename
                if not source.exists():raise ValueError(f'Missing completed reference: {source}')
                files[f'{target}/{filename}']=source
            result=json.loads((folder/'solve/results/result.json').read_text())
            assert result['status']=='complete' and result['random_starts']==starts
            references.append(dict(label=f'{starts:,} starts',results=target))
        cases.append(dict(name=name,label=f'{size} x {size}'+(' native' if size==128 else ''),
                          input=input_name,references=references))
    for directory in ['optimization','utils','experimental_10_15']:
        for file in (ROOT/directory).rglob('*.py'):files[file.relative_to(ROOT).as_posix()]=file
    identity=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_truth_profile_'+uuid.uuid4().hex[:8]
    archive=HERE/(identity+'.zip')
    script=(HERE.parent/'bouchet_bead_pd.sbatch').read_text().replace(
        '#SBATCH --job-name=pd-bead-experiment','#SBATCH --job-name=pd-truth-profile').replace(
        '#SBATCH --time=02:00:00','#SBATCH --time=00:30:00').replace(
        'python -u -m experimental_10_15.bead_pd solve input.npz --backend cuda --output results "$@"',
        'python -u -m experimental_10_15.profile_loss run profile_manifest.json --backend cuda --output results')
    manifest=dict(identity=identity,diagnostic='alpha * imposed + cumulative; profiled regularized object',
        fractions=dict(minimum=-.5,maximum=2.,step=.05),
        sha256={name:hashlib.sha256(file.read_bytes()).hexdigest() for name,file in files.items()})
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name,file in files.items():bundle.write(file,name)
        bundle.writestr('profile_manifest.json',json.dumps(dict(cases=cases),indent=2))
        bundle.writestr('profile.sbatch',script)
        bundle.writestr('bundle.json',json.dumps(manifest,indent=2))
    write_json(HERE/'bundle.json',manifest)
    prepared=dict(identity=identity,archive=str(archive.resolve()),host='mcw77@bouchet.ycrc.yale.edu',
                  remote_dir=DEFAULT_REMOTE_ROOT+'/'+identity)
    write_json(HERE/'prepared.json',prepared)
    print(f'Prepared 51 fractions for each of {len(cases)} resolutions: {archive}',flush=True)
    return prepared


def submit():
    existing=list(HERE.glob('job_*.json'))
    if existing:
        if len(existing)!=1:raise ValueError('Multiple receipts; inspect before proceeding.')
        print('Already submitted: '+str(existing[0]));return
    job=prepare();host=job['host'];remote=job['remote_dir']
    remote_command(host,'mkdir -p '+shlex.quote(DEFAULT_REMOTE_ROOT))
    subprocess.run(['scp',*ssh_options(host),job['archive'],host+':'+remote+'.zip'],check=True)
    script='import pathlib,zipfile; p=pathlib.Path('+repr(remote)+'); p.mkdir(exist_ok=False); zipfile.ZipFile('+repr(remote+'.zip')+').extractall(p)'
    remote_command(host,'python3 -c '+shlex.quote(script))
    job_id=remote_command(host,'cd '+shlex.quote(remote)+' && sbatch --parsable --output='+
        shlex.quote(remote+'/slurm-%j.out')+' --error='+shlex.quote(remote+'/slurm-%j.err')+' profile.sbatch').strip().split(';')[0]
    if not job_id.isdigit():raise ValueError('Unexpected sbatch response: '+job_id)
    write_json(HERE/f'job_{job_id}.json',dict(job,job_id=job_id,bundle_identity=job['identity']))
    print('Submitted truth-scaling profile job '+job_id,flush=True)


def collect(local_only=False):
    from experimental_10_15.profile_loss import plot
    if not (HERE/'solve/results/result.json').exists():
        if local_only:raise ValueError('No downloaded completed profile yet.')
        receipts=list(HERE.glob('job_*.json'))
        if len(receipts)!=1:raise ValueError('Submit the profile job first.')
        status=watch_job(receipts[0])
        if status.get('status')!='complete':raise ValueError('Profile job did not complete; inspect remote logs.')
        fetch_job(receipts[0],HERE/'solve')
    plot(HERE/'solve/results',HERE/'figures')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','submit','collect'])
    parser.add_argument('--local-only',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='submit':submit()
    else:collect(args.local_only)


if __name__=='__main__':main()
