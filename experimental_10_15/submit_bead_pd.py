"""Explicit submit/watch/fetch commands for experimental fits on Bouchet.

No remote connection is made by prepare, compare, or bundle. submit is explicit.
The remote repository is never overwritten: each job has a source/data snapshot.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import shlex
import shutil
import subprocess
import time
import uuid
import math
import os
import stat
import sys
import zipfile

from experimental_10_15.bead_pd import prepare_run,preparation_arguments,solver_arguments
from experimental_10_15.bead_pd_io import write_json,load_acquisition,solver_settings,performance_settings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_ROOT = '/nfs/roberts/project/pi_cr876/mcw77/fall_2026_lab/results/bouchet/bead_experiments'


def validate_remote(host,root):
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*',host):
        raise ValueError('Use an SSH hostname/alias, optionally user@host, without options.')
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',root) or '..' in PurePosixPath(root).parts:
        raise ValueError('Remote root must be an absolute path without spaces or parent traversals.')


def bundle_run(run,output,*,size=None,pupil_size=None,background=None,z_sign=None,exclude_planes=None,min_snr=None,exclude_images=None,flip_x=False,
               background_border_pixels=None,
               rho=None,sample_planes=None,sample_extent_um=None,max_iterations=None,random_starts=None,
               svd_method=None,screening_batch_size=None,psf_batch_size=None,
               derivative_batch_size=None,frequency_batch_size=None):
    performance = performance_settings(svd_method=svd_method,screening_batch_size=screening_batch_size,
        psf_batch_size=psf_batch_size,derivative_batch_size=derivative_batch_size,
        frequency_batch_size=frequency_batch_size)
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    settings = solver_settings(load_acquisition(run),rho=rho,sample_planes=sample_planes,
        sample_extent_um=sample_extent_um,max_iterations=max_iterations,random_starts=random_starts)
    rho,sample_planes,sample_extent_um,max_iterations,random_starts = (settings[k] for k in
        ('rho','sample_planes','sample_extent_um','max_iterations','random_starts'))
    if (rho <= 0 or not math.isfinite(rho) or sample_planes < 2 or sample_extent_um <= 0 or not math.isfinite(sample_extent_um) or max_iterations < 1
            or random_starts < 0):
        raise ValueError('Invalid solver settings.')
    solver_args = ['--rho',str(rho),'--sample-planes',str(sample_planes),
        '--sample-extent-um',str(sample_extent_um),'--max-iterations',str(max_iterations),
        '--random-starts',str(random_starts)]
    for key,value in performance.items():
        solver_args.extend(['--'+key.replace('_','-'),str(value)])
    input_path = prepare_run(run,output,size=size,pupil_size=pupil_size,background=background,z_sign=z_sign,
                             exclude_planes=exclude_planes,min_snr=min_snr,exclude_images=exclude_images,flip_x=flip_x,
                             background_border_pixels=background_border_pixels)
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    archive = output/f'bead_pd_{identity}.zip'
    files = {Path('input.npz'):input_path,Path('preparation.json'):output/'preparation.json'}
    if min_snr is not None:files[Path('snr_qc.csv')]=output/'snr_qc.csv'
    # Capture the currently tested local sources, including uncommitted changes.
    for directory in ['optimization','utils','experimental_10_15']:
        for file in (ROOT/directory).rglob('*.py'):
            files[file.relative_to(ROOT)] = file
    for file in [ROOT/'experimental_10_15/bouchet_bead_pd.sbatch',ROOT/'environment.yml']:
        files[file.relative_to(ROOT)] = file
    for filename in ['metadata.mat','acquisition.mat']:
        files[Path('original_acquisition')/filename] = Path(run)/filename
    for name in ['alignment.json','alignment.npz','alignment_shifts.csv','source_metadata.mat','simulation_provenance.json','reference_fit.json','noise_calibration.json','noise_provenance.json']:
        if (Path(run)/name).is_file():files[Path('original_acquisition')/name] = Path(run)/name
    try:
        revision = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,
                                  capture_output=True,check=True).stdout.strip()
    except (subprocess.CalledProcessError,FileNotFoundError):
        revision = None
    def digest(file):
        with file.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
    manifest = dict(identity=identity,git_revision=revision,solver_args=solver_args,performance=performance,
        sha256={str(name):digest(file) for name,file in files.items()})
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zip:
        for name,file in files.items(): zip.write(file,str(name))
        zip.writestr('bundle.json',json.dumps(manifest,indent=2)+'\n')
    write_json(output/'bundle.json',manifest)
    return archive,manifest


def ssh_options(host):
    """Reuse one authenticated OpenSSH connection across SSH, SCP and CLI runs."""
    validate_remote(host,'/tmp')
    # A short path avoids macOS's small Unix-domain socket path limit.
    directory = Path('/tmp')/f'bead-pd-ssh-{os.getuid()}'
    directory.mkdir(mode=0o700,exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError(f'SSH control directory must be private and owned by you: {directory}')
    socket = directory/hashlib.sha256(host.encode()).hexdigest()[:24]
    return ['-o','ControlMaster=auto','-o','ControlPersist=600',
            '-o',f'ControlPath={socket}','-o','ServerAliveInterval=30',
            '-o','ServerAliveCountMax=3']


def remote_command(host,command):
    return subprocess.run(['ssh',*ssh_options(host),host,command],
                          text=True,capture_output=True,check=True).stdout


def submit_run(run,output,*,host,remote_root=DEFAULT_REMOTE_ROOT,**options):
    validate_remote(host,remote_root)
    archive,manifest = bundle_run(run,output,**options)
    remote_dir = str(PurePosixPath(remote_root)/manifest['identity'])
    remote_archive = remote_dir+'.zip'
    # Two explicit steps: upload the snapshot, then submit exactly that snapshot.
    remote_command(host,'mkdir -p '+shlex.quote(remote_root))
    subprocess.run(['scp',*ssh_options(host),str(archive),f'{host}:{remote_archive}'],check=True)
    extract = ('import pathlib,zipfile; p=pathlib.Path('+repr(remote_dir)+'); '
               'p.mkdir(exist_ok=False); zipfile.ZipFile('+repr(remote_archive)+').extractall(p)')
    remote_command(host,'python3 -c '+shlex.quote(extract))
    submit = ('cd '+shlex.quote(remote_dir)+' && sbatch --parsable '
        '--output='+shlex.quote(remote_dir+'/slurm-%j.out')+' '
        '--error='+shlex.quote(remote_dir+'/slurm-%j.err')+' '
        'experimental_10_15/bouchet_bead_pd.sbatch '+shlex.join(manifest['solver_args']))
    job_id = remote_command(host,submit).strip().split(';')[0]
    if not job_id.isdigit():
        raise ValueError(f'Unexpected sbatch response: {job_id!r}; check the remote directory before resubmitting.')
    receipt = dict(host=host,job_id=job_id,remote_dir=remote_dir,archive=str(archive.resolve()),
        bundle_identity=manifest['identity'])
    path = Path(output)/f'job_{job_id}.json'; write_json(path,receipt)
    print(f'Submitted job {job_id}. Receipt: {path}',flush=True)
    return path


def read_receipt(receipt):
    job = json.loads(Path(receipt).read_text())
    validate_remote(job['host'],job['remote_dir'])
    if not str(job['job_id']).isdigit(): raise ValueError('Invalid job ID in receipt.')
    return job


def watch_job(receipt,*,interval=10,once=False,max_failures=5):
    if not 1 <= interval <= 60: raise ValueError('Polling interval must be between 1 and 60 seconds.')
    if max_failures < 1: raise ValueError('max_failures must be positive.')
    job = read_receipt(receipt)
    previous = None
    failures = 0
    while True:
        script = ('import json,pathlib; p=pathlib.Path('+repr(job['remote_dir']+'/results/progress.json')+'); '
                  'print(p.read_text() if p.exists() else "{}")')
        try:
            progress = json.loads(remote_command(job['host'],'python3 -c '+shlex.quote(script)))
            try:
                state = remote_command(job['host'], 'squeue -h -j '+job['job_id']+' -o %T').strip()
            except subprocess.CalledProcessError as exc:
                # squeue may reject a completed job ID; transport failures must retry.
                if exc.returncode == 255: raise
                state = ''
            if not state:
                state = remote_command(job['host'],'sacct -n -X -j '+job['job_id']+' --format=State --parsable2').strip()
        except (subprocess.CalledProcessError,json.JSONDecodeError) as exc:
            failures += 1
            reason = (exc.stderr or '').strip() if isinstance(exc,subprocess.CalledProcessError) else str(exc)
            if isinstance(exc,subprocess.CalledProcessError):
                reason = f"exit {exc.returncode}: " + (reason or 'No stderr returned by SSH.')
            print(f'Status read failed ({failures}/{max_failures}): {reason}',file=sys.stderr,flush=True)
            if failures >= max_failures:
                raise RuntimeError(f'Cannot read status for job {job["job_id"]}. '
                    f'The remote job was not cancelled or resubmitted. Resume with: '
                    f'python -m experimental_10_15.submit_bead_pd watch {shlex.quote(str(receipt))}') from exc
            time.sleep(interval)
            continue
        failures = 0
        update = dict(slurm_state=state or 'Unknown',**progress)
        if update != previous:
            print(json.dumps(update,indent=2),flush=True); previous = update
        if once: return update
        if progress.get('status') in ('complete','failed') or (state and not any(
                s in state for s in ('PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED','REQUEUED'))):
            print('Remote results: '+job['remote_dir']+'/results',flush=True)
            return update
        time.sleep(interval)


def fetch_job(receipt,output):
    job = read_receipt(receipt); output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Fetch destination must be new or empty to avoid overwriting previous results.')
    output.mkdir(parents=True,exist_ok=True)
    subprocess.run(['scp',*ssh_options(job['host']),'-r',job['host']+':'+job['remote_dir']+'/results',str(output)],check=True)
    for name in ['bundle.json',f'slurm-{job["job_id"]}.out',f'slurm-{job["job_id"]}.err']:
        subprocess.run(['scp',*ssh_options(job['host']),job['host']+':'+job['remote_dir']+'/'+name,str(output)],check=True)
    shutil.copy2(receipt,output/'job.json')
    print(f'Results saved under {output}/results',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for command in ['bundle','submit']:
        p = sub.add_parser(command); preparation_arguments(p); solver_arguments(p)
        if command == 'submit':
            p.add_argument('--host',required=True,help='SSH login hostname/alias capable of sbatch')
            p.add_argument('--remote-root',default=DEFAULT_REMOTE_ROOT)
            p.add_argument('--watch',action='store_true',help='Watch until job completion after submission')
            p.add_argument('--fetch-output',type=Path,
                           help='Watch until completion, then fetch to this new/empty local directory')
    p = sub.add_parser('watch'); p.add_argument('receipt',type=Path)
    p.add_argument('--interval',type=float,default=10); p.add_argument('--once',action='store_true')
    p.add_argument('--max-failures',type=int,default=5,help='Consecutive failed status polls before stopping; never resubmits')
    p = sub.add_parser('fetch'); p.add_argument('receipt',type=Path); p.add_argument('--output',type=Path,required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    if command == 'bundle': print(bundle_run(**args)[0])
    elif command == 'submit':
        watch = args.pop('watch'); destination = args.pop('fetch_output')
        if destination is not None and destination.exists() and any(destination.iterdir()):
            raise ValueError('Fetch destination must be new or empty.')
        receipt = submit_run(**args)
        if watch or destination is not None:
            watch_job(receipt)
        if destination is not None:
            fetch_job(receipt,destination)
    elif command == 'watch': watch_job(**args)
    else: fetch_job(**args)


if __name__ == '__main__': main()
