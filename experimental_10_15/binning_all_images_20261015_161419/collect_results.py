"""Watch existing reviewed jobs, fetch results, and build their HTML replays."""
from pathlib import Path
import json
import sys
import argparse
import shlex

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experimental_10_15.submit_bead_pd import watch_job,fetch_job,remote_command
from experimental_10_15.replay import export_replay
from experimental_10_15.results_site import build_site
from experimental_10_15.bead_pd_io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--ready-only',action='store_true',help='Check all jobs and collect only finished solves; do not wait')
    mode.add_argument('--local-only',action='store_true',help='Render already downloaded results without SSH')
    args=parser.parse_args()
    jobs=json.loads((HERE/'jobs.json').read_text())
    ready=set()
    if args.ready_only:
        assert len({job['host'] for job in jobs})==1
        script='''import pathlib,json
jobs=JOBS
states=[]
for job in jobs:
 root=pathlib.Path(job['remote_dir'])/'results'
 result=root/'result.json';progress=root/'progress.json'
 state=json.loads(result.read_text()) if result.exists() else json.loads(progress.read_text()) if progress.exists() else {}
 states.append(dict(name=job['name'],job_id=job['job_id'],status=state.get('status','Awaiting results'),
  complete=state.get('status')=='complete' and all((root/name).is_file() for name in ['result.json','optimizer.json','solution.npz'])))
print(json.dumps(states))
'''.replace('JOBS',repr(jobs))
        states=json.loads(remote_command(jobs[0]['host'],'python3 -c '+shlex.quote(script)))
        write_json(HERE/'latest_completion_status.json',states)
        ready={row['name'] for row in states if row['complete']}
    build_site(HERE)
    errors=[]
    for job in jobs:
        folder=HERE/job['name'];fetched=folder/'solve';results=fetched/'results'
        complete=all((results/name).is_file() for name in ['result.json','optimizer.json','solution.npz'])
        if not complete:
            if args.local_only or (args.ready_only and job['name'] not in ready):
                print(job['name']+': awaiting completed download',flush=True)
                continue
            if not args.ready_only:watch_job(job['receipt'])
            try:
                fetch_job(job['receipt'],fetched)
            except Exception as exc:
                errors.append(job['name']);print(f'{job["name"]}: download failed: {exc}',flush=True)
                continue
        if not (results/'result.json').exists():
            print(f'{job["name"]}: no completed solution; inspect {results}/progress.json',flush=True)
            continue
        output=fetched/'figures/correction_replay.html'
        if not output.exists():
            try:
                export_replay(results,folder/'submission/input.npz',output)
            except Exception as exc:
                errors.append(job['name']);print(f'{job["name"]}: replay failed: {exc}',flush=True)
                continue
        print(f'{job["name"]}: {output}',flush=True)
        build_site(HERE)
    build_site(HERE)
    if errors:raise SystemExit('Incomplete cases: '+', '.join(errors))


if __name__=='__main__':main()
