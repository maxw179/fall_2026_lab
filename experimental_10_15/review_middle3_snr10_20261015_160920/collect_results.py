"""Watch existing reviewed jobs, fetch results, and build their HTML replays."""
from pathlib import Path
import json
import sys
import runpy

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experimental_10_15.submit_bead_pd import watch_job,fetch_job
from experimental_10_15.replay import export_replay


def main():
    jobs=json.loads((HERE/'jobs.json').read_text())
    for job in jobs:
        folder=HERE/job['name'];fetched=folder/'solve';results=fetched/'results'
        complete=all((results/name).is_file() for name in ['result.json','optimizer.json','solution.npz'])
        if not complete:
            watch_job(job['receipt'])
            fetch_job(job['receipt'],fetched)
        if not (results/'result.json').exists():
            print(f'{job["name"]}: no completed solution; inspect {results}/progress.json',flush=True)
            continue
        output=fetched/'figures/correction_replay.html'
        if not output.exists():
            export_replay(results,folder/'submission/input.npz',output)
        print(f'{job["name"]}: {output}',flush=True)
    runpy.run_path(str(HERE/'build_site.py'),run_name='__main__')


if __name__=='__main__':main()
