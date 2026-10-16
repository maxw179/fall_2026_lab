"""Submit the six reviewed inputs; preserve receipts and skip existing jobs."""
from pathlib import Path
import argparse
import json
import sys
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experimental_10_15.bead_pd import prepare_run
from experimental_10_15.bead_pd_io import write_json
from experimental_10_15.submit_bead_pd import submit_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='mcw77@bouchet.ycrc.yale.edu')
    args=parser.parse_args()
    summary=json.loads((HERE/'summary.json').read_text())
    jobs=[]
    for row in summary['runs']:
        folder=HERE/row['name'];submission=folder/'submission'
        receipts=list(submission.glob('job_*.json'))
        if len(receipts)>1:raise RuntimeError(f'Multiple receipts: {submission}')
        if receipts:
            receipt=receipts[0]
            print('Retaining existing submission: '+str(receipt),flush=True)
        else:
            # Recheck exact reviewed measurements and metadata before uploading.
            prepared=prepare_run(row['source'],submission,exclude_planes=row['excluded_planes'],
                                 min_snr=summary['min_snr'])
            with np.load(folder/'input.npz',allow_pickle=False) as reviewed,np.load(prepared,allow_pickle=False) as current:
                assert set(reviewed.files)==set(current.files)
                for key in reviewed.files:np.testing.assert_array_equal(current[key],reviewed[key],err_msg=key)
            try:
                receipt=submit_run(row['source'],submission,host=args.host,
                    exclude_planes=row['excluded_planes'],min_snr=summary['min_snr'])
            except subprocess.CalledProcessError as exc:
                print(exc.stderr or str(exc),file=sys.stderr,flush=True)
                raise
        job=json.loads(receipt.read_text())
        jobs.append(dict(name=row['name'],included=row['included'],below_snr=row['below_snr'],
                         receipt=str(receipt),**job))
        write_json(HERE/'jobs.json',jobs)
    print('All six receipts saved in '+str(HERE/'jobs.json'),flush=True)


if __name__=='__main__':main()
