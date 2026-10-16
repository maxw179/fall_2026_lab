"""Prepare or submit all 12 full-stack binning comparisons without dependencies."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experimental_10_15.bead_pd import prepare_run
from experimental_10_15.bead_pd_io import performance_settings,write_json
from experimental_10_15.submit_bead_pd import bundle_run,submit_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='mcw77@bouchet.ycrc.yale.edu')
    parser.add_argument('--prepare-only',action='store_true',help='Build/validate ZIPs without connecting')
    args=parser.parse_args()
    summary=json.loads((HERE/'summary.json').read_text())
    jobs=[];packages=[]
    for row in summary['runs']:
        folder=HERE/row['name'];output=folder/'submission'
        receipts=list(output.glob('job_*.json'))
        if len(receipts)>1:raise RuntimeError(f'Multiple receipts: {output}')
        if receipts:
            receipt=receipts[0]
            print('Retaining existing job: '+str(receipt),flush=True)
        else:
            prepared=prepare_run(row['source'],output,size=row['size'],
                                 exclude_planes=row['excluded_planes'],min_snr=None)
            with np.load(folder/'input.npz',allow_pickle=False) as expected,np.load(prepared,allow_pickle=False) as current:
                assert set(expected.files)==set(current.files)
                for key in expected.files:np.testing.assert_array_equal(expected[key],current[key],err_msg=key)
                assert current['images'].shape==(row['included'],row['size'],row['size'])
                assert 'min_snr' not in current and 'image_snr' not in current
            options=dict(size=row['size'],exclude_planes=row['excluded_planes'],min_snr=None,
                         **performance_settings())
            if args.prepare_only:
                archive,manifest=bundle_run(row['source'],output,**options)
                packages.append(dict(name=row['name'],included=row['included'],size=row['size'],
                    archive=str(archive.resolve()),identity=manifest['identity'],performance=manifest['performance']))
                write_json(HERE/'prepared_bundles.json',packages)
                print(f'Prepared {row["name"]}: {row["included"]} images; fast CUDA; no filters/dependencies',flush=True)
                continue
            try:
                receipt=submit_run(row['source'],output,host=args.host,**options)
            except subprocess.CalledProcessError as exc:
                print(exc.stderr or str(exc),file=sys.stderr,flush=True)
                raise
        job=json.loads(receipt.read_text())
        jobs.append(dict(name=row['name'],size=row['size'],included=row['included'],
                         receipt=str(receipt),**job))
        write_json(HERE/'jobs.json',jobs)
    print(f'{len(packages)} bundles prepared.' if args.prepare_only else
          f'{len(jobs)} job receipts saved. No Slurm dependencies were set.',flush=True)


if __name__=='__main__':main()
