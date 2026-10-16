"""Prepare locally; --submit explicitly uploads and queues the three sizes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experimental_10_15.bead_pd import prepare_run
from experimental_10_15.bead_pd_io import load_acquisition,block_average,write_json
from experimental_10_15.submit_bead_pd import submit_run
from experimental_10_15.results_site import build_site


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    source=HERE.parent/'3d_many_fluorobeads_random_aligned_0p15w/tp76e42de4_ca3a_4172_9562_d6cf4b94918c'
    acquisition=load_acquisition(source)
    selected=np.flatnonzero(acquisition.acquired & acquisition.is_pd)
    assert acquisition.status=='complete' and len(selected)==144
    summary=dict(title='Multiple fluorobeads · 1,000 random starts · native, 64 × 64 and 32 × 32',min_snr=None,random_starts=1000,
        z_selection=None,dependency=None,
        criteria=['All 9 acquired Z planes','No SNR filtering','128 × 128 native; 64 × 64 and 32 × 32 block means','Fast CUDA QR','1,000 random starts'],
        note='All 144 acquired PD images are retained per fit across Z = -2 to +2 µm in 0.5 µm steps. '
             'Native pixels are unchanged; binned images are block means with the same 6.5 µm FOV. '
             'The fit screens 1,000 random initial candidates. All 17 aberration modes are free. The object is reconstructed freely; no single-bead object is imposed.',runs=[])
    jobs=[]
    for size in (128,64,32):
        name=f'multiple_beads_{size}x{size}'
        submission=HERE/name/'submission'
        path=prepare_run(source,submission,size=size,pupil_size=64)
        with np.load(path) as data:
            np.testing.assert_array_equal(data['images'],block_average(acquisition.images[selected],size))
            np.testing.assert_array_equal(data['source_indices'],selected)
            assert len(data['modes'])==17 and float(data['field_of_view_um'])==6.5
        summary['runs'].append(dict(name=name,acquisition='multiple_beads',
            label='Multiple beads'+(' · native' if size==128 else ''),size=size,
            source=str(source.resolve()),included=len(selected),excluded_planes=[],
            z_um=(np.unique(acquisition.focal_z[selected])*1000).tolist()))
        write_json(HERE/'summary.json',summary)
        receipts=list(submission.glob('job_*.json'))
        if len(receipts)>1:raise ValueError(f'Multiple receipts in {submission}; inspect before resubmitting.')
        if receipts:
            receipt=receipts[0]
        elif args.submit:
            receipt=submit_run(source,submission,host='mcw77@bouchet.ycrc.yale.edu',size=size,pupil_size=64,
                random_starts=1000,svd_method='qr',screening_batch_size=8,psf_batch_size=8,derivative_batch_size=8,frequency_batch_size=4096)
        else:
            print(f'Prepared {name}: {len(selected)} images',flush=True)
            continue
        jobs.append(dict(name=name,size=size,included=len(selected),receipt=str(receipt.resolve()),
                         **json.loads(receipt.read_text())))
        write_json(HERE/'jobs.json',jobs)
    if jobs:build_site(HERE)


if __name__=='__main__':main()
