"""Prepare five independent rho cases; only the user runs --submit on their Mac."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experimental_10_15.bead_pd import prepare_run
from experimental_10_15.bead_pd_io import write_json
from experimental_10_15.plot_many_bead_images import SOURCE
from experimental_10_15.reference_bead_simulation import FLAGGED
from experimental_10_15.submit_bead_pd import submit_run
from experimental_10_15.results_site import build_site

CASES=[(f'rho_1e{exponent}',10.**exponent) for exponent in range(47,52)]
OPTIONS=dict(size=64,pupil_size=64,exclude_images=FLAGGED,flip_x=True,background_border_pixels=10)
SOLVER=dict(sample_planes=21,sample_extent_um=4.,max_iterations=50,random_starts=1000,
    svd_method='qr',screening_batch_size=8,psf_batch_size=8,derivative_batch_size=8,frequency_batch_size=4096)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true')
    args=parser.parse_args()
    summary=dict(title='Multiple beads · background removed · regularization comparison',
        min_snr=None,random_starts=1000,z_selection=None,dependency=None,
        criteria=['134 PD images','64 × 64 block means','X mirror','10-pixel border median per image',
                  'All 17 modes free','1,000 screened starts','Fast CUDA QR','Five independent jobs'],
        note='Only rho varies: 1e47, 1e48, 1e49, 1e50 and 1e51. All nine Z planes, same ten artifact exclusions, '
             'same 21-plane free signed object spanning 4 microns, same initialization seed and fast settings. '
             'No intensity scale factors. Jobs have no dependencies and may run concurrently when GPUs are available. '
             'Previous rho=1e46 solve had 0.099212 waves RMS error. The fixed-candidate loss sweep is not a guarantee of recovery.',
        runs=[dict(name=name,acquisition='multiple_beads',label=f'Regularization rho = {rho:.3g}',rho=rho,
                   size=64,source=str(SOURCE.resolve()),included=134,excluded_planes=[],
                   excluded_image_numbers=FLAGGED,z_um=[-2.,-1.5,-1.,-.5,0.,.5,1.,1.5,2.]) for name,rho in CASES])
    write_json(HERE/'summary.json',summary)
    baseline=HERE.parent/'multiple_beads_64x64_flipx_background_subtracted_20261015/multiple_beads_64x64/submission/input.npz'
    with np.load(baseline) as data:expected={k:data[k].copy() for k in data.files}
    jobs=[]
    for name,rho in CASES:
        submission=HERE/name/'submission'
        receipts=list(submission.glob('job_*.json'))
        if len(receipts)>1:raise ValueError(f'Multiple job receipts in {submission}; inspect before resubmitting.')
        if receipts:
            receipt=receipts[0]
            # Do not rewrite a submitted input packet on a later invocation.
            path=submission/'input.npz'
        else:
            path=prepare_run(SOURCE,submission,**OPTIONS)
        with np.load(path) as data:
            for key in ['images','source_indices','modes','focal_z','diversity_strengths','imposed_for_validation',
                        'cumulative_correction','image_background_subtracted','microscope_json']:
                np.testing.assert_array_equal(data[key],expected[key])
            assert data['images'].shape==(134,64,64) and len(data['modes'])==17
        write_json(submission/'run_settings.json',dict(rho=rho,**SOLVER,**OPTIONS,
            preparation='Input images and commands identical to completed background-subtracted rho=1e46 case.',
            initialization='Same solver default seed; no truth or previous estimate used as an initializer.'))
        if not receipts:
            if not args.submit:
                print(f'Prepared {name}: rho={rho:.8g}; 134 images; independent CUDA job',flush=True)
                continue
            receipt=submit_run(SOURCE,submission,host='mcw77@bouchet.ycrc.yale.edu',rho=rho,**OPTIONS,**SOLVER)
        jobs.append(dict(name=name,size=64,rho=rho,included=134,receipt=str(receipt.resolve()),
                         **json.loads(receipt.read_text())))
        write_json(HERE/'jobs.json',jobs)
        build_site(HERE)


if __name__=='__main__':main()
