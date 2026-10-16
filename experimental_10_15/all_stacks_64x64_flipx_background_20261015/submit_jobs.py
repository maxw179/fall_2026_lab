"""Prepare six full acquired stacks; user runs --submit to queue independent fits."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experimental_10_15.bead_pd import prepare_run
from experimental_10_15.bead_pd_io import load_acquisition,block_average,subtract_border_background,write_json
from experimental_10_15.submit_bead_pd import submit_run
from experimental_10_15.results_site import build_site,LABELS

# Explicit original sources; do not consume filtered/aligned/derived inputs.
ACQUISITIONS=[
    ('no_aberration','tpab1b99d6_fbdc_499d_a11c_a22bdda60563',[5]),
    ('system_AO','tpfebc297e_4bbb_4bb3_a781_d9f1c112c45b',[]),
    ('spherical_0p15w','tpd79692a3_bfa2_43c9_8480_2be795af8a43',[]),
    ('random_0p15w','tp8ea7b0c3_4796_4935_a179_c8189b8a4086',[]),
    ('random_aligned_0p15w','tpe9b7330b_5551_402c_9fe0_92002dbcb0fd',[]),
    ('random_aligned_0p25w','tpeb937c8f_5a32_43ff_a10b_84c9110db9ee',[]),
]
RHO=1e46
COMMON=dict(size=64,pupil_size=64,flip_x=True,background_border_pixels=10)
SOLVER=dict(rho=RHO,sample_planes=21,sample_extent_um=4.,max_iterations=50,random_starts=1000,
    svd_method='qr',screening_batch_size=8,psf_batch_size=8,derivative_batch_size=8,frequency_batch_size=4096)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--submit',action='store_true')
    args=parser.parse_args();runs=[];sources={}
    for tag,run_id,excluded in ACQUISITIONS:
        source=HERE.parent/f'3d_fluorobeads_{tag}'/run_id
        acquisition=load_acquisition(source,exclude_planes=excluded)
        ids=np.flatnonzero(acquisition.acquired&acquisition.is_pd)
        if len(ids)<2:raise ValueError(f'Insufficient acquired PD data in {source}')
        if len(acquisition.modes)!=17:raise ValueError(f'Unexpected mode count in {source}')
        name=f'{tag}_64x64';sources[name]=(source,acquisition,ids)
        runs.append(dict(name=name,acquisition=f'3d_fluorobeads_{tag}',label=LABELS[tag],source=str(source.resolve()),
            size=64,rho=RHO,included=len(ids),excluded_planes=excluded,excluded_image_numbers=[],
            source_status=acquisition.status,z_um=(np.unique(acquisition.focal_z[ids])*1000).tolist()))
    summary=dict(title='Full acquired stacks · 64 × 64 · X mirrored · background removed',
        min_snr=None,z_selection=None,dependency=None,random_starts=1000,rho=RHO,
        criteria=['Six original acquisitions','64 × 64 block means','X mirror','Per-image 10-pixel border median',
                  'No SNR filtering','No Z-range clipping','All 17 modes free','1,000 screened starts','Fast CUDA QR'],
        note='All acquired PD images at all valid Z planes, no inherited SNR or Z-range selection. '
             'The documented failed plane 05 in the no-aberration acquisition remains discarded; this is an acquisition error, not Z-range filtering. '
             'No software registration, intensity scale factors, or many-bead artifact-image exclusions. '
             'Each signed binned image is X mirrored and has its own border median subtracted. '
             'One independent job per acquisition at rho=1e46; all 17 modes free, 21 object planes over 4 microns, max 50 Newton updates. '
             'The many-bead stack is handled separately by its current regularization batch.',runs=runs)
    write_json(HERE/'summary.json',summary)
    jobs=[]
    for config in runs:
        name=config['name'];source,acquisition,ids=sources[name]
        submission=HERE/name/'submission';receipts=list(submission.glob('job_*.json'))
        if len(receipts)>1:raise ValueError(f'Multiple receipts in {submission}; inspect before resubmitting.')
        if receipts:
            receipt=receipts[0];path=submission/'input.npz'
        else:path=prepare_run(source,submission,exclude_planes=config['excluded_planes'],**COMMON)
        expected,offsets=subtract_border_background(block_average(acquisition.images[ids],64)[:,::-1,:],10)
        with np.load(path) as data:
            np.testing.assert_array_equal(data['source_indices'],ids)
            np.testing.assert_array_equal(data['images'],expected)
            np.testing.assert_array_equal(data['image_background_subtracted'],offsets)
            np.testing.assert_array_equal(data['diversity_strengths'],acquisition.diversities[ids]+acquisition.baseline)
            np.testing.assert_array_equal(data['focal_z'],acquisition.focal_z[ids])
            assert data['images'].shape==(len(ids),64,64)
        write_json(submission/'run_settings.json',dict(**SOLVER,**COMMON,exclude_planes=config['excluded_planes'],
            source=str(source.resolve()),source_indices=ids,min_snr=None,z_selection=None,
            initialization='Same default seed; no truth or previous solution supplied as initializer.'))
        if not receipts:
            if not args.submit:
                print(f'Prepared {name}: {len(ids)} images; Z={config["z_um"]}; rho={RHO:g}',flush=True)
                continue
            receipt=submit_run(source,submission,host='mcw77@bouchet.ycrc.yale.edu',
                               exclude_planes=config['excluded_planes'],**COMMON,**SOLVER)
        jobs.append(dict(name=name,size=64,included=len(ids),receipt=str(receipt.resolve()),
                         **json.loads(receipt.read_text())))
        write_json(HERE/'jobs.json',jobs);build_site(HERE)


if __name__=='__main__':main()
