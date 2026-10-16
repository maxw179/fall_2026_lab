"""Animate saved aberration iterates and refitted 3D objects without optimization.

Uses utils.plotting.animate_reconstruction (polar phase + thresholded voxels).
Default: the completed rho=1e49 many-bead fit. No cluster access is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from experimental_10_15.image_model_diagnostics import ROOT,load_case,workspace
from experimental_10_15.bead_pd import acquisition_grid
from experimental_10_15.bead_pd_io import write_json
from utils import psf,plotting

DEFAULT=ROOT/'multiple_beads_64x64_regularization_20261015/rho_1e49'


def selected_history(history,stride):
    if not history or not isinstance(stride,int) or stride<1:raise ValueError('Require history and a positive integer stride.')
    return [history[i] for i in sorted(set(range(0,len(history),stride))|{len(history)-1})]


def sample_objects(values,grid,z):
    return [psf.Image_Mask_3D([psf.Image_Mask(psf.Arbitrary_Grid(
        grid.L_ffp_x,grid.L_ffp_y,grid.grid_ffp_x,grid.grid_ffp_y,grid.x_offset,grid.y_offset,float(level)),plane)
        for level,plane in zip(z,sample)]) for sample in values]


def export(case_folder=DEFAULT,output=None,stride=1,percentile=98.,dpi=150,interval=700.):
    case_folder=Path(case_folder).resolve()
    output=Path(output) if output else case_folder/'solve/figures/animations'
    if not 0<=percentile<=100 or dpi<1 or interval<=0:raise ValueError('Invalid animation settings.')
    output.mkdir(parents=True,exist_ok=True)
    case=load_case(case_folder)
    history=json.loads((case_folder/'solve/results/optimizer.json').read_text())['history']
    selected=selected_history(history,stride)
    np.testing.assert_allclose(selected[-1]['strengths'],case['estimate'],rtol=1e-10,atol=1e-12)
    sources={name:hashlib.sha256((case_folder/name).read_bytes()).hexdigest() for name in
             ['submission/input.npz','solve/results/optimizer.json','solve/results/solution.npz','solve/results/result.json']}
    identity=json.dumps(dict(sources=sources,iterations=[h['iteration'] for h in selected]),sort_keys=True)
    cache=output/'refitted_objects.npz'
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved['identity'])!=identity:raise ValueError('Animation cache belongs to different inputs/iterations; choose another output directory.')
            values=saved['samples'].copy()
    else:
        ws=workspace(case);values=[]
        for i,entry in enumerate(selected):
            sample=ws.estimate_sample(np.asarray(entry['strengths']))
            values.append(sample.image_mask)
            print(f'Refit object {i+1}/{len(selected)}: saved iteration {entry["iteration"]}',flush=True)
        values=np.asarray(values)
        with np.load(case_folder/'solve/results/solution.npz') as saved:
            scale=max(float(np.max(abs(saved['sample']))),np.finfo(float).tiny)
            np.testing.assert_allclose(values[-1],saved['sample'],rtol=1e-5,atol=scale*1e-8)
        np.savez_compressed(cache,identity=np.array(identity),samples=values)
        del ws
    grid=acquisition_grid(float(case['field_of_view_um']),case['images'].shape[-2:])
    samples=sample_objects(values,grid,case['sample_z'])
    info=dict(history=[dict(entry,strengths=(np.asarray(entry['strengths'])-case['cumulative_correction']).tolist(),
        title=f'Saved iteration {entry["iteration"]}\n'+rf'$\rho=10^{{{np.log10(case["solver"]["rho"]):g}}}$') for entry in selected])
    with plt.style.context('bmh'),plt.rc_context({'text.usetex':True,'font.size':9}):
        phase,objects=plotting.animate_reconstruction(info,case['modes'],psf.Microscope(**case['params']),
            precomputed_samples=samples,percentile=percentile,res=151,dpi=dpi,interval=interval)
        phase.save(output/'aberration_estimate.gif',writer=PillowWriter(fps=1000/interval),dpi=dpi)
        objects.save(output/'object_estimate.gif',writer=PillowWriter(fps=1000/interval),dpi=dpi)
        document=('<!doctype html><html><head><meta charset="utf-8"><title>Saved aberration and object estimates</title></head>'
            '<body><h1>Aberration estimate</h1>'+phase.to_jshtml()+
            '<h1>Object estimate</h1>'+objects.to_jshtml()+
            '<p>Saved optimizer iterations; objects are refitted with the recorded regularization and optics. '
            'A fixed threshold and shared voxel color scale are used across all frames. '
            'No new optimization or physical microscope correction is performed.</p></body></html>')
        (output/'animations.html').write_text(document)
    write_json(output/'animation_metadata.json',dict(case=case_folder,sources=sources,
        iterations=[h['iteration'] for h in selected],rho=case['solver']['rho'],percentile=percentile,
        threshold=float(np.percentile(values[-1],percentile)),dpi=dpi,interval_ms=interval,
        phase='Estimated imposed phase (saved strengths minus known cumulative correction); polar [-pi,pi] rad.',
        object='Exact signed object refit at each recorded phase; fixed final-object percentile threshold; shared Greys_r scale.'))
    print(output/'animations.html',flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',dest='case_folder',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path);parser.add_argument('--stride',type=int,default=1)
    parser.add_argument('--percentile',type=float,default=98.)
    parser.add_argument('--dpi',type=int,default=150);parser.add_argument('--interval',type=float,default=700.)
    export(**vars(parser.parse_args()))
