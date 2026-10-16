"""Reference-driven lateral drift correction for MATLAB bead acquisitions.

Default: register reference_after to reference_before within each Z plane and
interpolate the applied translation through that plane's acquisition sequence.
Diversity frames never determine shifts. Raw acquisition files remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import shutil

import numpy as np
from scipy.io import loadmat,savemat
from scipy.ndimage import gaussian_filter,shift as translate
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from PIL import Image

from experimental_10_15.bead_pd_io import load_acquisition,write_json,_records


def _edge_median(image):
    return float(np.median(np.concatenate([image[0],image[-1],image[:,0],image[:,-1]])))


def _registration_image(image,sigma):
    # Background removal and smoothing affect shift estimation only.
    result=gaussian_filter(np.asarray(image,dtype=float),sigma)
    result-=np.mean(result)
    if not np.all(np.isfinite(result)) or np.linalg.norm(result) <= np.finfo(float).tiny:
        raise ValueError('Reference image has no finite spatial contrast.')
    return result


def _ncc(reference,moving):
    a=reference-reference.mean(); b=moving-moving.mean()
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/denominator) if denominator else 0.


def register_translation(reference,moving,*,max_shift=8.,sigma=1.,min_correlation=.4):
    """Return translation to APPLY to moving in model [x(column),y(row)] pixels.

    Linear, zero-padded correlation supplies the integer start. Bounded subpixel
    refinement maximizes normalized correlation on a fixed interior overlap.
    No circular wrapping, rotation, scaling or deformation is fitted.
    """
    reference=np.asarray(reference,dtype=float);moving=np.asarray(moving,dtype=float)
    if reference.shape != moving.shape or reference.ndim != 2:
        raise ValueError('Registration requires matching 2D reference images.')
    if (not np.isfinite(max_shift) or max_shift <= 0 or not np.isfinite(sigma) or sigma < 0
            or not np.isfinite(min_correlation) or not -1 <= min_correlation <= 1):
        raise ValueError('Invalid registration bounds, smoothing or correlation threshold.')
    margin=int(np.ceil(max_shift))+1
    if min(reference.shape) <= 2*margin+4:
        raise ValueError('max_shift is too large for this image; reduce --max-shift-pixels.')
    a=_registration_image(reference,sigma);b=_registration_image(moving,sigma)
    core=(slice(margin,-margin),slice(margin,-margin))
    corr=fftconvolve(a,b[::-1,::-1],mode='full')
    radius=int(np.floor(max_shift));center=np.array(a.shape)-1
    region=corr[center[0]-radius:center[0]+radius+1,center[1]-radius:center[1]+radius+1]
    initial=np.array(np.unravel_index(np.argmax(region),region.shape),dtype=float)-radius
    def score(offset):
        shifted=translate(b,offset,order=1,mode='constant',cval=0.,prefilter=False)
        return _ncc(a[core],shifted[core])
    before=score(np.zeros(2))
    result=minimize(lambda offset:-score(offset),initial,method='Powell',
        bounds=[(-max_shift,max_shift)]*2,options=dict(xtol=.005,ftol=1e-8,maxiter=100))
    # Keep the best of zero, correlation peak and refinement: minimize() may
    # otherwise pick a secondary maximum on a repeated bead pattern.
    choices=[np.zeros(2),initial,result.x]
    values=[score(value) for value in choices]
    best=int(np.argmax(values));offset=np.asarray(choices[best]);after=values[best]
    if np.any(np.abs(offset) >= max_shift-.05):
        raise ValueError(f'Registration reached the {max_shift:g}-pixel search bound; inspect references or increase it.')
    if after < min_correlation:
        raise ValueError(f'Reference correlation {after:.3f} is below {min_correlation:g}; registration rejected.')
    return offset,dict(correlation_before=before,correlation_after=after)


def _source_log(run,count):
    """Retain original log fields where scipy can read them; v7.3 uses index timing."""
    try:
        saved=loadmat(Path(run)/'acquisition.mat',simplify_cells=True)
        records=_records(saved.get('log',[]))
    except NotImplementedError:
        records=[]
    return [dict(records[i]) if i < len(records) else {} for i in range(count)]


def _interpolation_positions(log,indices,before,after):
    try:
        stamps=[datetime.strptime(str(log[i]['finishedAt']),'%Y%m%dT%H%M%S').timestamp()
                for i in indices]
        stamps=np.asarray(stamps);span=stamps[-1]-stamps[0]
        if span <= 0 or np.any(np.diff(stamps)<=0):raise ValueError('Invalid reference timestamps')
        return (stamps-stamps[0])/span,'finishedAt timestamps (second resolution)'
    except (KeyError,ValueError,TypeError):
        return (indices-before)/(after-before),'acquisition index (timestamps absent or unusable)'


def estimate_shifts(acquisition,log,*,align_planes=False,anchor_plane=None,max_shift=8.,sigma=1.,min_correlation=.4):
    """Estimate all shifts exclusively from paired baseline-reference frames."""
    a=acquisition;active=np.flatnonzero(a.acquired)
    if not len(active):raise ValueError('No acquired images remain after exclusions.')
    planes=np.unique(a.plane_index[active]);pairs={}
    for plane in planes:
        ids=np.flatnonzero(a.acquired & (a.plane_index==plane))
        before=ids[(a.names[ids]=='reference_before') & ~a.is_pd[ids]]
        after=ids[(a.names[ids]=='reference_after') & ~a.is_pd[ids]]
        if len(before)!=1 or len(after)!=1:
            raise ValueError(f'Plane {plane} needs acquired reference_before and reference_after images; exclude incomplete planes explicitly.')
        first,last=int(before[0]),int(after[0])
        if first >= last or np.any(ids < first) or np.any(ids > last):
            raise ValueError(f'Plane {plane} references must bracket all acquired images.')
        if not np.allclose(a.total[first],a.total[last],atol=1e-12,rtol=0):
            raise ValueError(f'Plane {plane} reference phases differ.')
        if not np.allclose(a.focal_z[ids],a.focal_z[first],atol=1e-12,rtol=0):
            raise ValueError(f'Plane {plane} has inconsistent focus metadata.')
        pairs[int(plane)]=(first,last,ids)
    if anchor_plane is None:
        anchor_plane=int(min(planes,key=lambda p:abs(a.focal_z[pairs[int(p)][0]])))
    if anchor_plane not in pairs:raise ValueError('Requested anchor plane is missing or excluded.')
    anchor=pairs[anchor_plane][0];offsets=np.zeros((len(a.images),2));reports=[]
    for plane,(first,last,ids) in pairs.items():
        try:
            relative,quality=register_translation(a.images[first],a.images[last],
                max_shift=max_shift,sigma=sigma,min_correlation=min_correlation)
            base=np.zeros(2);across_quality=None
            if align_planes and plane != anchor_plane:
                if not np.allclose(a.total[anchor],a.total[first],atol=1e-12,rtol=0):
                    raise ValueError('Cross-plane references have different applied phases.')
                base,across_quality=register_translation(a.images[anchor],a.images[first],
                    max_shift=max_shift,sigma=sigma,min_correlation=min_correlation)
        except ValueError as exc:
            raise ValueError(f'Plane {plane}: {exc}') from exc
        fraction,timing=_interpolation_positions(log,ids,first,last)
        offsets[ids]=base+fraction[:,None]*relative
        reports.append(dict(plane_index=plane,reference_before_index=first,reference_after_index=last,
            reference_shift_xy_pixels=relative,plane_shift_xy_pixels=base,
            interpolation_time_source=timing,within_plane=quality,between_planes=across_quality))
    return offsets,reports,anchor_plane


def align_run(run,output,*,exclude_planes=None,align_planes=False,anchor_plane=None,
              max_shift_pixels=8.,smoothing_sigma=1.,min_correlation=.4,preview=False,use_tex=None):
    """Write an independent acquisition folder consumable by existing PD tools."""
    run=Path(run).resolve();output=Path(output).resolve()
    if output==run or output in run.parents:
        raise ValueError('Output must be separate from the source acquisition.')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Alignment output must be new or empty; original data will not be overwritten.')
    meta=loadmat(run/'metadata.mat',simplify_cells=True)['metadata']
    if 'alignment' in meta:
        raise ValueError('This is already an aligned acquisition. Restart from the raw folder to avoid repeated resampling.')
    if anchor_plane is not None and not align_planes:
        raise ValueError('--anchor-plane requires --align-planes.')
    a=load_acquisition(run,exclude_planes=exclude_planes)
    log=_source_log(run,len(a.images))
    offsets,reports,anchor=estimate_shifts(a,log,align_planes=align_planes,anchor_plane=anchor_plane,
        max_shift=max_shift_pixels,sigma=smoothing_sigma,min_correlation=min_correlation)
    aligned=np.full_like(a.images,np.nan);valid=np.zeros_like(a.images,dtype=bool)
    x,y=np.indices(a.images.shape[-2:]);fills=np.full(len(a.images),np.nan)
    for i in np.flatnonzero(a.acquired):
        fills[i]=_edge_median(a.images[i])
        aligned[i]=translate(a.images[i],offsets[i],order=1,mode='constant',cval=fills[i],prefilter=False)
        valid[i]=(x-offsets[i,0]>=0)&(x-offsets[i,0]<=a.images.shape[1]-1)&(
            y-offsets[i,1]>=0)&(y-offsets[i,1]<=a.images.shape[2]-1)
        log[i].update(name=str(a.names[i]),acquired=True,alignmentShiftXY=offsets[i],
            alignmentSourceIndex=int(i),filename=f'Aligned_{i+1:03d}_{a.names[i]}.tif')
    for i in np.flatnonzero(~a.acquired):log[i].update(name=str(a.names[i]),acquired=False)
    def digest(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    report=dict(source=str(run),coordinate_order='x=column, y=row; shifts are applied translations',
        units='source image pixels',mode='across planes + within-plane drift' if align_planes else 'within-plane drift only',
        anchor_plane=anchor if align_planes else None,excluded_planes=list(exclude_planes or []),
        registration='smoothed-reference normalized correlation; bounded subpixel translation',
        interpolation='linear drift between reference centers, using finishedAt timestamps when usable',
        resampling='one linear interpolation of original floating data per image; no intensity normalization',
        padding='per-image median border intensity; no circular wrapping; original FOV and pixel spacing retained',
        limitations='XY drift only; cannot infer nonlinear motion or motion during averaging from two references. Cross-Z registration can confuse morphology changes with motion.',
        source_sha256={name:digest(run/name) for name in ['metadata.mat','acquisition.mat']},
        shifts_xy_pixels=offsets,planes=reports)
    output.mkdir(parents=True,exist_ok=True)
    # Original full metadata, coefficient order, FOV and phase settings survive.
    meta['alignment']=dict(source=str(run),report='alignment.json',method=report['mode'],
                           shiftsXY=offsets,excludedPlanes=np.asarray(exclude_planes or [],dtype=int))
    meta['status']='aligned'
    savemat(output/'metadata.mat',dict(metadata=meta),long_field_names=True,do_compression=True)
    savemat(output/'acquisition.mat',dict(images=aligned.transpose(2,1,0),log=np.array(log,dtype=object)),
            long_field_names=True,do_compression=True)
    shutil.copy2(run/'metadata.mat',output/'source_metadata.mat')
    np.savez_compressed(output/'alignment.npz',shifts_xy_pixels=offsets,valid_mask=valid,
        common_valid_mask=np.all(valid[a.acquired],axis=0),acquired=a.acquired,
        source_indices=np.arange(len(a.images)),padding_values=fills)
    fov=float(a.optics['fieldOfViewUm']);spacing=fov/np.array(a.images.shape[-2:])
    rows=np.column_stack([np.arange(len(a.images)),a.plane_index,a.acquired,offsets,offsets*spacing])
    np.savetxt(output/'alignment_shifts.csv',rows,delimiter=',',comments='',
        header='source_index,plane_index,acquired,shift_x_pixels,shift_y_pixels,shift_x_um,shift_y_um',
        fmt=['%d','%d','%d','%.8g','%.8g','%.8g','%.8g'])
    for i in np.flatnonzero(a.acquired):
        tiff_preview=np.uint16(np.clip(aligned[i].T,0,65535).round())
        Image.fromarray(tiff_preview).save(output/log[i]['filename'])
    write_json(output/'alignment.json',report)
    if preview:render_alignment_preview(run,output,use_tex=use_tex)
    print(f'Aligned {np.count_nonzero(a.acquired)} images into {output}',flush=True)
    return output



def render_alignment_preview(run,output,*,use_tex=None):
    """Inspect reference pairs with the existing acquired-plane plotting helper."""
    import matplotlib.pyplot as plt
    from experimental_10_15.bead_pd import acquisition_grid
    from systematic_testing.correction_examples.plot_pd_correction_examples import plot_acquisition_comparison
    from utils.zernike import Aberration
    raw=load_acquisition(run);aligned=load_acquisition(output)
    if use_tex is None:use_tex=bool(raw.config.get('analysis',{}).get('useTex',True))
    pairs=[]
    for plane in np.unique(aligned.plane_index[aligned.acquired]):
        ids=np.flatnonzero(aligned.acquired & (aligned.plane_index==plane))
        first=ids[raw.names[ids]=='reference_before'][0]
        last=ids[raw.names[ids]=='reference_after'][0]
        pairs.append((first,last))
    pairs=np.asarray(pairs);references=pairs.ravel()
    scale=max(float(raw.images[references].max()),float(aligned.images[references].max()),np.finfo(float).tiny)
    grid=acquisition_grid(raw.optics['fieldOfViewUm'],raw.images.shape[-2:])
    alpha=float(np.arcsin(raw.optics['NA']/raw.optics['refractiveIndex']))
    folder=Path(output)/'figures';folder.mkdir(exist_ok=True)
    for label,stack in [('original',raw.images),('aligned',aligned.images)]:
        for page,start in enumerate(range(0,len(pairs),4),1):
            selected=pairs[start:start+4];before,after=selected.T
            commands=[Aberration(raw.modes,raw.total[i]) for i in before]
            fig,_=plot_acquisition_comparison(grid,stack[before]/scale,stack[after]/scale,
                commands,raw.focal_z[before],alpha,title=f'Reference pairs: {label}',use_tex=use_tex,
                inset_loc='lower right',column_titles=('Reference before','Reference after'))
            for extension in ['png','pdf']:
                fig.savefig(folder/f'references_{label}_page_{page:02d}.{extension}',dpi=300)
            plt.close(fig)
    return folder

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview',action='store_true',help='Export original/aligned reference-pair figures')
    p.add_argument('--no-tex',action='store_true',help='Disable LaTeX for preview figures explicitly')
    p.add_argument('--exclude-plane',dest='exclude_planes',type=int,action='append')
    p.add_argument('--align-planes',action='store_true',help='Also register before-references across Z; inspect morphology-dependent errors')
    p.add_argument('--anchor-plane',type=int,help='1-based common anchor for --align-planes; default closest to focus')
    p.add_argument('--max-shift-pixels',type=float,default=8.)
    p.add_argument('--smoothing-sigma',type=float,default=1.)
    p.add_argument('--min-correlation',type=float,default=.4)
    options=vars(p.parse_args())
    options['use_tex']=False if options.pop('no_tex') else None
    align_run(**options)


if __name__=='__main__':main()
