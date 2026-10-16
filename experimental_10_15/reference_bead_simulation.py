"""Infer fixed 1-um spheres from reference images and replay recorded commands.

Reference-only geometry/brightness calibration; no aberrated image enters the
sample fit. System AO is assumed to cancel native phase, as in existing replays.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy import fft,ndimage,optimize
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from experimental_10_15.bead_pd import acquisition_grid
from experimental_10_15.bead_pd_io import load_acquisition,block_average,optics_parameters,write_json
from experimental_10_15.plot_many_bead_images import SOURCE,image_label
from optimization._reconstruction_common import _derivative_geometry
from systematic_testing.correction_examples.plot_pd_correction_examples import plot_acquisition_comparison
from utils import psf,zernike

FLAGGED=[2,20,57,4,112,130,24,44,116,135]
ORIENTATIONS=('identity','flip_x','flip_y','rotate_180','transpose','rotate_90','rotate_270','anti_transpose')


def orient_images(images,orientation):
    """Apply a lossless global array transform; input order is image,x,y."""
    if orientation=='identity':return images.copy()
    if orientation=='flip_x':return images[:,::-1,:].copy()
    if orientation=='flip_y':return images[:,:,::-1].copy()
    if orientation=='rotate_180':return images[:,::-1,::-1].copy()
    if orientation=='transpose':return images.swapaxes(-1,-2).copy()
    if orientation=='rotate_90':return np.rot90(images,1,axes=(-2,-1)).copy()
    if orientation=='rotate_270':return np.rot90(images,3,axes=(-2,-1)).copy()
    if orientation=='anti_transpose':return images.swapaxes(-1,-2)[:,::-1,::-1].copy()
    raise ValueError('Unknown image orientation: '+str(orientation))


class SphereModel:
    """Cached vector propagation and linear convolution of a finite sphere.

    Equivalent to compute_image_3D for a centered sphere. The optimized ZoomFFT
    propagator shares pupil geometry with the solver. Translation uses native
    pixels before block means; no periodic wrapping or per-frame normalization.
    """
    def __init__(self,microscope,grid,modes,*,diameter_um=1.,planes=21,size=64):
        self.microscope=microscope;self.grid=grid;self.modes=modes;self.size=size
        self.shape=(grid.grid_ffp_x,grid.grid_ffp_y)
        self.spacing_um=(grid.get_xy()[0][1]-grid.get_xy()[0][0])*1000
        levels=np.linspace(-diameter_um/2,diameter_um/2,planes)
        sphere=psf.bead_img_3D(grid,levels/1000,[0],[0],[0],[diameter_um/2000])
        disks=sphere.image_mask;active=np.any(disks!=0,axis=(1,2))
        self.levels=levels[active];disks=disks[active]
        kernel_grid=psf._convolution_grid(grid)
        self.kernel_shape=(kernel_grid.grid_ffp_x,kernel_grid.grid_ffp_y)
        self.fft_shape=tuple(fft.next_fast_len(a+b-1) for a,b in zip(self.shape,self.kernel_shape))
        self.disk_ft=fft.rfft2(disks,s=self.fft_shape)
        self.crop=tuple(slice((k-1)//2,(k-1)//2+n) for n,k in zip(self.shape,self.kernel_shape))
        self.amplitudes,_,_,_,self.sz,self.basis,self.propagate=_derivative_geometry(
            microscope,kernel_grid,modes,'vector',propagation='zoom')
        self.basis=np.asarray(self.basis)

    def templates(self,coefficients,focus_um):
        distances=self.levels[None,:]-np.asarray(focus_um)[:,None]
        unique,inverse=np.unique(np.round(distances,10),return_inverse=True)
        lookup=inverse.reshape(distances.shape)
        common=np.exp(1j*np.tensordot(coefficients,self.basis,axes=1))
        kernels=np.empty((len(unique),self.fft_shape[0],self.fft_shape[1]//2+1),dtype=complex)
        for start in range(0,len(unique),8):
            levels=unique[start:start+8]
            pupil=(np.exp(1j*self.microscope.k*levels[:,None,None]/1000*self.sz)[...,None,:,:]
                   *common*self.amplitudes)
            field=self.propagate(pupil)
            intensity=np.sum(abs(field)**2,axis=1)**self.microscope.N_order
            kernels[start:start+len(levels)]=fft.rfft2(intensity,s=self.fft_shape)
        images=[]
        for indices in lookup:
            spectrum=np.sum(kernels[indices]*self.disk_ft,axis=0)
            images.append(fft.irfft2(spectrum,s=self.fft_shape)[self.crop])
        return np.maximum(np.asarray(images),0)

    def shift_bin(self,images,x_um,y_um):
        shifted=ndimage.shift(images,(0,x_um/self.spacing_um,y_um/self.spacing_um),
                              order=1,mode='constant',cval=0,prefilter=False)
        return block_average(shifted,self.size)


def infer_beads(acquisition,model):
    references=np.flatnonzero(acquisition.acquired&~acquisition.is_pd)
    if len(references)<3:raise ValueError('Need a reference Z stack.')
    commands=acquisition.total[references]
    if not np.allclose(commands,commands[0],rtol=0,atol=1e-12):
        raise ValueError('Reference commands differ; a single reference template is insufficient.')
    reference_images=block_average(acquisition.images[references],model.size)
    z_values=np.unique(acquisition.focal_z[references]*1000)
    focus=acquisition.focal_z[references]*1000
    pooled=np.array([reference_images[np.isclose(focus,z)].mean(axis=0) for z in z_values])
    x,y=acquisition_grid(acquisition.optics['fieldOfViewUm'],pooled.shape[-2:]).get_xy()
    x=x*1000;y=y*1000;spacing=x[1]-x[0]
    projection=ndimage.gaussian_filter(pooled.max(axis=0),1.)
    separation=max(3,int(np.ceil(.8/spacing))|1)
    maxima=(projection==ndimage.maximum_filter(projection,size=separation))&(projection>.2*projection.max())
    components,count=ndimage.label(maxima)
    peaks=np.array([np.unravel_index(np.argmax(np.where(components==i,projection,-np.inf)),projection.shape)
                    for i in range(1,count+1)])
    if not 1<=len(peaks)<=6:raise ValueError(f'Detected {len(peaks)} beads; inspect reference segmentation.')
    # Fit one static sample to both reference repeats; do not align any data.
    edges=np.concatenate([pooled[:,0,:].ravel(),pooled[:,-1,:].ravel(),pooled[:,:,0].ravel(),pooled[:,:,-1].ravel()])
    background=float(np.median(edges));noise=max(float(np.median(abs(edges-background))/.6744897501960817),1.)
    initial=[]
    for ix,iy in peaks:
        aperture=(x[:,None]-x[ix])**2+(y[None,:]-y[iy])**2<.55**2
        axial=(pooled[:,aperture]-background).sum(axis=1)
        initial.extend([x[ix],y[iy],z_values[np.argmax(axial)],np.log(max(projection[ix,iy]-background,1.))])
    initial=np.r_[initial,background];n=len(peaks)
    q=np.linspace(z_values.min()-2.5,z_values.max()+2.5,181)
    print(f'Building reference sphere templates; detected {n} beads.',flush=True)
    library=model.templates(commands[0],q)
    scale=float(library.max());library/=scale

    def prediction(parameters):
        beads=parameters[:-1].reshape(n,4)
        result=np.zeros_like(pooled)
        for xc,yc,zc,log_brightness in beads:
            locations=np.clip((z_values-zc-q[0])/(q[1]-q[0]),0,len(q)-1)
            lower=np.minimum(np.floor(locations).astype(int),len(q)-2);weight=locations-lower
            images=library[lower]*(1-weight[:,None,None])+library[lower+1]*weight[:,None,None]
            result+=np.exp(log_brightness)*model.shift_bin(images,xc,yc)
        return result+parameters[-1]

    lower=[];upper=[]
    for xc,yc,zc,log_brightness in np.asarray(initial[:-1]).reshape(n,4):
        lower.extend([max(x.min(),xc-.5),max(y.min(),yc-.5),-2.5,np.log(1.)])
        upper.extend([min(x.max(),xc+.5),min(y.max(),yc+.5),2.5,np.log(float(pooled.max())*20)])
    fit=optimize.least_squares(lambda p:(prediction(p)-pooled).ravel(),initial,
        bounds=(np.r_[lower,-np.inf],np.r_[upper,np.inf]),loss='soft_l1',f_scale=3*noise,
        x_scale='jac',max_nfev=150,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    print('Reference fit:',fit.message,flush=True)
    if not fit.success:raise ValueError('Reference geometry fit did not converge: '+fit.message)
    beads=fit.x[:-1].reshape(n,4)
    return dict(beads=beads,background=float(fit.x[-1]),physical_scale=scale,reference_indices=references,
        pooled_reference_z_um=z_values,reference_noise_estimate=noise,reference_fit_rmse=float(np.sqrt(np.mean((prediction(fit.x)-pooled)**2))),
        optimizer_success=bool(fit.success),optimizer_message=fit.message,nfev=fit.nfev,
        initial_parameters=initial,template_z_spacing_um=float(q[1]-q[0]))


def compare(run=SOURCE,output=None,size=64,transpose_images=False,orientation='identity',make_figures=True):
    if transpose_images:
        if orientation not in ('identity','transpose'):raise ValueError('Conflicting transpose/orientation options.')
        orientation='transpose'
    if output is None:
        suffix='' if orientation=='identity' else '_transposed' if orientation=='transpose' else '_'+orientation
        output=Path(__file__).resolve().parent/('reference_bead_simulation'+suffix)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    acquisition=load_acquisition(run);ids=np.flatnonzero(acquisition.acquired)
    acquisition.images=orient_images(acquisition.images,orientation)
    transpose_images=orientation=='transpose'
    native_grid=acquisition_grid(acquisition.optics['fieldOfViewUm'],acquisition.images.shape[-2:])
    microscope=psf.Microscope(**optics_parameters(acquisition.optics,64))
    model=SphereModel(microscope,native_grid,acquisition.modes,size=size)
    calibration=infer_beads(acquisition,model)
    calibration['measured_images_transposed']=bool(transpose_images)
    calibration['orientation_key']=orientation
    beads=calibration['beads'];scale=calibration['physical_scale'];background=calibration['background']
    write_json(output/'reference_fit.json',calibration)
    with (output/'bead_centers.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['bead','x_um','y_um','z_um','diameter_um','relative_brightness'])
        for i,(x,y,z,brightness) in enumerate(beads,1):writer.writerow([i,x,y,z,1.,np.exp(brightness)])
    measured=block_average(acquisition.images[ids],size);simulated=np.empty_like(measured)
    command_groups={}
    for row,i in enumerate(ids):command_groups.setdefault(tuple(acquisition.total[i]),[]).append(row)
    for number,(coefficients,rows) in enumerate(command_groups.items(),1):
        z=acquisition.focal_z[ids[rows]]*1000
        unique_z,inverse=np.unique(z,return_inverse=True)
        focus=np.concatenate([unique_z-bead[2] for bead in beads])
        templates=model.templates(np.asarray(coefficients),focus)/scale
        combined=np.zeros((len(unique_z),size,size))
        for i,(x,y,zc,brightness) in enumerate(beads):
            combined+=np.exp(brightness)*model.shift_bin(templates[i*len(unique_z):(i+1)*len(unique_z)],x,y)
        simulated[rows]=combined[inverse]+background
        print(f'Simulated command {number}/{len(command_groups)}',flush=True)
    rmse=np.sqrt(np.mean((measured-simulated)**2,axis=(1,2)))
    with (output/'image_metrics.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['image_number','name','z_um','reference','flagged_artifact','rmse','measured_peak','simulated_peak'])
        for row,i in enumerate(ids):
            writer.writerow([i+1,acquisition.names[i],acquisition.focal_z[i]*1000,not acquisition.is_pd[i],
                             i+1 in FLAGGED,rmse[row],measured[row].max(),simulated[row].max()])
    np.savez_compressed(output/'comparison.npz',measured=measured,simulated=simulated,source_indices=ids,
        focal_z=acquisition.focal_z[ids],names=acquisition.names[ids],rmse=rmse,
        bead_parameters=beads,background=background,measured_images_transposed=bool(transpose_images),orientation_key=orientation)
    flagged=np.isin(ids+1,FLAGGED)
    report=dict(source=Path(run).resolve(),image_count=len(ids),size=size,bead_diameter_um=1.,
        measured_images_transposed=bool(transpose_images),
        orientation_key=orientation,orientation='Global measured-array transform: '+orientation+'; applied before reference fit; optical modes and commands unchanged.',
        calibration=calibration,flagged_image_numbers=FLAGGED,
        fit_used='18 reference images only; no PD images or estimated aberrations used to fit positions/brightness.',
        model='Finite uniform 1-um spheres; recorded commanded Zernike phase; native128 simulation then block mean64; vector 3P optics.',
        system_ao='Fixed system AO assumed to cancel native aberration; no residual phase fitted.',
        spatial_registration='One static sample; no measured images shifted or aligned.',
        gain='One fixed brightness per bead and one constant signed background, fitted to references only.',
        rmse=rmse,retained_pd_rmse=float(np.sqrt(np.mean((measured[(acquisition.is_pd[ids])&~flagged]-simulated[(acquisition.is_pd[ids])&~flagged])**2))),
        display='Measured/simulated share one normalization within each aberration type across all Z; flagged images included and marked *.',
        interpretation='A reference-constrained model prediction, not an exact/noise-matched copy. Finite sampling, model mismatch and drift may remain.')
    write_json(output/'comparison.json',report)
    if not make_figures:return output
    return render_comparison(output)


def render_comparison(output):
    output=Path(output)
    import json
    report=json.loads((output/'comparison.json').read_text())
    acquisition=load_acquisition(report['source']);size=report['size']
    microscope=psf.Microscope(**optics_parameters(acquisition.optics,64))
    with np.load(output/'comparison.npz') as data:
        measured=data['measured'];simulated=data['simulated'];ids=data['source_indices']
    figures=output/'figures';figures.mkdir(exist_ok=True)
    grid=acquisition_grid(acquisition.optics['fieldOfViewUm'],(size,size));groups={}
    for row,i in enumerate(ids):
        key=str(acquisition.names[i]).removesuffix('_before').removesuffix('_after')
        groups.setdefault(key,[]).append(row)
    with PdfPages(output/'measured_vs_simulated.pdf') as pdf:
        for group_number,(name,rows) in enumerate(groups.items(),1):
            peak=max(float(measured[rows].max()),float(simulated[rows].max()))
            for page,start in enumerate(range(0,len(rows),4),1):
                chosen=np.array(rows[start:start+4]);indices=ids[chosen]
                commands=[zernike.Aberration(acquisition.modes,acquisition.total[i]) for i in indices]
                numbers=', '.join(f'{i+1:03d}'+('*' if i+1 in FLAGGED else '') for i in indices)
                title=image_label(acquisition,indices[0]).replace(' before','')+'\nImages '+numbers+' (* flagged artifact)'
                if report.get('orientation_key','identity')!='identity':title+=' | '+report['orientation_key'].replace('_',' ')
                fig,_=plot_acquisition_comparison(grid,measured[chosen]/peak,simulated[chosen]/peak,commands,
                    acquisition.focal_z[indices],microscope.alpha,title=title)
                fig.savefig(figures/f'group_{group_number:02d}_page_{page:02d}.png',dpi=300)
                pdf.savefig(fig,dpi=300);plt.close(fig)
            print(f'Plotted comparison group {group_number}/{len(groups)}',flush=True)
    print(output/'measured_vs_simulated.pdf',flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--size',type=int,default=64)
    parser.add_argument('--transpose-images',action='store_true',help='Transpose measured x/y before fitting references and comparing; keep optical commands unchanged.')
    parser.add_argument('--orientation',choices=ORIENTATIONS,default='identity')
    compare(**vars(parser.parse_args()))
