"""Export acquisition contact sheets grouped by Z plane or aberration type."""
import argparse
import csv
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experimental_10_15.bead_pd_io import load_acquisition,optics_parameters,write_json,block_average
from experimental_10_15.bead_pd import acquisition_grid
from systematic_testing.correction_examples.plot_pd_correction_examples import plot_acquisition_gallery
from utils import psf,zernike

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'3d_many_fluorobeads_random_aligned_0p15w/tp76e42de4_ca3a_4172_9562_d6cf4b94918c'


def image_label(acquisition,i):
    name=str(acquisition.names[i]);probe=acquisition.diversities[i]
    active=np.flatnonzero(abs(probe)>1e-12)
    if not acquisition.is_pd[i]:return 'Reference after' if 'after' in name else 'Reference before'
    if not len(active):return 'Aberrated zero after' if 'after' in name else 'Aberrated zero before'
    return '; '.join(fr'$({acquisition.modes[j,0]:+d},{acquisition.modes[j,1]:d}),\ {probe[j]:+g}$ w' for j in active)


def export(run=SOURCE,output=ROOT/'many_bead_acquired_images',group_by='plane',size=None):
    acquisition=load_acquisition(run);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    images=block_average(acquisition.images,size)
    ids=np.flatnonzero(acquisition.acquired)
    peak=float(images[ids].max())
    if peak<=0:raise ValueError('Acquisition has no positive image signal.')
    grid=acquisition_grid(acquisition.optics['fieldOfViewUm'],images.shape[-2:])
    microscope=psf.Microscope(**optics_parameters(acquisition.optics,64))
    rows=[];planes=[];groups={}
    for i in ids:
        key=(int(acquisition.plane_index[i]) if group_by=='plane'
             else str(acquisition.names[i]).removesuffix('_before').removesuffix('_after'))
        groups.setdefault(key,[]).append(i)
    with PdfPages(output/('all_z_planes.pdf' if group_by=='plane' else 'all_aberration_types.pdf')) as document:
        for key,members in groups.items():
            selected=np.array(members,dtype=int)
            z_um=acquisition.focal_z[selected]*1000
            group_peak=peak if group_by=='plane' else float(images[selected].max())
            if group_peak<=0:raise ValueError(f'Group {key} has no positive image signal.')
            labels=[]
            for i in selected:
                label=image_label(acquisition,i)
                if group_by=='plane':labels.append(f'Image {i+1:03d}\n'+label)
                else:
                    suffix=('after' if 'after' in str(acquisition.names[i]) else 'before') if len(members)>9 else ''
                    labels.append(fr'$z={acquisition.focal_z[i]*1000:+g}$ $\mu$m '+suffix+f'\nImage {i+1:03d}')
                rows.append(dict(group=str(key),plane=int(acquisition.plane_index[i]),
                                 z_um=float(acquisition.focal_z[i]*1000),image_number=int(i+1),name=str(acquisition.names[i]),
                                 is_pd=bool(acquisition.is_pd[i])))
            commands=[zernike.Aberration(acquisition.modes,acquisition.total[i]) for i in selected]
            title=(fr'Many fluorobeads: $z={z_um[0]:+g}$ $\mu$m ({len(selected)} images)' if group_by=='plane'
                   else 'Many fluorobeads: '+image_label(acquisition,selected[0]).replace(' before','')+f' ({len(selected)} images)')
            if size is not None:title+=fr' [$ {size}\times {size} $]'
            fig,_=plot_acquisition_gallery(grid,images[selected]/group_peak,commands,labels,microscope.alpha,
                title=title,intensity_label='Intensity / '+('acquisition-wide' if group_by=='plane' else 'aberration-type')+' peak')
            stem=(f'plane_{key:02d}_z_{z_um[0]:+g}um' if group_by=='plane'
                  else 'aberration_'+re.sub(r'[^A-Za-z0-9_-]','_',key))
            for extension in ['png','pdf']:fig.savefig(output/(stem+'.'+extension),dpi=300)
            document.savefig(fig,dpi=300);plt.close(fig)
            planes.append(dict(group=str(key),z_um=np.unique(z_um),images=len(selected),stem=stem,normalization_peak=group_peak))
            print(stem,flush=True)
    with (output/'image_index.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(output/'gallery.json',dict(source=Path(run).resolve(),groups=planes,group_by=group_by,image_count=len(rows),
        normalization=('One positive peak across every acquired image at every Z; display floor zero.' if group_by=='plane'
                       else 'One positive peak within each aberration type across all Z and before/after repeats; display floor zero.'),
        image_shape=images.shape[-2:],downsampling='Nonoverlapping block means; unchanged FOV.',
        global_peak=peak,phase_insets='Metadata commanded Zernike phase; fixed raw system AO map is not included.',
        array_order='Model x,y arrays transposed once by the existing plotting helper.'))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=ROOT/'many_bead_acquired_images')
    parser.add_argument('--group-by',choices=['plane','aberration'],default='plane')
    parser.add_argument('--size',type=int,help='Bin square images to this size using block means.')
    export(**vars(parser.parse_args()))
