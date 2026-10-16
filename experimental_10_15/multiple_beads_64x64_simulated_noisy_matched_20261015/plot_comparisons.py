"""Export every experimental/noisy simulation pair in one PDF."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experimental_10_15.bead_pd_io import load_acquisition,optics_parameters
from experimental_10_15.bead_pd import acquisition_grid
from experimental_10_15.plot_many_bead_images import SOURCE,image_label
from experimental_10_15.reference_bead_simulation import FLAGGED
from systematic_testing.correction_examples.plot_pd_correction_examples import plot_acquisition_comparison
from utils import psf,zernike


def main():
    here=Path(__file__).resolve().parent
    acquisition=load_acquisition(SOURCE)
    noisy=load_acquisition(here/'simulated_acquisition')
    with np.load(here.parent/'reference_bead_orientation_scan/flip_x/comparison.npz') as data:
        measured=data['measured'].copy()
        ids=data['source_indices'].copy()
    simulated=noisy.images[ids]
    microscope=psf.Microscope(**optics_parameters(acquisition.optics,64))
    grid=acquisition_grid(acquisition.optics['fieldOfViewUm'],(64,64))
    groups={}
    for row,i in enumerate(ids):
        key=str(acquisition.names[i]).removesuffix('_before').removesuffix('_after')
        groups.setdefault(key,[]).append(row)
    output=here/'experimental_vs_noisy/all_images.pdf'
    output.parent.mkdir(exist_ok=True)
    with PdfPages(output) as pdf:
        for name,rows in groups.items():
            peak=max(float(measured[rows].max()),float(simulated[rows].max()))
            for start in range(0,len(rows),4):
                chosen=np.array(rows[start:start+4]);indices=ids[chosen]
                commands=[zernike.Aberration(acquisition.modes,acquisition.total[i]) for i in indices]
                numbers=', '.join(f'{i+1:03d}'+('*' if i+1 in FLAGGED else '') for i in indices)
                title=image_label(acquisition,indices[0]).replace(' before','')+'\nImages '+numbers+' (* excluded from optimization)'
                fig,_=plot_acquisition_comparison(grid,measured[chosen]/peak,simulated[chosen]/peak,
                    commands,acquisition.focal_z[indices],microscope.alpha,title=title,
                    column_titles=('Experimental (X mirrored)','Noise-matched simulation'))
                pdf.savefig(fig,dpi=300);plt.close(fig)
            print('Completed '+name,flush=True)
    print(f'{len(ids)} image pairs: {output}',flush=True)


if __name__=='__main__':main()
