"""Reference-only bead refits for all eight global camera/SLM axis hypotheses."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experimental_10_15.reference_bead_simulation import (compare,render_comparison,orient_images,ORIENTATIONS,FLAGGED)
from experimental_10_15.plot_many_bead_images import SOURCE
from experimental_10_15.bead_pd_io import load_acquisition,block_average,optics_parameters,write_json
from experimental_10_15.bead_pd import acquisition_grid
from systematic_testing.correction_examples.plot_pd_correction_examples import plot_acquisition_comparison
from utils import psf,zernike

ROOT=Path(__file__).resolve().parent


def scan(output=ROOT/'reference_bead_orientation_scan'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    acquisition=load_acquisition(SOURCE);original=block_average(acquisition.images,64)
    folders={};scores=[]
    for orientation in ORIENTATIONS:
        folder=(ROOT/'reference_bead_simulation' if orientation=='identity' else
                ROOT/'reference_bead_simulation_transposed' if orientation=='transpose' else output/orientation)
        if not (folder/'comparison.npz').is_file():
            print('Testing '+orientation,flush=True)
            compare(output=folder,orientation=orientation,make_figures=False)
        report=json.loads((folder/'comparison.json').read_text())
        with np.load(folder/'comparison.npz') as data:
            ids=data['source_indices'];measured=data['measured'];predicted=data['simulated']
            np.testing.assert_array_equal(measured,orient_images(original[ids],orientation))
            keep=acquisition.is_pd[ids]&~np.isin(ids+1,FLAGGED)
            assert keep.sum()==134 and len(report['calibration']['reference_indices'])==18
            assert not acquisition.is_pd[np.asarray(report['calibration']['reference_indices'])].any()
            error=float(np.sqrt(np.mean((measured[keep]-predicted[keep])**2)))
            correlations=[]
            for a,b in zip(measured[keep],predicted[keep]):
                a=a.ravel()-a.mean();b=b.ravel()-b.mean()
                correlations.append(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
            row=dict(orientation=orientation,unflagged_pd_images=int(keep.sum()),pd_rmse=error,
                reference_rmse=report['calibration']['reference_fit_rmse'],mean_pd_correlation=float(np.mean(correlations)),
                flagged_112_rmse=float(np.sqrt(np.mean((measured[np.flatnonzero(ids==111)[0]]-predicted[np.flatnonzero(ids==111)[0]])**2))),
                flagged_130_rmse=float(np.sqrt(np.mean((measured[np.flatnonzero(ids==129)[0]]-predicted[np.flatnonzero(ids==129)[0]])**2))),
                folder=str(folder.resolve()))
            scores.append(row);folders[orientation]=folder
            print(row,flush=True)
        write_json(output/'scores.json',scores)
    scores.sort(key=lambda row:row['pd_rmse'])
    with (output/'scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
    best=scores[0]['orientation'];baseline=next(row['pd_rmse'] for row in scores if row['orientation']=='identity')
    result=dict(best_orientation=best,pd_rmse_improvement_percent=100*(1-scores[0]['pd_rmse']/baseline),
        selection='Aggregate pixel RMSE over 134 unflagged PD images; no PD-dependent refit or intensity adjustment.',
        calibration='Refit 3D bead centers, per-bead brightness and global background to references only for each transform.',
        transform_convention='Arrays are image,x,y; flips act on model x/y; rotations use np.rot90 on the last two array axes.',
        commands='Optical phase coefficients and Z values unchanged for every hypothesis.',
        scores=scores,best_folder=str(folders[best].resolve()))
    write_json(output/'summary.json',result)
    # The requested examples plus two unflagged frames of the identical probe.
    chosen_numbers=[76,94,112,130]
    grid=acquisition_grid(acquisition.optics['fieldOfViewUm'],(64,64))
    microscope=psf.Microscope(**optics_parameters(acquisition.optics,64))
    examples=list(dict.fromkeys(['identity','transpose',best]))
    with PdfPages(output/'orientation_examples.pdf') as document:
        for orientation in examples:
            with np.load(folders[orientation]/'comparison.npz') as data:
                ids=data['source_indices'];rows=np.array([np.flatnonzero(ids==i-1)[0] for i in chosen_numbers])
                images=data['measured'];predicted=data['simulated']
                same_probe=acquisition.names[ids]==acquisition.names[chosen_numbers[0]-1]
                peak=max(images[same_probe].max(),predicted[same_probe].max())
                labels=', '.join(str(i)+('*' if i in FLAGGED else '') for i in chosen_numbers)
                commands=[zernike.Aberration(acquisition.modes,acquisition.total[i-1]) for i in chosen_numbers]
                fig,_=plot_acquisition_comparison(grid,images[rows]/peak,predicted[rows]/peak,commands,
                    acquisition.focal_z[np.asarray(chosen_numbers)-1],microscope.alpha,
                    title=orientation.replace('_',' ')+': images '+labels+'\n* flagged artifacts; excluded from orientation scoring')
                document.savefig(fig,dpi=300)
                fig.savefig(output/('examples_'+orientation+'.png'),dpi=300);plt.close(fig)
    if not (folders[best]/'measured_vs_simulated.pdf').is_file():render_comparison(folders[best])
    print('Best tested orientation: '+best+'; '+str(folders[best]/'measured_vs_simulated.pdf'),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reference_bead_orientation_scan')
    scan(**vars(parser.parse_args()))
