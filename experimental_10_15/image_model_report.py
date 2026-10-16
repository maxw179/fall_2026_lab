"""Static figures and evidence summary for the local model diagnostics."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experimental_10_15.bead_pd import acquisition_grid
from experimental_10_15.bead_pd_io import load_acquisition,write_json
from experimental_10_15.image_model_diagnostics import ROOT,CASE,SIM,load_case
from experimental_10_15.plot_many_bead_images import SOURCE
from experimental_10_15.reference_bead_simulation import FLAGGED
from systematic_testing.correction_examples.plot_pd_iteration_rms import plot_error_histories
from systematic_testing.correction_examples.plot_pd_correction_examples import plot_acquisition_comparison
from utils import psf,zernike


def report(output):
    output=Path(output);figures=output/'figures';figures.mkdir(exist_ok=True)
    audit=json.loads((output/'audit.json').read_text());acq=load_acquisition(SOURCE);case=load_case()
    ids=case['source_indices'];z=case['focal_z']*1000;levels=np.unique(z)
    nominal=json.loads((output/'sweeps/nominal.json').read_text())
    corrected=json.loads((output/'background_checks/border_10.json').read_text())
    sim=json.loads((output/'background_checks/simulation_border_subtracted.json').read_text())
    injected=json.loads((output/'background_checks/simulation_experimental_background.json').read_text())
    with np.load(output/'reference_measurements.npz') as data:measure={k:data[k].copy() for k in data.files}
    with np.load(ROOT/'reference_bead_orientation_scan/flip_x/comparison.npz') as data:prediction=data['simulated'].copy()
    with np.load(output/'truth_fit.npz') as data:raw_residual=data['residual'].copy()
    with np.load(output/'estimate_fit.npz') as data:wrong_residual=data['residual'].copy()
    with np.load(output/'background_checks/border_10_truth.npz') as data:good_residual=data['residual'].copy();good_prediction=data['predicted'].copy()
    with np.load(output/'background_checks/border_10_experimental_estimate.npz') as data:corrected_wrong=data['residual'].copy()
    with np.load(output/'background_checks/simulation_border_subtracted_truth.npz') as data:sim_residual=data['residual'].copy()
    residuals={'Experimental: raw truth':raw_residual,'Experimental: raw estimate':wrong_residual,
               'Experimental: background removed, truth':good_residual,
               'Experimental: background removed, old estimate':corrected_wrong,
               'Noisy simulation: background removed, truth':sim_residual}
    rows=[]
    for j,i in enumerate(ids):
        row=dict(image=int(i+1),name=str(acq.names[i]),z_um=float(z[j]),background=float(measure['background'][i]))
        for name,array in residuals.items():row[name]=float(np.sqrt(np.mean(array[j]**2)))
        rows.append(row)
    with (output/'inverse_residuals.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    pages=[]
    def trace(name,x,series,title,xlabel,ylabel,**kwargs):
        fig,ax=plot_error_histories(x,series,title=title,xlabel=xlabel,ylabel=ylabel,integer_x=False,**kwargs)
        if len(series)>2:
            ax.legend(loc='upper center',bbox_to_anchor=(.5,1.23),ncol=2,fontsize=7)
        for ext in ['png','pdf']:fig.savefig(figures/(name+'.'+ext),dpi=300)
        pdf.savefig(fig,dpi=300);plt.close(fig);pages.append(name)
    with PdfPages(output/'diagnostics.pdf') as pdf:
        times=measure['elapsed_minutes']
        trace('01_background',times,{'Measured border median':measure['background'],
              'Reference-sphere model border':np.full(len(times),-4.561239626300128)},
              'Background changes within the acquisition','Acquisition time [min]','Saved averaged intensity',nonnegative=False)
        trace('02_background_noise',times,{'Border MAD noise':measure['sigma']},
              'Framewise noise floor','Acquisition time [min]','Robust standard deviation')
        ref_before=np.array([np.flatnonzero((acq.focal_z*1000==v)&(acq.names=='reference_before'))[0] for v in levels])
        ref_after=ref_before+17
        ratio=measure['flux'][ref_after]/measure['flux'][ref_before]
        trace('03_reference_repeat_flux',levels,{f'Bead {i+1}':ratio[:,i] for i in range(ratio.shape[1])},
              'Reference aperture flux: after / before','Requested Z offset [$\\mu$m]','Flux ratio',nonnegative=False)
        displacement=np.linalg.norm(measure['centroids'][ref_after]-measure['centroids'][ref_before],axis=2)
        trace('04_reference_centroids',levels,{f'Bead {i+1}':displacement[:,i] for i in range(displacement.shape[1])},
              'Reference centroid repeatability (defocus-biased indicator)','Requested Z offset [$\\mu$m]','Centroid displacement [$\\mu$m]')
        refpairs=[p for p in audit['repeat_pairs'] if p['reference']]
        trace('05_repeat_spatial_correlation',levels,{'X neighbor':np.array([p['residual_neighbor_x_correlation'] for p in refpairs]),
              'Y neighbor':np.array([p['residual_neighbor_y_correlation'] for p in refpairs])},
              'Reference-repeat residual correlation','Requested Z offset [$\\mu$m]','Neighbor correlation',nonnegative=False)
        # Axial aperture curves use the same fixed ROIs and background treatment.
        fit=json.loads((ROOT/'reference_bead_orientation_scan/flip_x/reference_fit.json').read_text())
        x,y=acquisition_grid(6.5,(64,64)).get_xy();x=x*1000;y=y*1000
        from experimental_10_15.empirical_noise import background_noise
        pb,_=background_noise(prediction)
        for bead,(xc,yc,*_) in enumerate(fit['beads']):
            mask=(x[:,None]-xc)**2+(y[None,:]-yc)**2<.65**2
            measured_flux=(measure['flux'][ref_before,bead]+measure['flux'][ref_after,bead])/2
            model_flux=((prediction[ref_before]-pb[ref_before,None,None])[:,mask].sum(axis=1)+
                        (prediction[ref_after]-pb[ref_after,None,None])[:,mask].sum(axis=1))/2
            trace(f'06_axial_bead_{bead+1}',levels,{'Reference measured':measured_flux,'Reference sphere model':model_flux},
                  f'Bead {bead+1}: background-subtracted axial response','Requested Z offset [$\\mu$m]','Aperture sum',nonnegative=False)
        trace('07_residual_vs_time',times[ids],{name:np.sqrt(np.mean(r*r,axis=(1,2))) for name,r in residuals.items()},
              'Object refitted at each fixed wavefront','Acquisition time [min]','Measured-region residual RMS')
        trace('08_residual_vs_z',levels,{name:np.array([np.sqrt(np.mean(r[np.isclose(z,v)]**2)) for v in levels])
              for name,r in residuals.items()},'Residual across Z','Requested Z offset [$\\mu$m]','Measured-region residual RMS')
        # Phase curves have a common normalization, not a separate denominator for each line.
        phase=json.loads((output/'background_checks/corrected_phase_scales.json').read_text())
        truepoint=next(row for row in phase if row['probe_scale']==1.)
        fractions=np.asarray(truepoint['fractions']);denom=truepoint['losses'][list(fractions).index(1.)]
        trace('09_corrected_phase_scale',fractions,{f"Probe scale {row['probe_scale']:g}":np.asarray(row['losses'])/denom for row in phase},
              'Background removed: imposed/probe scale checks','Fraction of imposed wavefront','Loss / nominal truth loss')
        # Sensitivity plots compare the same two candidates inside each model.
        raw=[];good=[]
        for value in [.9,1.,1.1]:
            a=nominal if value==1 else json.loads((output/f'sweeps/z_scale_{value:g}.json').read_text())
            b=corrected if value==1 else json.loads((output/f'background_checks/corrected_zscale_{value:g}.json').read_text())
            raw.append(a);good.append(b)
        trace('10_z_scale',np.array([.9,1,1.1]),{'Raw: truth':np.array([r['truth_loss']/nominal['truth_loss'] for r in raw]),
              'Raw: old estimate':np.array([r['estimate_loss']/nominal['truth_loss'] for r in raw]),
              'Background removed: truth':np.array([r['truth_loss']/corrected['truth_loss'] for r in good]),
              'Background removed: old estimate':np.array([r['experimental_estimate_loss']/corrected['truth_loss'] for r in good])},
              'Z scale sensitivity; normalized within background treatment','Requested Z multiplier','Relative loss')
        grid=acquisition_grid(6.5,(64,64));microscope=psf.Microscope(**case['params'])
        # Representative plus worst corrected residual images, using shared image style.
        worst=np.argsort(np.mean(good_residual**2,axis=(1,2)))[-4:]
        central=np.flatnonzero(np.isclose(z,0))[:4]
        selected=np.unique(np.r_[central,worst])
        for first in range(0,len(selected),4):
            chosen=selected[first:first+4]
            commands=[zernike.Aberration(case['modes'],acq.total[ids[j]]) for j in chosen]
            images=case['images'][chosen]-measure['background'][ids[chosen],None,None]
            peak=max(images.max(),good_prediction[chosen].max())
            numbers=', '.join(f'{ids[j]+1:03d}' for j in chosen)
            for label,other in [('Known-wavefront fit',good_prediction[chosen]),('Absolute residual',abs(good_residual[chosen]))]:
                fig,_=plot_acquisition_comparison(grid,images/peak,other/peak,commands,case['focal_z'][chosen],microscope.alpha,
                    title='Background removed; images '+numbers,column_titles=('Experimental',label))
                name=f'11_images_{first}_{label.replace(" ","_")}'
                for ext in ['png','pdf']:fig.savefig(figures/(name+'.'+ext),dpi=300)
                pdf.savefig(fig,dpi=300);plt.close(fig);pages.append(name)
    summary=dict(raw=nominal,background_removed=corrected,noisy_simulation_background_removed=sim,
                 noisy_simulation_with_experimental_background=injected,
                 background_quantiles=np.quantile(measure['background'][ids],[0,.1,.5,.9,1]),
                 pdf_pages=len(pages),pages=pages)
    write_json(output/'summary.json',summary)
    sweep=list(csv.DictReader((output/'model_sweeps.csv').open()))
    axes=list(csv.DictReader((output/'pupil_axes.csv').open()))
    controls=list(csv.DictReader((output/'control_offsets.csv').open()))
    bgrows=list(csv.DictReader((output/'background_validation.csv').open()))
    out=[
        '# Experimental image-formation diagnostics',
        '',
        'All checks run locally on copies of the signed MATLAB averages. No acquisition, submission packet, correction, alignment or saved optimization was changed. The main comparison uses the same 134 PD images, X mirror, 17 modes, 21 object planes and regularization as the completed experimental solve.',
        '',
        '## Main result',
        '',
        f"Framewise background subtraction changes known-truth residual RMS from {nominal['truth_rmse']:.3f} to {corrected['truth_rmse']:.3f}; the matched noisy simulation gives {sim['truth_rmse']:.3f}. The old experimental estimate/truth loss ratio changes from {nominal['estimate_over_truth']:.4f} to {corrected['experimental_estimate_over_truth']:.4f}. The object is refit at each fixed wavefront; the aberration is not reoptimized.",
        '',
        'The existing simulation has a nearly constant background. The experiment has large frame-to-frame border-offset changes. Adding those experimental offsets to the frozen noisy simulation raises its known-truth residual to '+f"{injected['truth_rmse']:.3f}. This isolates a substantial contribution from backgrounds; it does not reproduce every experimental artifact or establish that the remaining optical model is exact.",
        '',
        '## Background robustness and remaining optical sensitivity',
        '',
        '| Treatment | Truth residual RMS | Old estimate / truth loss |',
        '|---|---:|---:|',
    ]
    for row in bgrows:out.append(f"| {row['variant']} | {float(row['truth_rmse']):.3f} | {float(row['experimental_estimate_over_truth']):.4f} |")
    out.extend(['','Independent border widths and corners give a check against bead-tail contamination. Reference-only interpolation tests whether a background measured just at the ends of a plane is sufficient. All comparisons within a row share the same data treatment. Loss magnitudes across different intensity scales, regularization, NA or excitation order are not a calibrated model evidence; PSF amplitude and object priors change too.',
        '',
        'After background subtraction, the tested imposed-wavefront scaling curve has its minimum at alpha=1.0 on the sampled grid for each positive probe scale (0.75, 1.0, 1.25). The nominal probe scale has the lowest sampled loss. Modest Z/NA/FOV changes yield much smaller residual changes than background removal; the old estimate remains worse than truth throughout the corrected sensitivity tests.',
        '', '## Acquisition and repeat checks', '',
        f"The acquisition took {audit['duration_minutes']:.1f} minutes. Each saved frame averages {audit['frames_per_saved_image']} exposures; individual exposures were not saved. Z metadata: {audit['z_position_source']}. Native system aberration metadata: {audit['native_system_aberration']}.",
        '',
        'Reference aperture sums, backgrounds, noise and centroid differences are in frames.csv and repeats.csv. Centroids use fixed apertures and can be biased by defocus/background; they are motion indicators. No measured image was registered. Repeat slope regressions smooth both inputs to limit errors-in-variables attenuation; aperture flux ratios are the primary intensity-drift diagnostic.',
        '',
        '## Phase and implementation checks', '',
        f"Independent MATLAB-polynomial versus Python evaluation agrees to {audit['matlab_python_polynomial_max_error']:.3g}. Both use normalized pupil radius sin(theta)/sin(alpha), normalized real Zernikes and coefficients in waves. The writer uses 256 raw units per full cycle. This verifies software conventions, not the actual phase response, relay pupil, LUT or physical SLM alignment.",
        '',
        'Raw and corrected imposed/probe phase scale profiles are in phase_scales.json and background_checks/corrected_phase_scales.json. Pupil reflections/rotations transform every mode, including known diversities; camera images and polarization stay fixed. The previous eight camera-orientation reference fits are reused as separate existing evidence.',
        '', '| Pupil hypothesis | Truth loss / nominal raw truth |','|---|---:|'])
    out.extend(f"| {r['name']} | {float(r['truth_loss'])/nominal['truth_loss']:.4f} |" for r in axes)
    out.extend(['','## Other sensitivity tests','','| Variant | Truth residual RMS | Old estimate / truth loss |','|---|---:|---:|'])
    out.extend(f"| {r['variant']} | {float(r['truth_rmse']):.3f} | {float(r['estimate_over_truth']):.4f} |" for r in sweep)
    out.extend(['','These tests hold the two coefficient vectors fixed while refitting the object. They do not find a new optimizer minimum under each altered model. Z-window and probe-sign subsets are in subsets.csv; the old estimate was learned on all images, so these are sensitivity checks, not independent validation.',
        '', '## Historical zero controls', '',
        'acquisition_inventory.csv checks exact system-AO map equality. The System AO control matches; the no-aberration control does not. control_offsets.csv evaluates both signs and two parity conventions of the historical estimated vectors. Those controls were solved with different image conventions and sample data: neither vector is an independently measured common residual, and a transferred offset must not be called ground truth.',
        '', '| Control | Same AO map | Parity | Sign | Loss / nominal raw truth |','|---|---|---|---:|---:|'])
    out.extend(f"| {r['control']} | {r['same_system_ao']} | {r['parity']} | {r['sign']} | {float(r['loss'])/nominal['truth_loss']:.4f} |" for r in controls)
    out.extend(['','## Noise, object assumptions and limits','',
        'sphere_residuals.csv evaluates reference-constrained finite spheres before and after an imagewise gain/offset fit, using the repeat-calibrated noise model. sphere_blur.csv tests simple axis-dependent Gaussian blur. These nuisance fits use the evaluated images and are explanatory bounds, not validated calibrations. Inverse residual comparisons refit a free signed object and do not impose perfect spheres. Negative object mass is recorded for both experiment and simulation; it is not itself proof of a physical negative object.',
        '',
        'Unavailable from these data: individual-exposure statistics, absolute photon gain, independent stage readback, actual pupil illumination/SLM phase/LUT response, and a separately measured system residual. Brightness, focus drift, background structure and residual optics can remain confounded. Full spatial Poisson likelihood, nonnegative object reconstruction and newly optimized corrected-data aberrations were not fitted in this diagnostic.',
        '', '## Recommended next experiment with existing data','',
        'Rerun the experimental aberration optimization after subtracting a robust per-image border background, keeping the existing exclusions, orientation and fast settings. Compare the recovered coefficients to the imposed truth and inspect the corrected residual figures. Keep the background values in provenance. The present result makes this a much stronger next test than increasing random starts.',
        '', '## Reproduce','',
        '```bash','conda activate phase_diversity_env','python -m experimental_10_15.image_model_diagnostics','```',
        '', 'The pipeline checkpoints each variant. diagnostics.pdf contains the scientific figures; individual 300-DPI PNG/PDF exports are in figures/. All residual arrays and machine-readable tables remain alongside this report.'])
    (output/'REPORT.md').write_text('\n'.join(out)+'\n')
    print(output/'diagnostics.pdf',flush=True)
