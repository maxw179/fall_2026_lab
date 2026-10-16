"""Plot the three-way comparison using the established scientific helpers."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from experimental_10_15.compare_background_solutions import NEW
from experimental_10_15.image_model_diagnostics import load_case,ROOT
from experimental_10_15.bead_pd import acquisition_grid
from experimental_10_15.bead_pd_io import write_json
from experimental_10_15.plot_many_bead_images import SOURCE
from experimental_10_15.bead_pd_io import load_acquisition
from systematic_testing.correction_examples.plot_pd_iteration_rms import plot_error_histories
from systematic_testing.correction_examples.plot_pd_correction_examples import plot_acquisition_comparison
from utils import psf,zernike


def report(output):
    output=Path(output);figures=output/'figures';figures.mkdir(exist_ok=True)
    summary=json.loads((output/'summary.json').read_text());case=load_case(NEW);acq=load_acquisition(SOURCE)
    candidates=summary['candidates'];labels=list(candidates);modes=case['modes']
    rows=summary['losses'];bg=[r for r in rows if r['dataset']=='background_PD']
    truth_loss=next(r['loss'] for r in bg if r['candidate']=='Truth')
    constraints=list(csv.DictReader((output/'reference_constraints.csv').open()))
    transfer=list(csv.DictReader((output/'reference_prediction_controls.csv').open()))
    gainrows=list(csv.DictReader((output/'alternating_gains.csv').open()))
    postfit=list(csv.DictReader((output/'postfit_gains.csv').open()))
    grouped=list(csv.DictReader((output/'postfit_group_gains.csv').open()))
    frame=list(csv.DictReader((output/'frame_residuals.csv').open()))
    phase={label:np.asarray(v) for label,v in candidates.items()}
    truth=phase['Truth'];squared=(phase['New estimate']-truth)**2
    trefoil=modes[:,0]==-3
    fraction=float(squared[trefoil].sum()/squared.sum())
    paths=json.loads((output/'paths.json').read_text());pages=[]
    with PdfPages(output/'comparison.pdf') as pdf:
        def trace(name,x,series,title,xlabel,ylabel,**options):
            fig,ax=plot_error_histories(x,series,title=title,xlabel=xlabel,ylabel=ylabel,integer_x=False,**options)
            if len(series)>2:ax.legend(loc='upper center',bbox_to_anchor=(.5,1.23),ncol=2,fontsize=7)
            return fig,ax
        def save(name,fig):
            for ext in ['pdf','png']:fig.savefig(figures/(name+'.'+ext),dpi=300)
            pdf.savefig(fig,dpi=300);plt.close(fig);pages.append(name)
        fig,ax=trace('paths',np.asarray(paths[0]['fractions']),{r['label']:np.asarray(r['losses'])/truth_loss for r in paths},
             'Background-subtracted PD objective','Fraction along truth-to-estimate segment','Loss / truth loss')
        ax.axvline(0,color='k',lw=.6);ax.axvline(1,color='k',lw=.6,ls=':');save('01_loss_paths',fig)
        indices=np.arange(len(modes))
        fig,ax=trace('coefficients',indices,phase,'Signed Zernike coefficients','Mode $(m,n)$','Coefficient [waves]',nonnegative=False)
        ax.set_xticks(indices,labels=[f'$({m},{n})$' for m,n in modes],rotation=90);save('02_coefficients',fig)
        fig,ax=trace('errors',indices,{label:abs(phase[label]-truth) for label in labels if label!='Truth'},
            'Absolute coefficient errors','Mode $(m,n)$','Absolute error [waves]')
        ax.set_xticks(indices,labels=[f'$({m},{n})$' for m,n in modes],rotation=90);save('03_coefficient_errors',fig)
        z=case['focal_z']*1000;levels=np.unique(z)
        arrays={}
        for label in labels:
            with np.load(output/('background_PD_'+label.replace(' ','_')+'.npz')) as data:arrays[label]=data['residual'].copy()
        fig,ax=trace('z',levels,{label:np.array([np.sqrt(np.mean(r[np.isclose(z,v)]**2)) for v in levels]) for label,r in arrays.items()},
            'Same fitted object across all PD images','Requested Z offset [$\\mu$m]','Residual RMS')
        save('04_residual_vs_z',fig)
        fig,ax=trace('standardized',levels,{label:np.array([np.sqrt(np.mean([float(r['standardized_residual_rms'])**2 for r in frame
              if r['candidate']==label and np.isclose(float(r['z_um']),v)])) for v in levels]) for label in labels},
              'Post-fit noise-standardized residuals','Requested Z offset [$\\mu$m]','Standardized residual RMS')
        save('05_standardized_residual',fig)
        groups=[]
        for v in case['diversity_strengths']:
            groups.append('Zero' if not np.any(v) else tuple(modes[np.argmax(abs(v))]))
        unique=list(dict.fromkeys(groups))
        group_ids=[np.array([g==key for g in groups]) for key in unique]
        fig,ax=trace('probes',np.arange(len(unique)),{label:np.array([np.sqrt(np.mean(r[keep]**2)) for keep in group_ids]) for label,r in arrays.items()},
              'Residual grouped by diversity mode (both signs)','Probe $(m,n)$','Residual RMS')
        ax.set_xticks(np.arange(len(unique)),labels=['Zero' if g=='Zero' else f'$({g[0]},{g[1]})$' for g in unique],rotation=45)
        save('06_residual_by_probe',fig)
        for dataset in ['experiment','noisy_simulation']:
            for hypothesis in ['common_residual','fixed_unaberrated_references']:
                selected=[r for r in transfer if r['dataset']==dataset and r['hypothesis']==hypothesis]
                curves={label:np.array([np.sqrt(np.mean([float(r['rmse'])**2 for r in selected if r['candidate']==label and np.isclose(float(r['z_um']),v)])) for v in levels]) for label in labels}
                fig,ax=trace('reference',levels,curves,dataset.replace('_',' ')+': '+hypothesis.replace('_',' '),
                    'Requested Z offset [$\\mu$m]','Held-out reference prediction RMS')
                save('07_reference_'+dataset+'_'+hypothesis,fig)
        for dataset in ['experiment','noisy_simulation']:
            gain_data={label:json.loads((output/('gains_'+dataset+'_'+label.replace(' ','_')+'.json')).read_text()) for label in labels}
            fig,ax=trace('gains',np.arange(len(case['images']))+1,{label:np.asarray(r['gains']) for label,r in gain_data.items()},
                dataset.replace('_',' ')+': fitted image gains','Retained PD image index','Gain (mean squared gain = 1)')
            save('09_gains_'+dataset,fig)
            # Exact histories may terminate at different iterations; plot each
            # with the shared helper, then append the remaining traces.
            first=gain_data['Truth'];history=first['history'];normal=history[0]['loss']
            fig,ax=trace('gain_history',np.array([r['iteration'] for r in history]),
                {'Truth':np.array([r['loss']/normal for r in history])},
                dataset.replace('_',' ')+': object/gain alternation','Alternating update','Loss / initial truth loss')
            for label in labels[1:]:
                h=gain_data[label]['history'];ax.plot([r['iteration'] for r in h],[r['loss']/normal for r in h],'.-',label=label,markersize=4)
            ax.set_ylim(0,max(r['history'][0]['loss']/normal for r in gain_data.values())*1.15)
            ax.legend(loc='upper center',bbox_to_anchor=(.5,1.23),fontsize=8)
            save('10_gain_iterations_'+dataset,fig)
        # Show three candidates on identical representative frames using the
        # same measured/simulated layout and phase inset conventions.
        newr=arrays['New estimate'];tr=arrays['Truth']
        improvement=np.mean(tr**2-newr**2,axis=(1,2))
        selected=np.unique(np.r_[np.argsort(improvement)[-2:],np.argsort(improvement)[:2]])
        grid=acquisition_grid(float(case['field_of_view_um']),(64,64));alpha=psf.Microscope(**case['params']).alpha
        images=case['images'][selected]
        peak=max(images.max(),max((case['images'][selected]-r[selected]).max() for r in arrays.values()))
        commands=[zernike.Aberration(modes,acq.total[case['source_indices'][j]]) for j in selected]
        numbers=', '.join(str(case['source_indices'][j]+1) for j in selected)
        for label,r in arrays.items():
            fig,_=plot_acquisition_comparison(grid,images/peak,(images-r[selected])/peak,commands,case['focal_z'][selected],alpha,
                title='Images '+numbers,column_titles=('Background-subtracted data',label+' prediction'))
            save('08_examples_'+label.replace(' ','_'),fig)
    gain_exp={r['candidate']:r for r in gainrows if r['dataset']=='experiment'}
    gain_ratio=float(gain_exp['New estimate']['loss'])/float(gain_exp['Truth']['loss'])
    out=['# Old / new / true wavefront diagnostic','',
         'Read-only comparison on the same 134 PD images, X mirror, 64x64 means, signed per-image background subtraction, modes, object grid and rho. Each candidate gets its own refitted object. Both saved optimizer endpoint losses replay within numerical precision. No Bouchet connection or new aberration optimization.',
         '', '## Main findings', '',
         f"The new estimate is closer to truth ({summary['rms_error_waves']['New estimate']:.4f} waves error versus {summary['rms_error_waves']['Old estimate']:.4f} before), but the corrected PD objective still favors it over truth by {100*(1-next(r['loss'] for r in bg if r['candidate']=='New estimate')/truth_loss):.2f}%. The matched noisy simulation favors truth over both experimental vectors.",
         '',
         f"Alternating a free signed object with positive per-image gains reduces the new estimate's loss advantage over truth to {100*(1-gain_ratio):.2f}%. Mean squared gain is constrained to one, preventing arbitrary global intensity rescaling. These are at most 12 alternating updates per fixed phase, not new aberration optimizations or proof of the global nuisance minimum.",
         '',
         'The reference tests distinguish two possibilities: interpreting the new error as a common system residual worsens the joint fit substantially. Holding reference PSFs fixed and unaberrated instead still favors the new PD estimate in experiment, but not in the matched noisy simulation. References alone therefore do not remove this experimental preference; simply calling the remaining discrepancy native system aberration is not supported by these tests.',
         '', '| Dataset | Candidate | Loss / truth | Measured residual RMS |','|---|---|---:|---:|']
    for dataset in dict.fromkeys(r['dataset'] for r in rows):
        selected=[r for r in rows if r['dataset']==dataset];denom=selected[0]['loss']
        for r in selected:out.append(f"| {dataset} | {r['candidate']} | {r['loss']/denom:.4f} | {r['measured_rmse']:.3f} |")
    out.extend(['','## Wavefront error and loss geometry','',
        f"Wavefront error against the imposed phase: old {summary['rms_error_waves']['Old estimate']:.6f} waves RMS; new {summary['rms_error_waves']['New estimate']:.6f} waves RMS. The known imposed wavefront is validation truth; it does not independently include an unknown system-AO residual.",
        '',f"The two sine-trefoil coefficients (m=-3, n=3 and 5) account for {100*fraction:.1f}% of the new estimate's squared coefficient error (coefficient-space fraction, not a separately integrated pupil RMS fraction).",
        '', 'Loss paths interpolate whole signed coefficient vectors from truth to each saved estimate while refitting the object. Fraction 0 is truth; fraction 1 is the saved estimate. A lower loss away from truth demonstrates objective preference along that path; it does not establish a global optimum.',
        '', '## Reference tests: two different hypotheses','',
        'The 18 references were not used in the two experimental aberration optimizations. They contain baseline + system AO, with no imposed/cumulative PD wavefront. All reference images receive their own border subtraction.',
        '',
        '**Common-residual hypothesis:** the estimate equals the known imposed+cumulative phase plus a native/system-AO residual. Reference phase therefore includes estimate-minus-known-imposed/cumulative. This tests whether the remaining error can be interpreted as one shared system residual. It explicitly uses the known imposed phase for this diagnostic.',
        '',
        '**Fixed-unaberrated-reference hypothesis:** reference PSFs are held at their known baseline, independently of the PD candidate. This assumes system AO perfectly cancels native aberration and tests whether the reference images constrain object changes that otherwise absorb PD wavefront error. It is a different model from a shared residual; it cannot establish independently that system AO is perfect.',
        '',
        'Joint comparisons refit one object to PD+references, holding the wavefront candidate fixed. Held-out predictions instead fit the object to PD only and predict references without refitting to them. No registration, imagewise gain or offset refit is applied in the main reference prediction/control tables. The matched noisy simulation controls for reference brightness, inverse-model discretization and object extrapolation effects.',
        '', '| Data | Reference hypothesis | Candidate | Joint loss / truth | PD RMS | Reference RMS |',
        '|---|---|---|---:|---:|---:|'])
    for dataset in ['experiment','noisy_simulation']:
        for hypothesis in ['common_residual','fixed_unaberrated_references']:
            selected=[r for r in constraints if r['dataset']==dataset and r['hypothesis']==hypothesis]
            denom=float(next(r['loss'] for r in selected if r['candidate']=='Truth'))
            for r in selected:out.append(f"| {dataset} | {hypothesis} | {r['candidate']} | {float(r['loss'])/denom:.4f} | {float(r['pd_rmse']):.3f} | {float(r['reference_rmse']):.3f} |")
    out.extend(['','## Subsets and residuals','',
        'frame_residuals.csv gives per-image RMS, fixed-foreground/background RMS, residual mean, spatial neighbor correlation and noise-standardized residuals. The noise variance is fixed from reference-sphere predictions and empirical repeat noise. Standardization is applied after the unweighted object fit; it is not a newly minimized Poisson or spatially weighted objective.',
        '', '| Subset | Candidate | Loss / truth |','|---|---|---:|'])
    for name in ['middle_Z','outer_Z','positive_probes','negative_probes']:
        selected=json.loads((output/(name+'.json')).read_text());denom=selected[0]['loss']
        for r in selected:out.append(f"| {name} | {r['candidate']} | {r['loss']/denom:.4f} |")
    out.extend(['','Subset objects are refitted independently; the candidate wavefronts were learned on all PD images, so subset comparisons are sensitivity tests rather than independent aberration validation. Border widths 5 and 16 test whether the new solution depends on the exact estimator.',
        '', '## Relative-brightness diagnostics', '',
        '| Dataset | Candidate | Alternating loss / same-data truth | Residual RMS | Gain p10 | Gain p90 |',
        '|---|---|---:|---:|---:|---:|'])
    for dataset in ['experiment','noisy_simulation']:
        selected=[r for r in gainrows if r['dataset']==dataset]
        denom=float(next(r['loss'] for r in selected if r['candidate']=='Truth'))
        for r in selected:out.append(f"| {dataset} | {r['candidate']} | {float(r['loss'])/denom:.4f} | {float(r['rmse']):.3f} | {float(r['gain_p10']):.3f} | {float(r['gain_p90']):.3f} |")
    out.extend(['',
        'Gain updates minimize the same padded Gaussian objective subject to positive gains and mean(gain^2)=1; each object update is the original regularized Fourier solve. Every alternating objective is checked for monotonic decrease. This is a diagnostic nuisance model, not an independently measured laser-power correction. Gain freedom can also hide optical phase errors. Fixed-object gain tests below hold the original fitted object unchanged and are less demanding explanatory bounds.',
        '', '| Dataset | Candidate | Gain grouping | Post-fit residual RMS |','|---|---|---|---:|'])
    for r in grouped:out.append(f"| {r['dataset']} | {r['candidate']} | {r['grouping']} | {float(r['rmse']):.3f} |")
    for r in postfit:out.append(f"| {r['dataset']} | {r['candidate']} | Individual image | {float(r['rmse']):.3f} |")
    out.extend(['',
        'A shared gain for each diversity command helps more than a shared gain for each Z plane. Probe command and within-plane acquisition order are confounded because each plane uses the same sequence. The data do not establish whether the effect is illumination/SLM throughput, timing, forward-model mismatch or a combination. No measured images were rescaled or new gain model added to the production solver.',
        '', '## Next useful tests', '',
        'Test a constrained gain-aware phase solve and judge the signed coefficient error, not only training loss. Start with command-shared gains as a smaller model, with a global-scale constraint; compare against imagewise gains and the matched noisy simulation. Keep the same data and retain gain/provenance diagnostics. No cluster run was prepared or submitted by this investigation.',
        '', '## Files','',
        '- comparison.pdf and figures/: shared plotting style, static scientific figures.',
        '- loss_comparison.csv, coefficients.csv, paths.json: canonical objective and phase comparisons.',
        '- reference_constraints.csv, reference_prediction_controls.csv: joint and held-out tests with matched simulation controls.',
        '- *_fit arrays and prediction/residual NPZ files: unnormalized values for follow-up analysis.',
        '', 'Reproduce locally: `python -m experimental_10_15.compare_background_solutions` (uses the phase_diversity_env environment; LaTeX required for plots). No cluster operations are performed.'])
    (output/'REPORT.md').write_text('\n'.join(out)+'\n')
    write_json(output/'report_summary.json',dict(pdf_pages=len(pages),new_trefoil_squared_coefficient_error_fraction=fraction,pages=pages))
    print(output/'comparison.pdf',flush=True)
