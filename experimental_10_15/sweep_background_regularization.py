"""Profile regularization at fixed new/true phases, reusing exact object SVDs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import fft
from experimental_10_15.compare_background_solutions import NEW
from experimental_10_15.image_model_diagnostics import ROOT,load_case,workspace,csv_rows
from experimental_10_15.bead_pd_io import write_json

OUTPUT=ROOT/'background_regularization_sweep'


def regularization_path(ws, strengths, rhos, *, canonical_checks=True):
    """Reuse fixed SVD geometry; recompute the optimal object at every rho."""
    rhos=np.asarray(rhos,dtype=float)
    if rhos.ndim!=1 or not np.all(np.isfinite(rhos)) or np.any(rhos<=0):raise ValueError('Require positive finite rho values.')
    ws.reuse_svd=True
    fit=ws._get_fit(np.asarray(strengths))[0]
    factors=fit.factors;initial_loss=fit.loss;initial_rho=ws.rho
    del fit
    # Objects and Fourier residuals use the exact same expressions as the
    # canonical CPU solver. Only geometry/factorization is reused across rho.
    records=[]
    pixels=np.prod(ws.shape)
    zero_loss=float(np.sum(ws.weights*np.sum(abs(ws.observed)**2,axis=0)))
    images=fft.irfft2(ws.observed.reshape(len(ws.focal_z),ws.shape[0],ws.shape[1]//2+1),s=ws.shape)[ws.crop]
    for rho in rhos:
        estimated=np.empty((len(ws.sample_z),len(ws.weights)),complex)
        residual=np.empty_like(ws.observed)
        for start,stop,u,s,vh in factors:
            d=ws.observed[:,start:stop].T
            rank=s.shape[1];left=u[:,:,:rank]
            projected=(left.conj().swapaxes(-1,-2)@d[...,None])[...,0]
            fitted=s/(s*s+rho)*projected
            estimated[:,start:stop]=(vh[:,:rank].conj().swapaxes(-1,-2)@fitted[...,None])[...,0].T
            remaining=(left@(rho/(s*s+rho)*projected)[...,None])[...,0]
            if len(ws.focal_z)>rank:remaining+=d-(left@projected[...,None])[...,0]
            residual[:,start:stop]=remaining.T
        data=float(np.sum(ws.weights*np.sum(abs(residual)**2,axis=0)))
        norm=float(np.sum(ws.weights*np.sum(abs(estimated)**2,axis=0)))
        penalty=rho*norm;loss=data+penalty
        interior=fft.irfft2(residual.reshape(len(ws.focal_z),ws.shape[0],ws.shape[1]//2+1),s=ws.shape)[ws.crop]
        records.append(dict(rho=float(rho),loss=loss,data_loss=data,regularization_loss=penalty,
            measured_rmse=float(np.sqrt(np.mean(interior**2))),object_norm_squared=norm/pixels,
            loss_over_zero_object=loss/zero_loss,
            fitted_image_norm_fraction=float(np.linalg.norm(images-interior)/np.linalg.norm(images))))
        print(f'  rho={rho:.3g}: loss={loss:.8g}, residual RMS={records[-1]["measured_rmse"]:.4f}',flush=True)
    if np.any(rhos==initial_rho):
        actual=records[int(np.flatnonzero(rhos==initial_rho)[0])]['loss']
        np.testing.assert_allclose(actual,initial_loss,rtol=1e-11)
    checks=[]
    if canonical_checks:
        ws.reuse_svd=False
        for index in sorted({0,len(rhos)//2,len(rhos)-1}):
            ws.rho=float(rhos[index]);canonical=float(ws.loss(strengths))
            relative=abs(canonical-records[index]['loss'])/canonical
            if relative>1e-10:raise ValueError(f'SVD path/canonical mismatch: {relative:g}')
            checks.append(dict(rho=ws.rho,relative_error=relative))
    ws.rho=initial_rho
    return records,checks


def run(output=OUTPUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    case=load_case(NEW);rhos=10.**np.arange(40.,52.01,.5)
    candidates={'Truth':case['imposed_for_validation']+case['cumulative_correction'],
                'New estimate':case['estimate'],'Naive zero':np.zeros_like(case['estimate'])}
    records={};checks={}
    for label,coeff in candidates.items():
        file=output/(label.replace(' ','_')+'.json')
        if file.exists():
            saved=json.loads(file.read_text());records[label]=saved['records'];checks[label]=saved['checks']
            np.testing.assert_array_equal([r['rho'] for r in records[label]],rhos)
            np.testing.assert_array_equal(saved['coefficients'],coeff)
        else:
            print('Regularization path:',label,flush=True)
            ws=workspace(case)
            records[label],checks[label]=regularization_path(ws,coeff,rhos)
            write_json(file,dict(records=records[label],checks=checks[label],coefficients=coeff))
            del ws
    rows=[]
    for t,e,n in zip(records['Truth'],records['New estimate'],records['Naive zero']):
        rows.append(dict(rho=t['rho'],new_over_truth=e['loss']/t['loss'],truth_loss=t['loss'],new_loss=e['loss'],
            truth_over_naive=t['loss']/n['loss'],new_over_naive=e['loss']/n['loss'],naive_loss=n['loss'],
            naive_rmse=n['measured_rmse'],naive_data_loss=n['data_loss'],naive_regularization_loss=n['regularization_loss'],
            truth_rmse=t['measured_rmse'],new_rmse=e['measured_rmse'],
            truth_data_loss=t['data_loss'],new_data_loss=e['data_loss'],
            truth_regularization_loss=t['regularization_loss'],new_regularization_loss=e['regularization_loss'],
            truth_object_norm_squared=t['object_norm_squared'],new_object_norm_squared=e['object_norm_squared'],
            truth_loss_over_zero_object=t['loss_over_zero_object'],new_loss_over_zero_object=e['loss_over_zero_object'],
            truth_fitted_image_norm_fraction=t['fitted_image_norm_fraction'],new_fitted_image_norm_fraction=e['fitted_image_norm_fraction']))
    csv_rows(output/'sweep.csv',rows)
    write_json(output/'summary.json',dict(rows=rows,canonical_checks=checks,
        description='Same 134 background-subtracted images and original object grid; fixed new phase, imposed+cumulative truth and zero unknown aberration; known diversity probes remain in all cases. Optimal signed object refitted at each rho. No intensity factors or aberration reoptimization.'))
    plot(output,records,rows)


def plot(output,records,rows):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from systematic_testing.correction_examples.plot_pd_iteration_rms import plot_error_histories
    rhos=np.array([r['rho'] for r in rows]);ratio=np.array([r['new_over_truth'] for r in rows])
    with PdfPages(output/'regularization_sweep.pdf') as pdf:
        def figure(name,series,title,ylabel,focus=False):
            fig,ax=plot_error_histories(rhos,series,title=title,xlabel=r'Regularization $\rho$',ylabel=ylabel,integer_x=False)
            ax.set_xscale('log');ax.axvline(1e46,color='k',lw=.8,ls=':')
            if focus:
                ax.axhline(1,color='k',lw=.8,ls='--')
                ax.set_ylim(min(min(v) for v in series.values())-.03 if min(min(v) for v in series.values())<1 else .97,
                            max(max(max(v) for v in series.values()),1)+.03)
            if len(series)>2:ax.legend(loc='upper center',bbox_to_anchor=(.5,1.23),ncol=2,fontsize=7)
            for ext in ['png','pdf']:fig.savefig(output/(name+'.'+ext),dpi=300)
            pdf.savefig(fig,dpi=300);plt.close(fig)
        figure('loss_ratio',{'New estimate / truth':ratio},'Background-subtracted PD: fixed wavefronts','Profiled loss ratio',True)
        figure('truth_vs_naive',{'Truth / naive zero':np.array([r['truth_over_naive'] for r in rows]),
                               'New estimate / naive zero':np.array([r['new_over_naive'] for r in rows])},
               'Same fitted-object objective; zero unknown phase baseline','Profiled loss / naive-zero loss',True)
        figure('residual_rms',{k:np.array([r['measured_rmse'] for r in v]) for k,v in records.items()},
               'Object refitted at each regularization','Measured-region residual RMS')
        zero=records['Truth'][0]['loss']/records['Truth'][0]['loss_over_zero_object']
        figure('loss_terms',{label+' '+term:np.array([r[key]/zero for r in values])
               for label,values in records.items() for term,key in [('residual','data_loss'),('penalty','regularization_loss')]},
               'Objective decomposition','Loss term / zero-object loss')
        index=int(np.flatnonzero(rhos==1e46)[0])
        figure('object_norm',{label:np.array([r['object_norm_squared']/values[index]['object_norm_squared'] for r in values])
               for label,values in records.items()},'Reconstructed-object norm','Squared norm / same-phase norm at $10^{46}$')
    text=['# Regularization sweep: new estimate, truth and naive zero','',
        'Same 134 background-subtracted 64x64 PD images, original X mirror/exclusions, 21 object planes over 4 microns. Each fixed wavefront gets its own optimal signed object at every rho. No intensity factors, phase reoptimization or cluster operations.',
        '',
        '25 half-decade values from 1e40 through 1e52. The ratio remains near 0.83 through 1e48, rises to 0.9698 at 1e51, crosses one between 1e51 and 3.16e51, and reaches 1.0687 at 1e52. Thus sufficiently strong regularization reverses the preference between these two fixed wavefronts.',
        '',
        'The added naive-zero check changes the interpretation at the upper end: truth/naive = 0.5043 at 1e46, 0.6764 at 1e51, 0.9272 at 3.16e51 and 1.3154 at 1e52. At 3.16e51 truth has lower loss than both the new estimate and naive zero. At 1e52 naive zero wins against both. Treat the 1e52 optimization as an overregularization control rather than assuming its truth/new ranking predicts good recovery. The prepared three-job batch was not changed or submitted by this check.',
        '',
        'At 3.16e51 the residual-only loss still slightly favors the new estimate (ratio 0.9769), but its regularization penalty is larger (ratio 1.1931), making total loss favor truth (ratio 1.0484). Measured residual RMS increases to 7.125 for truth and 7.042 for the new estimate. Predictions retain about 68% and 65% of the measured image norm, respectively; the crossing is not merely both objects collapsing to zero. Nonetheless stronger regularization trades data fit for a smaller reconstructed object.',
        '',
        'A fresh phase optimization near 1e51–3.16e51 is a reasonable next diagnostic. This fixed-candidate sweep cannot establish its recovered wavefront or rule out other lower-loss phases, and choosing rho solely because it favors the known answer is not independent validation.',
        '', 'The SVD of each fixed image-formation operator is reused exactly. Independent canonical loss checks at both ends and the middle validate the path. A ratio below one favors the saved new estimate; above one favors truth.',
        '',
        'Naive zero means all unknown aberration coefficients equal zero, with all known diversity commands retained and a separately refitted object. It is distinct from the zero-object data-only baseline. Ratios to naive below one favor the numerator candidate.',
        '', '| Rho | New / truth loss | Truth / naive loss | New / naive loss | Truth RMS | New RMS | Naive RMS |','|---|---:|---:|---:|---:|---:|---:|']
    text.extend(f"| {r['rho']:.3g} | {r['new_over_truth']:.6f} | {r['truth_over_naive']:.6f} | {r['new_over_naive']:.6f} | {r['truth_rmse']:.4f} | {r['new_rmse']:.4f} | {r['naive_rmse']:.4f} |" for r in rows)
    text.extend(['','At very large rho both objects approach zero and both losses approach the same data-only baseline. A ratio near one in that regime means loss of discrimination, not successful recovery. Raw objective values at different rho are different objectives and should not be minimized against each other to choose rho.',
                 '', 'Files: regularization_sweep.pdf (five figures); standalone loss_ratio.png/pdf, truth_vs_naive.png/pdf, residual_rms.png/pdf, loss_terms.png/pdf, object_norm.png/pdf; sweep.csv and summary.json.',
                 '', 'Reproduce locally: `python -m experimental_10_15.sweep_background_regularization`.'])
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(output/'regularization_sweep.pdf',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    run(**vars(parser.parse_args()))
