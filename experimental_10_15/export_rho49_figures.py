"""Export presentation PNGs from the completed rho=1e49 many-bead solve."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from experimental_10_15.image_model_diagnostics import ROOT,load_case,workspace,fit_diagnostics
from experimental_10_15.bead_pd import acquisition_grid
from experimental_10_15.bead_pd_io import load_acquisition,write_json
from experimental_10_15.plot_many_bead_images import SOURCE
from experimental_10_15.replay import build_replay
from systematic_testing.ao_animation.viewer import render_replay
from systematic_testing.correction_examples.plot_pd_iteration_rms import plot_error_histories
from systematic_testing.correction_examples.plot_pd_correction_examples import plot_acquisition_comparison
from utils import psf,zernike

CASE=ROOT/'multiple_beads_64x64_regularization_20261015/rho_1e49'
OUTPUT=CASE/'solve/figures/presentation_pngs'


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    case=load_case(CASE);result=json.loads((CASE/'solve/results/result.json').read_text())
    assert result['status']=='complete' and result['rho']==1e49
    history=json.loads((CASE/'solve/results/optimizer.json').read_text())['history']
    np.testing.assert_allclose(history[-1]['strengths'],case['estimate'],rtol=1e-12)
    microscope=psf.Microscope(**case['params']);alpha=microscope.alpha;modes=case['modes']
    truth=case['imposed_for_validation']+case['cumulative_correction'];estimate=case['estimate']
    rms=lambda c:float(zernike.zernike_RMS(zernike.Aberration(modes,c-truth),alpha))
    iterations=np.array([r['iteration'] for r in history]);errors=np.array([rms(np.array(r['strengths'])) for r in history])
    prior=rms(case['cumulative_correction']);losses=np.array([r['loss'] for r in history])
    ws=workspace(case)
    final_metrics,residual,_=fit_diagnostics(ws,estimate,case['images'])
    truth_metrics,truth_residual,_=fit_diagnostics(ws,truth,case['images'])
    np.testing.assert_allclose(final_metrics['loss'],losses[-1],rtol=1e-10)
    prediction=case['images']-residual
    del ws
    files=[]
    def save(fig,name):
        fig.savefig(OUTPUT/name,dpi=300,facecolor='white');files.append(name)
    # Export the original AO viewer's final frame, retaining its three polar
    # panels, measured image, objective and full signed-wavefront error trace.
    frames,points,_=build_replay(CASE/'solve/results',CASE/'submission/input.npz')
    final=dict(frames[-1]);final['label']='Final estimate'
    final['initial_image_title']=r'Measured zero diversity, $z=0\,\mu$m'
    final['loss_history']=(iterations.tolist(),losses.tolist())
    final['trace_series']=[dict(label='Estimated wavefront',x=iterations.tolist(),y=errors.tolist(),color='tab:blue')]
    final['trace_title']='Wavefront estimation error';final['trace_ylabel']='RMS error [waves]'
    def export_overview(fig,index):
        for ax in fig.axes:
            if ax.images:
                ax.set_xticks([-.002,0,.002],labels=['-2','0','2'])
                ax.set_yticks([-.002,0,.002],labels=['-2','0','2'])
                ax.tick_params(labelrotation=0)
                ax.set_xlabel(r'$x$ [$\mu$m]');ax.set_ylabel(r'$y$ [$\mu$m]')
            if ax.get_ylabel()=='Intensity (Raw)':ax.set_ylabel('Averaged intensity')
        save(fig,'01_final_overview.png')
    with plt.style.context('bmh'),plt.rc_context({'text.usetex':True,'font.size':9,'figure.dpi':300}):
        render_replay([final],points,title=r'Multiple fluorobeads | $\rho=10^{49}$',
                      on_figure=export_overview)
    fig,ax=plot_error_histories(iterations,{'Estimated wavefront':errors},title=r'Wavefront recovery | $\rho=10^{49}$')
    ax.axhline(prior,color='0.35',ls='--',lw=1,label='Zero-estimate baseline')
    ax.set_ylim(0,max(prior,errors.max())*1.15)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.23),fontsize=8)
    ax.text(.97,.94,f'Final error: {errors[-1]:.4f} waves',transform=ax.transAxes,ha='right',va='top',fontsize=9)
    save(fig,'02_rms_convergence.png');plt.close(fig)
    fig,ax=plot_error_histories(iterations,{'Optimizer objective':losses/losses[0]},
        title=r'Objective convergence | $\rho=10^{49}$',ylabel='Loss / selected-start loss')
    ax.axhline(truth_metrics['loss']/losses[0],color='0.35',ls='--',lw=1,label='Known wavefront; object refitted')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.23),fontsize=8)
    save(fig,'03_loss_convergence.png');plt.close(fig)
    fig,ax=plot_error_histories(np.arange(len(modes)),{'Imposed wavefront':truth,'Estimated wavefront':estimate},
        title=r'Signed Zernike coefficients | $\rho=10^{49}$',xlabel='Mode $(m,n)$',ylabel='Coefficient [waves]',nonnegative=False)
    ax.set_xticks(np.arange(len(modes)),labels=[f'$({m},{n})$' for m,n in modes],rotation=90)
    save(fig,'04_zernike_coefficients.png');plt.close(fig)
    acq=load_acquisition(SOURCE);ids=case['source_indices'];z=case['focal_z']*1000
    grid=acquisition_grid(float(case['field_of_view_um']),(64,64))
    zero=np.all(np.isclose(case['diversity_strengths'],case['baseline'],atol=1e-12,rtol=0),axis=1)
    selected=np.array([np.flatnonzero(zero&np.isclose(z,value))[0] for value in [-1.,0.,.5,1.]])
    peak=max(case['images'][zero].max(),prediction[zero].max())
    commands=[zernike.Aberration(modes,acq.total[ids[j]]) for j in selected]
    fig,_=plot_acquisition_comparison(grid,case['images'][selected]/peak,prediction[selected]/peak,commands,
        case['focal_z'][selected],alpha,title=r'Zero-diversity Z stack | $\rho=10^{49}$',
        column_titles=('Experimental','Fitted prediction'))
    save(fig,'05_zero_diversity_stack.png');plt.close(fig)
    probe_modes=[(2,2),(-1,3),(-3,3),(0,4)];selected=[]
    for mode in probe_modes:
        index=int(np.flatnonzero(np.all(modes==mode,axis=1))[0])
        command=case['baseline'].copy();command[index]+=.1
        selected.append(int(np.flatnonzero(np.isclose(z,0)&np.all(np.isclose(case['diversity_strengths'],command,atol=1e-12),axis=1))[0]))
    selected=np.array(selected)
    # One scale across all four displayed probes, preserving brightness differences.
    peak=max(case['images'][selected].max(),prediction[selected].max())
    commands=[zernike.Aberration(modes,acq.total[ids[j]]) for j in selected]
    fig,axes=plot_acquisition_comparison(grid,case['images'][selected]/peak,prediction[selected]/peak,commands,
        case['focal_z'][selected],alpha,title=r'$+0.10$-wave probes at $z=0$ | $\rho=10^{49}$',
        column_titles=('Experimental','Fitted prediction'))
    for row,(m,n) in enumerate(probe_modes):
        axes[row,0].set_ylabel(f'$({m},{n})$\n'+r'$y$ [$\mu$m]',fontsize=8)
    save(fig,'06_diversity_probe_fits.png');plt.close(fig)
    levels=np.unique(z)
    fig,ax=plot_error_histories(levels,{
        'Known wavefront':np.array([np.sqrt(np.mean(truth_residual[np.isclose(z,v)]**2)) for v in levels]),
        'Estimated wavefront':np.array([np.sqrt(np.mean(residual[np.isclose(z,v)]**2)) for v in levels])},
        title=r'Image residual across Z | $\rho=10^{49}$',xlabel=r'Requested Z offset [$\mu$m]',
        ylabel='Measured-region residual RMS',integer_x=False)
    save(fig,'07_residual_vs_z.png');plt.close(fig)
    write_json(OUTPUT/'figure_metadata.json',dict(source=CASE,rho=result['rho'],files=files,
        initial_zero_estimate_rms_waves=prior,final_error_rms_waves=errors[-1],
        final_measured_rmse=final_metrics['measured_rmse'],truth_measured_rmse=truth_metrics['measured_rmse'],
        retained_images=len(ids),zero_diversity_z_um=[-1,0,.5,1],probe_modes=probe_modes,
        note='Actual acquired data and same-object forward predictions from the saved final fit. No new wavefront optimization or applied correction.'))
    (OUTPUT/'README.md').write_text(
        '# Rho = 1e49: presentation PNGs\n\n'
        f'134 experimental PD images, X mirrored, 64x64 block means, per-image border backgrounds removed. '
        f'Zero-estimate error {prior:.5f} → final estimation error {errors[-1]:.5f} waves RMS. '
        'System AO is assumed to cancel native aberration; the imposed metadata phase is the validation reference.\n\n'
        'Seven 300-DPI PNGs: final AO overview, RMS convergence, loss convergence, signed coefficients, '
        'zero-diversity Z stack, representative diversity fits, and residual across Z. '
        'The selected random initializer is iteration 0; the zero-estimate baseline is shown separately. '
        'Phase insets in acquisition comparisons display the known commanded phase; overview polar panels show imposed, estimated and error phase. '
        'Shared image normalization preserves brightness differences. Negative values remain in inference; grayscale display uses a zero lower limit. '
        'No physical correction or post-correction acquisition is shown.\n\n'
        'Reproduce from repository root: `python -m experimental_10_15.export_rho49_figures`.\n')
    print(f'{len(files)} PNGs saved to {OUTPUT}; RMS {prior:.5f} -> {errors[-1]:.5f} waves',flush=True)


if __name__=='__main__':main()
