"""Replay a downloaded experimental fit in the established case-study AO viewer.

System AO is assumed to correct the native wavefront. Estimates are evaluated
against the imposed coefficients recorded in the acquisition metadata.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from experimental_10_15.bead_pd import acquisition_grid
from experimental_10_15.bead_pd_io import write_json
from systematic_testing.ao_animation.viewer import _phase,render_replay
from utils import psf,zernike

NOTICE = ('System AO is assumed already applied and native aberration corrected. '
          'The reference wavefront is the imposed phase recorded in metadata. '
          'Estimation error is estimate minus imposed phase, after subtracting the '
          'known cumulative correction from later-loop estimates. The image is an '
          'actual retained acquisition; its title identifies zero diversity or the applied probe. '
          'No proposed-correction image is shown. '
          'Playback advances by saved optimizer iteration, not elapsed time.')


def build_replay(results, input_path, *, focus_um=0., stride=1):
    results = Path(results)
    if stride < 1 or not np.isfinite(focus_um):
        raise ValueError('Require a positive stride and finite focus.')
    optimizer = json.loads((results/'optimizer.json').read_text())
    summary = json.loads((results/'result.json').read_text())
    history = optimizer['history']
    if not history: raise ValueError('The saved optimizer history is empty.')
    with np.load(input_path,allow_pickle=False) as data:
        modes=data['modes']; baseline=data['baseline']; cumulative=data['cumulative_correction']
        images=data['images']; focal_z=data['focal_z']
        grid=acquisition_grid(float(data['field_of_view_um']),images.shape[-2:])
        microscope=psf.Microscope(**json.loads(str(data['microscope_json'])))
        source_indices=data['source_indices']; plane_index=data['plane_index']
        imposed=data['imposed_for_validation']
        known=data['diversity_strengths']
        names=data['names'] if 'names' in data else np.array(['probe']*len(images))
        data_origin=str(data['data_origin']) if 'data_origin' in data else 'measured'
        background_border=int(data['background_border_pixels']) if 'background_border_pixels' in data else None
    with np.load(results/'solution.npz',allow_pickle=False) as data:
        if not np.array_equal(modes,data['modes']): raise ValueError('Input and solve mode bases differ.')
        if not np.array_equal(focal_z,data['focal_z']): raise ValueError('Input and solve focal positions differ.')
        if not np.array_equal(cumulative,data['cumulative_correction']):
            raise ValueError('Input and solve cumulative corrections differ.')
        final=data['estimated_strengths']
    if not np.allclose(history[-1]['strengths'],final,rtol=1e-10,atol=1e-12):
        raise ValueError('Final saved iterate differs from the exported solution.')
    selected=sorted(set(range(0,len(history),stride))|{len(history)-1})
    zero_ids=np.flatnonzero(np.all(np.isclose(known,baseline,rtol=0,atol=1e-12),axis=1))
    if len(zero_ids):
        image_index=int(zero_ids[np.argmin(abs(focal_z[zero_ids]-focus_um/1000))])
        image_kind='zero-diversity'
    else:
        # Strict SNR screening can remove every zero-diversity frame. Display a
        # retained probe, without restoring a rejected image to the fit.
        distance=abs(focal_z-focus_um/1000)
        nearest=np.flatnonzero(np.isclose(distance,distance.min(),rtol=0,atol=1e-12))
        image_index=int(nearest[np.argmin(np.linalg.norm(known[nearest]-baseline,axis=1))])
        image_kind='probe'
    actual_focus_um=float(focal_z[image_index]*1000)
    image_title=f'{"Simulated" if data_origin=="simulation" else "Measured"} {image_kind}, z={actual_focus_um:g} µm'
    if image_kind=='probe':image_title+='\n'+str(names[image_index]).replace('_',' ')
    imposed_aberration=zernike.Aberration(modes,imposed)
    ticks=['Prior']+[str(history[i]['iteration']) for i in selected]
    frames=[]; rms=[]; estimates=[]
    for index,chosen in enumerate([None,*selected]):
        # The optimizer fits imposed + already-applied cumulative correction.
        # Remove that known offset before comparing to the original imposed phase.
        estimated_imposed=np.zeros(len(modes)) if chosen is None else np.asarray(history[chosen]['strengths'])-cumulative
        error_aberration=zernike.Aberration(modes,estimated_imposed-imposed)
        estimated_aberration=zernike.Aberration(modes,estimated_imposed)
        error=float(zernike.zernike_RMS(error_aberration,microscope.alpha))
        label='Before fitting (zero estimate)' if chosen is None else (
            f"Saved iteration {history[chosen]['iteration']} | loss {history[chosen]['loss']:.4g}")
        frame=dict(phase=_phase(imposed_aberration,microscope.alpha,101),
            phase_function=imposed_aberration.construct_map(microscope.alpha),
            estimate=_phase(estimated_aberration,microscope.alpha,101),
            estimate_function=estimated_aberration.construct_map(microscope.alpha),
            command=_phase(error_aberration,microscope.alpha,101),
            command_function=error_aberration.construct_map(microscope.alpha),
            image=images[image_index],acquisition=None,x=grid.get_xy()[0],y=grid.get_xy()[1],alpha=microscope.alpha,
            rms=error,count=len(images),loop=int(summary.get('loop_number',1)),progress=0,label=label,
            phase_titles=('Imposed wavefront (metadata)','Estimated imposed wavefront','Estimation error'),
            rms_label='estimation error RMS',state_label='Zero-estimate prior',
            trace_title='Estimation error vs imposed wavefront',trace_ylabel='Estimation error RMS (waves)',
            progress_tick_labels=ticks,initial_image_title=image_title,
            show_loss_panel=True,loss=None if chosen is None else history[chosen]['loss'])
        if chosen is not None:frame['optimization_rms']=error
        frames.append(frame);rms.append(error);estimates.append(estimated_imposed)
    points=[dict(images_taken=len(images),residual_rms=rms[0])]
    notice=NOTICE.replace('actual retained acquisition','retained simulated frame') if data_origin=='simulation' else NOTICE
    if background_border is not None:
        notice+=f' Each input image has its own {background_border}-pixel outer-border median subtracted at solver resolution; signed values are preserved.'
    details=dict(notice=notice,data_origin=data_origin,focus_um=actual_focus_um,source_indices=source_indices,plane_indices=plane_index,
        selected_iterations=[history[i]['iteration'] for i in selected],optimizer_success=summary['optimizer_success'],
        imposed_waves=imposed,estimated_imposed_waves=estimates,estimation_error_rms_waves=rms,
        reference='metadata imposed wavefront; system AO assumed to correct native aberration',
        displayed_image_source_index=int(source_indices[image_index]),
        displayed_image_is_zero_diversity=image_kind=='zero-diversity',
        displayed_image_name=str(names[image_index]),
        displayed_image_diversity_waves=known[image_index]-baseline,
        total_solve_seconds=summary['elapsed_seconds'],per_iteration_times_available=False)
    return frames,points,details


def export_replay(results,input_path,output,*,focus_um=0.,stride=1):
    frames,points,details=build_replay(results,input_path,focus_um=focus_um,stride=stride)
    title='Experimental bead fit: recovery of imposed wavefront'
    if details['data_origin']=='simulation':title='Simulated bead fit: recovery of imposed wavefront'
    document=render_replay(frames,points,title=title,
        on_frame=lambda item: print(f"Rendering {item['completed']}/{item['total']}",flush=True))
    import html
    document=document.replace('<body>','<body><p style="max-width:1200px">'+html.escape(details['notice'])+'</p>',1)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(document)
    details.update(results=str(Path(results).resolve()),input_path=str(Path(input_path).resolve()),
                   generated_at=datetime.now(timezone.utc).isoformat())
    write_json(output.with_suffix('.json'),details)
    print(output,flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('--input',dest='input_path',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--focus-um',type=float,default=0.)
    p.add_argument('--stride',type=int,default=1)
    export_replay(**vars(p.parse_args()))


if __name__=='__main__':main()
