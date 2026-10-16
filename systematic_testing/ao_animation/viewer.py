"""Portable AO replay: pre-rendered frames with browser playback controls."""
from __future__ import annotations

import base64
from io import BytesIO
import html
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from systematic_testing.correction import SimulatedSLM
from utils import zernike, plotting


class RecordingSLM(SimulatedSLM):
    """Drop-in simulator retaining actual acquisitions; no extra exposures."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acquisitions = []

    def acquire(self, focal_z_levels, diversities=None):
        positions = np.asarray(focal_z_levels, dtype=float)
        probes = list(diversities) if diversities is not None else [None] * len(positions)
        if len(probes) != len(positions):
            raise ValueError('One diversity is required per focal position.')
        command = self.slm_strengths.copy()
        first_count = self.images_taken
        images = super().acquire(positions, diversities)
        for i, (image, focus, probe) in enumerate(zip(images, positions, probes)):
            label = f'z={focus:g} mm; no probe'
            if probe is not None:
                label = f'z={focus:g} mm; modes={probe.modes}; bias={probe.strengths.tolist()} waves'
            self.acquisitions.append(dict(image=image.copy(), label=label,
                                          images_taken=first_count+i+1,
                                          slm_strengths=command.copy()))
        return images


def _phase(aberration, alpha, size):
    axis = np.linspace(-1, 1, size)
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x, y)
    theta = np.arcsin(np.minimum(radius, 1) * np.sin(alpha))
    phase = np.asarray(aberration.construct_map(alpha)(theta, np.arctan2(y, x)))
    return np.where(radius <= 1, phase / (2*np.pi), np.nan)


def replay_history(history, simulator, *, focus=0.0, max_updates=3, phase_size=101, include_acquisitions=False):
    """Build replay states from a feedback history and its originating simulator.

    Diagnostic images are noiseless, zero-probe, fixed-focus simulations. They do
    not mutate the simulator, consume its RNG, or count toward acquisition budget.
    Acquisition steps are hidden by default; include_acquisitions enables capture frames for other consumers.
    """
    if not history:
        raise ValueError('History must include the initial state.')
    if max_updates is not None and (not isinstance(max_updates, int) or max_updates < 0):
        raise ValueError('max_updates must be a nonnegative integer or None.')
    points = history if max_updates is None else history[:max_updates+1]
    counts = [int(p['images_taken']) for p in points]
    if any(b < a for a, b in zip(counts, counts[1:])):
        raise ValueError('History image counts must be nondecreasing.')
    frames = []
    acquisitions = getattr(simulator, 'acquisitions', []) if include_acquisitions else []
    for loop, point in enumerate(points):
        command = zernike.Aberration(simulator.modes, point['slm_strengths'])
        residual = simulator._truth + command
        x, y, _, image = simulator.microscope.compute_image_3D(
            image=simulator.sample, aberration=residual,
            focal_z_levels=np.array([focus]), mode=simulator.optical_mode)
        state = dict(phase=_phase(residual, simulator.microscope.alpha, phase_size),
                     command=_phase(command, simulator.microscope.alpha, phase_size),
                     estimate=None, image=image[0], acquisition=None,
                     x=x, y=y, alpha=simulator.microscope.alpha,
                     phase_function=residual.construct_map(simulator.microscope.alpha),
                     command_function=command.construct_map(simulator.microscope.alpha),
                     rms=float(point['residual_rms']), count=counts[loop], loop=loop,
                     label='Initial state' if loop == 0 else f'Correction {loop} applied',
                     progress=loop)
        if 'core_elapsed_seconds' in point:
            state['core_elapsed_seconds'] = point['core_elapsed_seconds']
        if loop:
            previous = frames[-1]
            for acquisition in acquisitions:
                if counts[loop-1] < acquisition['images_taken'] <= counts[loop]:
                    frames.append(dict(previous, acquisition=acquisition['image'],
                                       count=acquisition['images_taken'], loop=loop,
                                       label=f"Acquire: {acquisition['label']}"))
            iterates = point.get('optimizer_info', {}).get('history', [])
            if not iterates and point.get('estimated_strengths') is not None:
                iterates = [dict(strengths=point['estimated_strengths'], iteration='final')]
            for entry in iterates:
                estimate = zernike.Aberration(simulator.modes, entry['strengths'])
                estimate_map = _phase(estimate, simulator.microscope.alpha, phase_size)
                estimate_function = estimate.construct_map(simulator.microscope.alpha)
                fit_state = dict(previous, estimate=estimate_map, estimate_function=estimate_function,
                                   loop=loop, acquisition=None, count=counts[loop],
                                   optimization_rms=float(zernike.zernike_RMS(
                                       simulator._truth + zernike.Aberration(simulator.modes, points[loop-1]['slm_strengths']) - estimate,
                                       simulator.microscope.alpha)),
                                   label=f"Fit correction {loop}: iteration {entry['iteration']}")
                if 'core_elapsed_seconds' in entry:
                    fit_state['core_elapsed_seconds'] = entry['core_elapsed_seconds']
                frames.append(fit_state)
            # After application, express the last fit relative to the new SLM state.
            if iterates:
                remaining = estimate + command - zernike.Aberration(simulator.modes, points[loop-1]['slm_strengths'])
                state['estimate'] = _phase(remaining, simulator.microscope.alpha, phase_size)
                state['estimate_function'] = remaining.construct_map(simulator.microscope.alpha)
        frames.append(state)
    return frames, points


def _rms_trace(frame, points):
    """Committed RMS states, extended horizontally through the current fit."""
    end = frame['progress'] + 1
    counts = [float(p['images_taken']) for p in points[:end]]
    values = [float(p['residual_rms']) for p in points[:end]]
    if frame['count'] > counts[-1]:
        counts.append(float(frame['count']))
        values.append(values[-1])
    return counts, values


def render_replay(frames, points, *, title='Single-case AO replay', on_frame=None, on_figure=None):
    """Return the viewer; optional on_figure(fig,index) exports the same layout."""
    frames = [f for f in frames if f.get('acquisition') is None]
    if not frames:
        raise ValueError('At least one non-acquisition frame is required.')
    image_max = max(float(np.max(f['image'])) for f in frames)
    image_max = max(image_max, *(float(np.max(f['acquisition'])) for f in frames
                                if f['acquisition'] is not None), np.finfo(float).tiny)
    counts = [p['images_taken'] for p in points]
    rms = [p['residual_rms'] for p in points]
    progress_rms = [f.get('optimization_rms', f['rms']) for f in frames]
    correction_indices = [i for i, f in enumerate(frames) if 'optimization_rms' not in f]
    encoded = []
    for frame_index, frame in enumerate(frames):
        variants = []
        for show_acquisition in (False,):
            fig = Figure(figsize=(12, 7.6), layout=None)
            FigureCanvasAgg(fig)
            # Explicit positions reserve identical panel and colorbar space in every frame.
            axes = np.empty((2, 3), dtype=object)
            phase_caxes = []
            for col, left in enumerate((.055, .375, .695)):
                axes[0, col] = fig.add_axes([left, .565, .25, .30], projection='polar')
                phase_caxes.append(fig.add_axes([left, .505, .25, .018]))
                axes[1, col] = fig.add_axes([left, .105, .25, .285])
            image_cax = fig.add_axes([.322, .105, .012, .285])
            time_label = (f" | core time: {frame['core_elapsed_seconds']:.2f} s"
                          if 'core_elapsed_seconds' in frame else '')
            fig.suptitle(f"{title} • {frame['label']}\nImages acquired: {frame['count']} | {frame.get('rms_label', 'residual RMS')}: {frame['rms']:.4f} waves{time_label}")
            for ax, key, label in zip(axes[0], ('phase', 'estimate', 'command'),
                                      frame.get('phase_titles', ('True residual (simulation)', 'Estimated current residual', 'Accumulated SLM phase'))):
                function = frame.get(key+'_function')
                if frame[key] is None:
                    function = lambda theta, phi: np.zeros_like(theta)
                elif function is None:
                    from scipy.ndimage import map_coordinates
                    array = np.nan_to_num(frame[key])
                    def function(theta, phi, array=array):
                        radius = np.sin(theta)/np.sin(frame.get('alpha', .5))
                        coords = [(radius*np.sin(phi)+1)*(array.shape[0]-1)/2,
                                  (radius*np.cos(phi)+1)*(array.shape[1]-1)/2]
                        return 2*np.pi*map_coordinates(array, coords, order=1, mode='nearest')
                _, polar = plotting.zernike_plot(function, frame.get('alpha', .5),
                                                 res=101, ax=ax, show=False, colorbar=False)
                polar.set_title(label, fontsize=11, pad=12)
                col = ('phase', 'estimate', 'command').index(key)
                fig.colorbar(polar.collections[0], cax=phase_caxes[col], orientation='horizontal',
                             label='Phase (rad)', ticks=[-3, 0, 3])
                if frame[key] is None:
                    polar.collections[0].set_visible(False)
                    polar.text(.5, .5, 'Not fitted yet', ha='center', va='center', transform=polar.transAxes)
            acquired = show_acquisition and frame['acquisition'] is not None
            initial_gain = frame.get('initial_display_gain', 1.)
            images = [frames[0]['image']*initial_gain, frame['acquisition'] if acquired else frame['image']]
            x, y = frame.get('x'), frame.get('y')
            if x is None: x = np.arange(images[0].shape[0], dtype=float)
            if y is None: y = np.arange(images[0].shape[1], dtype=float)
            show_loss = frame.get('show_loss_panel',False)
            image_axes = axes[1,:1] if show_loss else axes[1,:2]
            image_data = images[:1] if show_loss else images
            plotting.plot_many_intensity(image_axes, x, y, image_data, normalize=False, vmax=image_max)
            initial_title = frame.get('initial_image_title', 'Initial unbiased image')
            if initial_gain != 1.:
                initial_title += f' [display ×{initial_gain:.1f}]'
            axes[1, 0].set_title(initial_title)
            if show_loss:
                loss_ids = [i for i in range(frame_index+1) if frames[i].get('loss') is not None]
                loss_values = [frames[i]['loss'] for i in loss_ids]
                # Final-state comparisons can retain the original full loss trace.
                if 'loss_history' in frame:
                    loss_ids, loss_values = frame['loss_history']
                loss_ax=axes[1,1]
                # Leave space for the measured-image colorbar and loss labels.
                loss_ax.set_position([.43,.105,.195,.285])
                loss_ax.plot(loss_ids,loss_values,'.-',color='tab:blue')
                loss_ax.set(title='Optimizer loss',xlabel='Saved optimization steps',ylabel='Reduced loss',
                            xlim=(0,max(len(frames)-1,1)))
                all_losses=[f['loss'] for f in frames if f.get('loss') is not None]
                if 'loss_history' in frame:
                    all_losses = list(loss_values)
                    loss_ax.set(xlabel='Saved iteration', xlim=(min(loss_ids), max(max(loss_ids), min(loss_ids)+1)))
                if all_losses:
                    lo,hi=min(all_losses),max(all_losses)
                    margin=max((hi-lo)*.08,abs(hi)*1e-6,1e-12)
                    loss_ax.set_ylim(lo-margin,hi+margin)
                loss_ax.grid(alpha=.25)
            else:
                axes[1, 1].set_title(frame.get('acquisition_title', 'Actual noisy acquisition') if acquired else frame.get('image_title', 'Current unbiased image (noiseless)'))
            fig.colorbar(axes[1, 0].images[0], cax=image_cax, label='Intensity (Raw)')
            ax = axes[1, 2]
            ax.plot(np.arange(frame_index+1), progress_rms[:frame_index+1],
                    color='tab:blue', linewidth=1.5, marker='.', markersize=4)
            committed = [i for i in correction_indices if i <= frame_index]
            ax.plot(committed, [progress_rms[i] for i in committed], 's',
                    color='tab:green', markersize=5, label=frame.get('state_label', 'Applied state'))
            ax.plot(frame_index, progress_rms[frame_index], 'o', color='tab:red')
            ax.set(xlabel='Optimization progress (saved steps)', ylabel=frame.get('trace_ylabel', 'RMS (waves)'),
                   xlim=(0, max(len(frames)-1, 1)),
                   ylim=(0, max(max(progress_rms), 1e-9)*1.15))
            ax.set_title(frame.get('trace_title', 'Intermediate correction RMS'), fontsize=11)
            if frame.get('progress_tick_labels') is not None:
                ids = np.unique(np.linspace(0,len(frames)-1,min(7,len(frames)),dtype=int))
                ax.set_xticks(ids,[frame['progress_tick_labels'][i] for i in ids])
            else:
                ax.set_xticks(correction_indices, ['Initial'] + [f"AO {i}" for i in range(1, len(correction_indices))])
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(alpha=.25)
            if 'trace_series' in frame:
                # An opt-in comparison of complete, precomputed RMS histories.
                # Default AO and per-iteration replay rendering stays unchanged.
                ax.clear()
                for series in frame['trace_series']:
                    ax.plot(series['x'], series['y'], '.-', label=series['label'],
                            color=series.get('color'), linewidth=1.5, markersize=4)
                xs = [x for series in frame['trace_series'] for x in series['x']]
                ys = [y for series in frame['trace_series'] for y in series['y']]
                ax.set(xlabel='Saved iteration', ylabel=frame.get('trace_ylabel', 'RMS (waves)'),
                       xlim=(min(xs), max(max(xs), min(xs)+1)), ylim=(0, max(max(ys), 1e-9)*1.15))
                ax.set_title(frame.get('trace_title', 'RMS comparison'), fontsize=11)
                ax.legend(loc='upper right', fontsize=7)
                ax.grid(alpha=.25)
            if on_figure is not None:on_figure(fig,frame_index)
            buffer = BytesIO(); fig.savefig(buffer, format='png', dpi=100, bbox_inches=None)
            variants.append('data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode())
            fig.clear()
        encoded.append(variants[0])
        if on_frame is not None:
            on_frame(dict(completed=frame_index+1, total=len(frames)))
    payload = json.dumps(encoded)
    return '''<!doctype html><html><meta charset="utf-8"><title>AO replay</title>
<style>body{font:15px system-ui;margin:16px;background:#fafafa}img{width:100%;max-width:1200px;aspect-ratio:1200/760} #position{min-width:75px;font-variant-numeric:tabular-nums} .controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}input[type=range]{flex:1;min-width:180px}</style>
<body><div class="controls"><button id="play">Play</button><button id="back">Previous</button><button id="next">Next</button><input id="seek" type="range" min="0" value="0"><span id="position"></span><label>Frame duration <select id="speed"><option value="250">0.25 s</option><option value="700" selected>0.7 s</option><option value="1400">1.4 s</option></select></label></div>
<img id="frame" alt="AO wavefront, images and correction progress">
<script>const frames=PAYLOAD;let index=0,timer=null;const el=id=>document.getElementById(id);
el('seek').max=frames.length-1;
function draw(){el('frame').src=frames[index];el('seek').value=index;el('position').textContent=`${index+1} / ${frames.length}`;}
function stop(){clearInterval(timer);timer=null;el('play').textContent='Play';}
function start(){if(index===frames.length-1)index=0;el('play').textContent='Pause';timer=setInterval(()=>{if(index<frames.length-1){index++;draw();}else stop();},Number(el('speed').value));draw();}
el('play').onclick=()=>timer?stop():start();el('seek').oninput=()=>{stop();index=Number(el('seek').value);draw();};
el('back').onclick=()=>{stop();index=Math.max(0,index-1);draw();};el('next').onclick=()=>{stop();index=Math.min(frames.length-1,index+1);draw();};el('speed').onchange=()=>{if(timer){stop();start();}};draw();</script></body></html>'''.replace('PAYLOAD', payload)


def show_replay(frames, points, *, title='Single-case AO replay', output_path=None, on_frame=None):
    """Display in Jupyter and optionally save a standalone HTML file."""
    from IPython.display import HTML
    document = render_replay(frames, points, title=title, on_frame=on_frame)
    if output_path is not None:
        Path(output_path).write_text(document)
    return HTML('<iframe style="width:100%;height:800px;border:0" srcdoc="' +
                html.escape(document, quote=True) + '"></iframe>')
