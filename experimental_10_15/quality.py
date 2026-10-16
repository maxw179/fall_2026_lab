"""Auditable peak/background SNR screening of signed, averaged bead images."""
from __future__ import annotations
import argparse
import csv
import html
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter
from experimental_10_15.bead_pd_io import (load_acquisition,analysis_settings,
    block_average,optics_parameters,write_json)

SNR_METHOD = dict(signal='maximum interior 3x3 patch mean minus background median',
    background='outer 15 percent band on each image edge',
    noise='background median absolute deviation / 0.6744897501960817',
    units='linear amplitude ratio, not dB',
    limitation='Spatial background-noise estimate on the saved average; not temporal or bead shot-noise SNR.')


def score_frames(images):
    """Estimate each frame independently; never alter the measured pixel values."""
    images = np.asarray(images,dtype=float)
    if images.ndim != 3 or min(images.shape[-2:]) < 7 or not np.isfinite(images).all():
        raise ValueError('SNR screening needs finite images of at least 7x7 pixels.')
    nx,ny = images.shape[-2:]
    width = max(2,int(np.ceil(min(nx,ny)*.15)))
    border = np.ones((nx,ny),bool);border[width:-width,width:-width] = False
    records = []
    for frame in images:
        background = float(np.median(frame[border]))
        sigma = float(np.median(abs(frame[border]-background))/.6744897501960817)
        patch = uniform_filter(frame,size=3,mode='reflect')[1:-1,1:-1]
        x,y = np.unravel_index(np.argmax(patch),patch.shape)
        peak = max(float(patch[x,y])-background,0.)
        if sigma == 0 and peak > 0:
            raise ValueError('Cannot estimate SNR: zero background noise with nonzero signal.')
        records.append((peak/sigma if sigma else 0.,background,sigma,peak,x+1,y+1))
    return {key:np.asarray([row[i] for row in records]) for i,key in enumerate(
        ['snr','background','noise_sigma','peak_signal','peak_x','peak_y'])}


def write_snr_csv(path,acquisition,indices,metrics,min_snr):
    with Path(path).open('w',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_index_zero_based','image_number_one_based','plane_index',
            'z_um','name','snr','included',*list(metrics)[1:]])
        for j,index in enumerate(indices):
            writer.writerow([int(index),int(index)+1,int(acquisition.plane_index[index]),
                float(acquisition.focal_z[index]*1000),acquisition.names[index],
                float(metrics['snr'][j]),bool(metrics['snr'][j]>=min_snr),
                *[float(metrics[key][j]) for key in list(metrics)[1:]]])


def review_runs(runs,output,*,min_snr=5.,z_um=(-1.,0.,1.),use_tex=True,refresh=False):
    """Prepare fitting inputs and show near-cutoff examples before any upload."""
    from experimental_10_15.bead_pd import prepare_run,acquisition_grid
    from systematic_testing.correction_examples.plot_pd_correction_examples import plot_acquired_diversities
    from utils import psf,zernike
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]; sections=[]
    for run in map(Path,runs):
        acquisition=load_acquisition(run)
        target=np.any(np.isclose(acquisition.focal_z[:,None]*1000,np.asarray(z_um)[None,:],
                                 rtol=0,atol=1e-8),axis=1)
        excluded_planes=sorted(set(acquisition.plane_index[~target].tolist()))
        name=run.parent.name
        destination=output/name
        if destination.exists():
            if not refresh:raise ValueError(f'Review output already exists: {destination}')
            import json
            previous=json.loads((destination/'review.json').read_text())
            if (previous['source']!=str(run.resolve()) or previous['threshold']!=min_snr
                    or previous['z_um']!=list(z_um)):
                raise ValueError('Refreshing a review requires identical source, Z planes and threshold.')
        path=prepare_run(run,destination,exclude_planes=excluded_planes,min_snr=min_snr)
        options=analysis_settings(acquisition)
        candidates=np.flatnonzero(acquisition.acquired&acquisition.is_pd&target)
        images=block_average(acquisition.images[candidates],options['size'])-options['background']
        metrics=score_frames(images);passed=metrics['snr']>=min_snr
        pd_count=int(np.sum(acquisition.acquired&acquisition.is_pd))
        row=dict(name=name,source=str(run.resolve()),pd_available=pd_count,
            outside_z=pd_count-len(candidates),candidates=len(candidates),
            below_snr=int(np.sum(~passed)),included=int(np.sum(passed)),
            excluded_planes=excluded_planes,input=str(path.resolve()),
            retained_per_z={str(z):int(np.sum(passed&np.isclose(acquisition.focal_z[candidates]*1000,z,
                                                             rtol=0,atol=1e-8))) for z in z_um},
            threshold=min_snr,z_um=list(z_um))
        chosen=[];panel_titles=[];focus=[];sample_details=[]
        for z in z_um:
            local=np.flatnonzero(np.isclose(acquisition.focal_z[candidates]*1000,z,rtol=0,atol=1e-8))
            good=local[passed[local]];bad=local[~passed[local]]
            # Weakest included and strongest excluded show the decision boundary.
            left=int(good[np.argmin(metrics['snr'][good])]) if len(good) else int(local[np.argmax(metrics['snr'][local])])
            right=int(bad[np.argmax(metrics['snr'][bad])]) if len(bad) else int(local[np.argmin(metrics['snr'][local])])
            chosen.extend([left,right]);focus.append(z/1000)
            labels=[]
            for j in [left,right]:
                index=int(candidates[j]);state='Included' if passed[j] else 'Excluded'
                labels.append(f'{state}: SNR {metrics["snr"][j]:.2f}\nImage {index+1}: '+
                              str(acquisition.names[index]).replace('_',' '))
                sample_details.append(dict(image_number=index+1,z_um=z,name=str(acquisition.names[index]),
                    snr=float(metrics['snr'][j]),included=bool(passed[j])))
            panel_titles.append(labels)
        scale=float(np.max(images[chosen]))
        phases=[zernike.Aberration(acquisition.modes,acquisition.total[candidates[j]]) for j in chosen]
        microscope=psf.Microscope(**optics_parameters(acquisition.optics,options['pupil_size']))
        fig,_=plot_acquired_diversities(acquisition_grid(acquisition.optics['fieldOfViewUm'],images.shape[-2:]),
            images[chosen].reshape(len(z_um),2,*images.shape[-2:]),phases,np.asarray(focus),microscope.alpha,
            column_titles=['Near-cutoff included','Near-cutoff excluded'],
            panel_titles=panel_titles,title=name.replace('3d_fluorobeads_','').replace('_',' ')+
                f' | keep {row["included"]}/{row["candidates"]}, SNR '+r'$\geq$'+f' {min_snr:g}',
            intensity_scale=scale,use_tex=use_tex)
        fig.savefig(destination/'examples.png',dpi=300)
        fig.savefig(destination/'examples.pdf',dpi=300)
        plt.close(fig)
        row['examples']=sample_details
        write_json(destination/'review.json',row);rows.append(row)
        records=''.join('<tr>'+''.join(f'<td>{html.escape(str(value))}</td>' for value in
            [s['image_number'],f'{s["z_um"]:+g}',s['name'],f'{s["snr"]:.3f}',
             'Included' if s['included'] else 'Excluded'])+'</tr>' for s in sample_details)
        sections.append(f'<h2>{html.escape(name)}</h2><p><a href="{name}/snr_qc.csv">All candidate SNR values (CSV)</a></p>'
            f'<img style="max-width:900px;width:100%" src="{name}/examples.png" alt="Included/excluded examples">'
            '<table><tr><th>Image</th><th>Z (um)</th><th>Command</th><th>SNR</th><th>Decision</th></tr>'+records+'</table>')
        print(f'{name}: {pd_count} PD, {row["outside_z"]} outside Z, '
              f'{row["below_snr"]} below SNR {min_snr:g}, {row["included"]} retained',flush=True)
    summary=dict(snr_method=SNR_METHOD,min_snr=min_snr,z_um=list(z_um),runs=rows,
        totals={key:sum(row[key] for row in rows) for key in
                ['pd_available','outside_z','candidates','below_snr','included']})
    write_json(output/'summary.json',summary)
    columns=['name','pd_available','outside_z','candidates','below_snr','included']
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    header=f'<tr><th>Run</th><th>Available PD</th><th>Outside Z</th><th>Middle 3</th><th>SNR below {min_snr:g}</th><th>Included</th></tr>'
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(row[k]))}</td>' for k in columns)+'</tr>' for row in rows)
    document='<!doctype html><html><head><meta charset="utf-8"><title>Three-plane SNR screening</title>'
    document+='<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}h2{margin-top:40px}</style></head><body>'
    document+=f'<h1>Three-plane SNR screening: keep SNR ≥ {min_snr:g}</h1><p>Z = '+', '.join(f'{z:+g}' for z in z_um)+' µm. Counts concern fitting images only; reference images never enter the fit.</p>'
    document+='<p>Signal: brightest interior 3 × 3 patch mean minus median edge background. Noise: edge-background MAD / 0.67449, using a 15% border. Evaluated on the saved averaged, signed images at the solver resolution. No extra averaging correction, normalization, alignment or pixel modification is applied to the solver images. This estimates spatial background contrast, not temporal SNR or bead shot noise.</p>'
    document+='<table>'+header+table+'</table><p>Examples use one intensity scale within each run; polar insets show commanded phase. If a plane has no excluded images, both examples are labelled Included.</p>'+''.join(sections)+'</body></html>'
    (output/'index.html').write_text(document)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--min-snr',type=float,default=5.)
    parser.add_argument('--z-um',nargs='+',type=float,default=[-1.,0.,1.])
    parser.add_argument('--no-tex',action='store_true')
    parser.add_argument('--refresh',action='store_true',help='Refresh an existing review with identical source and selection settings')
    args=vars(parser.parse_args());args['use_tex']=not args.pop('no_tex')
    review_runs(**args)


if __name__=='__main__':main()
