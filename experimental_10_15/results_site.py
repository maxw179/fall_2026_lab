"""Shared local results page for filtered and binned experimental batches."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote
import html
import json
import argparse
import numpy as np

LABELS={'no_aberration':'No aberration','system_AO':'System AO',
        'spherical_0p15w':'Spherical · 0.15 waves','random_0p15w':'Random · 0.15 waves',
        'random_aligned_0p15w':'Random aligned · 0.15 waves',
        'random_aligned_0p25w':'Random aligned · 0.25 waves'}


def build_site(batch):
    HERE=Path(batch).resolve()
    summary=json.loads((HERE/'summary.json').read_text())
    threshold=summary.get('min_snr')
    binned=all('size' in row for row in summary['runs'])
    z_values=summary.get('z_um') or sorted({z for row in summary['runs'] for z in row['z_um']})
    jobs=json.loads((HERE/'jobs.json').read_text())
    review={row['name']:row for row in summary['runs']}
    snapshot=HERE/'latest_completion_status.json'
    statuses={row['name']:row for row in json.loads(snapshot.read_text())} if snapshot.exists() else {}
    cases=[]
    for job in jobs:
        name=job['name'];folder=HERE/name;replay=folder/'solve/figures/correction_replay.html'
        config=review[name]
        acquisition=config.get('acquisition',name).replace('3d_fluorobeads_','')
        size=int(config.get('size',128))
        label=config.get('label',LABELS.get(acquisition,acquisition))
        if binned:label+=f' · {size} × {size}'
        with np.load(folder/'submission/input.npz',allow_pickle=False) as prepared:
            z_counts=[int(np.sum(np.isclose(prepared['focal_z']*1000,z,rtol=0,atol=1e-8))) for z in z_values]
            assert len(prepared['images'])==job['included']
            assert prepared['images'].shape[1:]==(size,size)
        remote=statuses.get(name,{})
        item=dict(name=name,label=label,size=size,
            job_id=job['job_id'],images=job['included'],ready=False,
            z_counts=z_counts,
            status=remote.get('slurm_state',remote.get('status','Awaiting results')),assets={})
        if replay.is_file() and replay.with_suffix('.json').is_file():
            details=json.loads(replay.with_suffix('.json').read_text())
            assert Path(details['input_path']).resolve()==(folder/'submission/input.npz').resolve()
            result=json.loads((folder/'solve/results/result.json').read_text())
            optimizer=json.loads((folder/'solve/results/optimizer.json').read_text())
            assert result['status']=='complete'
            assert len(details['source_indices'])==item['images']
            error=details['estimation_error_rms_waves']
            item.update(ready=True,status='Converged' if result['optimizer_success'] else 'Stopped without convergence',
                converged=bool(result['optimizer_success']),seconds=float(result['elapsed_seconds']),
                initial_rms=float(error[0]),final_rms=float(error[-1]),
                iterations=int(optimizer['history'][-1]['iteration']),
                replay=quote(replay.relative_to(HERE).as_posix()))
            for label,path in [('Correction MAT',folder/'solve/results/correction.mat'),
                ('Coefficients CSV',folder/'solve/results/coefficients.csv'),
                ('Fit summary',folder/'solve/results/result.json'),
                ('Optimizer history',folder/'solve/results/optimizer.json'),
                ('Image SNR scores',folder/'submission/snr_qc.csv')]:
                if path.exists():item['assets'][label]=quote(path.relative_to(HERE).as_posix())
        cases.append(item)
    ready=[row for row in cases if row['ready']]
    resolutions=sorted({row['size'] for row in cases},reverse=True)
    data=dict(min_snr=threshold,z_um=z_values,snr_method=summary.get('snr_method'),cases=cases,
        resolutions=resolutions,
        generated_at=datetime.now(ZoneInfo('America/New_York')).isoformat())
    (HERE/'site_data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    rows=[]
    for i,row in enumerate(cases):
        e=html.escape
        name=(f'<button class="case-button" type="button" data-case="{i}" aria-pressed="false">'
              f'{e(row["label"])}</button>') if row['ready'] else e(row['label'])
        numbers=([f'{row["seconds"]:.1f} s',str(row['iterations']),f'{row["initial_rms"]:.4f}',
                  f'{row["final_rms"]:.4f}'] if row['ready'] else ['—']*4)
        metric_cells=''.join('<td class="numeric">'+value+'</td>' for value in numbers[:3])
        color='increased' if row['ready'] and row['final_rms']>row['initial_rms']+1e-12 else 'numeric'
        metric_cells+=f'<td class="numeric {color}">{numbers[3]}</td>'
        rows.append(f'<tr data-row="{i}" data-size="{row["size"]}"><td>{name}<small>Job {e(row["job_id"])}</small></td>'
            f'<td><span class="status">{e(row["status"])}</span></td>'
            f'<td class="numeric">{row["size"]} × {row["size"]}</td>'
            f'<td class="numeric">{row["images"]}</td><td class="numeric">'+
            ' / '.join(map(str,row['z_counts']))+'</td>'+metric_cells+'</tr>')
    options=''.join(f'<option value="{i}">{html.escape(row["label"])}</option>'
                    for i,row in enumerate(cases) if row['ready'])
    first=next((i for i,row in enumerate(cases) if row['ready']),None)
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c').replace('&','\\u0026')
    document='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--ink:#182431;--muted:#566575;--line:#dce3e9;--accent:#164e72}
*{box-sizing:border-box}body{margin:0;background:#f5f7f9;color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:36px 28px 56px}h1{font-size:clamp(25px,3vw,38px);line-height:1.2;margin:8px 0 14px}
h2{font-size:23px;line-height:1.3;margin:0}p{margin:8px 0 14px}.eyebrow{color:var(--muted);font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
.muted,small{color:var(--muted)}small{display:block;font-size:12px}.criteria{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 26px}
.criteria span{background:white;border:1px solid var(--line);padding:5px 11px;border-radius:5px}.card{background:white;border:1px solid var(--line);border-radius:8px;margin:18px 0;padding:22px}
.stats{display:flex;gap:32px;flex-wrap:wrap}.stats b{font-size:27px;font-weight:650;display:block;line-height:1.2}.stats span{color:var(--muted);font-size:13px}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th{text-align:left;color:var(--muted);font-size:12px;font-weight:600;white-space:nowrap}
td,th{border-bottom:1px solid var(--line);padding:13px 10px;vertical-align:middle}th:first-child,td:first-child{padding-left:0;min-width:205px}.numeric{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.increased{color:#9b421b}tr.selected{background:#f0f6fa}.case-button{border:0;background:none;color:var(--accent);font:inherit;font-weight:600;text-align:left;padding:0;cursor:pointer;text-decoration:underline;text-underline-offset:3px}
.case-button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid #73b9e6;outline-offset:4px}.status{font-size:12px}.viewer-head{display:flex;justify-content:space-between;gap:20px;align-items:start;flex-wrap:wrap}
label{display:block;color:var(--muted);font-size:12px;margin-bottom:5px}select{max-width:100%;padding:8px 10px;border:1px solid #a8bac7;border-radius:4px;background:white;color:var(--ink);font:inherit}
a{color:var(--accent)}.actions{display:flex;gap:8px 16px;flex-wrap:wrap;margin:14px 0}.actions a{font-size:13px}.viewer-frame{display:block;width:100%;height:1000px;border:1px solid var(--line);border-radius:4px;background:white}
.note{font-size:13px;color:var(--muted);max-width:1050px}.footer{font-size:12px;color:var(--muted)}.filters{margin:12px 0}.empty{padding:22px;background:#f0f6fa;border-radius:6px}[hidden]{display:none!important}@media(max-width:700px){main{padding:22px 12px}.card{padding:14px}.stats{gap:20px}.viewer-frame{height:760px}}
</style></head><body><main>
<header><div class="eyebrow">Three-photon bead experiment</div><h1>__TITLE__</h1>
<p class="muted">Select an acquisition to play its saved optimizer history and inspect the estimate against the imposed wavefront.</p>
<div class="criteria">__CRITERIA__</div></header>
<section class="card stats" aria-label="Batch summary"><div><b>__READY__ / __TOTAL__</b><span>replays ready</span></div><div><b>__CONVERGED__</b><span>optimizer converged</span></div><div><b>__IMAGES__</b><span>retained fitting images</span></div></section>
<section class="card"><h2>Fit summaries</h2><p class="note">RMS columns show wavefront error in waves relative to the imposed coefficients saved in metadata. Native aberration is assumed corrected by the fixed system AO. Brown values indicate a final error larger than the zero-estimate starting error.</p>
<div class="filters"><label for="resolution-select">Image resolution</label><select id="resolution-select"><option value="all">All resolutions</option>__RESOLUTIONS__</select></div>
<div class="table-wrap"><table><thead><tr><th>Acquisition</th><th>Status</th><th class="numeric">Resolution</th><th class="numeric">Images</th><th class="numeric">At Z __Z_LABELS__</th><th class="numeric">Solve time</th><th class="numeric">Updates</th><th class="numeric">Initial RMS</th><th class="numeric">Final RMS</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p class="note">__NOTE__</p>__REVIEW_LINK__</section>
<p class="empty" id="empty-state" __EMPTY_HIDDEN__>No downloaded replay is available for this resolution yet. Collect finished results and rebuild this page using the batch collection command.</p>
<section class="card" id="viewer" __VIEWER_HIDDEN__><div class="viewer-head"><div><h2 id="viewer-title">Optimizer replay</h2><p id="viewer-summary" class="muted"></p></div>
<div><label for="case-select">Acquisition</label><select id="case-select">__OPTIONS__</select></div></div>
<div class="actions" id="downloads"></div>
<iframe id="replay-frame" class="viewer-frame" title="Saved optimizer replay" src="__FIRST_REPLAY__"></iframe>
<p class="note">The established viewer shows imposed phase, estimated imposed phase, estimation error, a retained measured frame, optimizer loss and RMS error. Playback advances through saved iterations. Use the controls inside the viewer to play or scrub; open the replay separately for a larger view.</p></section>
<footer class="footer">Local results snapshot · Generated __GENERATED__ · <a href="site_data.json">Summary data</a></footer>
</main><script type="application/json" id="site-data">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('site-data').textContent);
const selector=document.getElementById('case-select');
const resolution=document.getElementById('resolution-select');
function choose(index){
 const row=data.cases[index];if(!row||!row.ready)return;
 selector.value=String(index);document.getElementById('viewer-title').textContent=row.label;
 document.getElementById('viewer-summary').textContent=`Job ${row.job_id} · ${row.images} images · ${row.status} · ${row.seconds.toFixed(1)} seconds`;
 const frame=document.getElementById('replay-frame');if(frame.getAttribute('src')!==row.replay)frame.src=row.replay;
 frame.title='Optimizer replay: '+row.label;
 const downloads=document.getElementById('downloads');downloads.replaceChildren();
 for(const [label,path] of [['Open replay separately',row.replay],...Object.entries(row.assets)]){
  const link=document.createElement('a');link.textContent=label;link.href=path;link.target='_blank';link.rel='noopener';downloads.appendChild(link);
 }
 for(const button of document.querySelectorAll('[data-case]'))button.setAttribute('aria-pressed',String(Number(button.dataset.case)===index));
 for(const tr of document.querySelectorAll('[data-row]'))tr.classList.toggle('selected',Number(tr.dataset.row)===index);
}
selector.addEventListener('change',()=>choose(Number(selector.value)));
for(const button of document.querySelectorAll('[data-case]'))button.addEventListener('click',()=>choose(Number(button.dataset.case)));
function filterResolution(){
 const size=resolution.value;
 for(const tr of document.querySelectorAll('[data-row]'))tr.hidden=size!=='all'&&tr.dataset.size!==size;
 const available=data.cases.map((row,index)=>({row,index})).filter(x=>x.row.ready&&(size==='all'||String(x.row.size)===size));
 const old=Number(selector.value);selector.replaceChildren();
 for(const item of available){const option=document.createElement('option');option.value=String(item.index);option.textContent=item.row.label;selector.appendChild(option);}
 document.getElementById('viewer').hidden=available.length===0;
 document.getElementById('empty-state').hidden=available.length>0;
 if(available.length)choose(available.some(x=>x.index===old)?old:available[0].index);
}
resolution.addEventListener('change',filterResolution);
filterResolution();
</script></body></html>'''
    title='Binned bead results · 64 × 64 and 32 × 32' if binned else f'Aberration estimates · SNR ≥ {threshold:g}'
    title=summary.get('title',title)
    criteria=(["All valid acquired Z planes","No SNR filtering","64 × 64 and 32 × 32 images","CUDA QR"] if binned else
              ["Requested Z: "+', '.join(f'{z:+g}' for z in z_values)+' µm',
               f'Peak / background noise ≥ {threshold:g}',f'{resolutions[0]} × {resolutions[0]} images','CUDA QR'])
    criteria=summary.get('criteria',criteria)
    note=('Images are block means of the raw signed acquisition. FOV and intensity units are preserved. '
          'Only the previously flagged bad plane 05 in the no-aberration acquisition is excluded; its +2 µm count is zero.' if binned else
          'SNR is measured on the saved averaged image: brightest 3 × 3 patch above median edge background, divided by robust edge noise. Counts show the images actually retained at each requested Z.')
    note=summary.get('note',note)
    replacements={'__READY__':str(len(ready)),'__TOTAL__':str(len(cases)),
        '__TITLE__':html.escape(title),'__CRITERIA__':''.join('<span>'+html.escape(x)+'</span>' for x in criteria),
        '__RESOLUTIONS__':''.join(f'<option value="{size}">{size} × {size}</option>' for size in resolutions),
        '__Z_LABELS__':html.escape(' / '.join(f'{z:+g}' for z in z_values)),'__NOTE__':html.escape(note),
        '__REVIEW_LINK__':'<p><a href="index.html">Review included/excluded images and every SNR score</a></p>' if (HERE/'index.html').is_file() else '',
        '__EMPTY_HIDDEN__':'hidden' if ready else '', '__VIEWER_HIDDEN__':'' if ready else 'hidden',
        '__CONVERGED__':str(sum(row.get('converged',False) for row in cases)),
        '__IMAGES__':str(sum(row['images'] for row in cases)),
        '__ROWS__':''.join(rows),'__OPTIONS__':options,
        '__FIRST_REPLAY__':html.escape(cases[first]['replay']) if first is not None else 'about:blank',
        '__GENERATED__':html.escape(data['generated_at']),'__DATA__':payload,
        '__FIRST__':str(first) if first is not None else '-1'}
    for key,value in replacements.items():document=document.replace(key,value)
    (HERE/'replays.html').write_text(document)
    print(f'{len(ready)}/{len(cases)} replay links prepared: '+str(HERE/'replays.html'))
    return HERE/'replays.html'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    build_site(parser.parse_args().batch)


if __name__=='__main__':main()
